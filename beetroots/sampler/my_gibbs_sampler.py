r"""Contains a class of sampler used in the Meudon PDR code Bayesian inversion problems
"""
import copy
from typing import Optional, Tuple, Union

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from scipy.special import softmax
from scipy.stats import norm
from tqdm.auto import tqdm

from beetroots.modelling.component_distribution import ComponentDistribution
from beetroots.modelling.target_distribution.abstract_target_distribution import TargetDistribution
from beetroots.sampler.abstract_sampler import Sampler
from beetroots.sampler.saver.my_gibbs_saver import MyGibbsSaver
from beetroots.sampler.utils.mml import EBayesMMLELogRate
from beetroots.sampler.utils.sampler_params import MyGibbsSamplerParams


class MyGibbsSampler(Sampler):
    r"""Defines a variant of the sampler proposed in :cite:t:`paludEfficientSamplingNon2023` that randomly combines two transition kernels :

    1. a independent MTM-chromatic Gibbs transition kernel
    2. a position-dependent MALA transition kernel with the RMSProp pre-conditioner

    In this variant, we replace the approximate likelihood by a hierarchical approach introducing an auxiliary variable. We sample the two sets of variables (physical parameters and auxiliary variables) using a Gibbs sampler.
    This implementation is still more general as it allows to sample from any number of variables (not only two). If the number of variables is one, it falls back to the original implementation.  
    """

    def __init__(
        self,
        my_gibbs_sampler_params: MyGibbsSamplerParams,
        D: int,
        L: int,
        N: int,
        rng: xp.random.Generator = xp.random.default_rng(42),
    ):
        r"""

        Parameters
        ----------
        my_sampler_params : MySamplerParams
            contains the main parameters of the algorithm
        D : int
            total number of physical parameters to reconstruct
        L : int
            number of observables per component :math:`n`
        N : int
            total number of pixels to reconstruct
        rng : numpy.random.Generator, optional
            random number generator (for reproducibility), by default xp.random.default_rng(42)
        """

        self.proposal_distribution_mtm = my_gibbs_sampler_params.proposal_distributions_mtm
        r"""dict[function]: function to generate random start values for the variables"""

        self.generate_random_start_type = my_gibbs_sampler_params.generate_random_start_type
        r"""dict[str]: type of random start values for the variables if no indicator prior (Standard or half-standard)"""

        # P-MALA params
        # ! redefine size of params
        self.eps0 = my_gibbs_sampler_params.initial_step_size
        r"""dict[float]: step size used in the Position-dependent MALA transition kernel, denoted :math:`\epsilon > 0` in the article"""

        self.lambda_ = my_gibbs_sampler_params.extreme_grad
        r"""dict[float]: limit value that avoids division by zero when computing the RMSProp preconditioner, denoted :math:`\eta > 0` in the article"""

        self.alpha = my_gibbs_sampler_params.history_weight
        r"""dict[float]: weight of past values of :math:`v` in the exponential decay (cf RMSProp preconditioner), denoted :math:`\alpha \in ]0,1[` in the article"""

        # MTM params
        # assert xp.isclose(
        #     int(pow(my_gibbs_sampler_params.k_mtm, 1 / D)) ** D, my_gibbs_sampler_params.k_mtm
        # ), "number of candidates for mtm needs to have an integer D-root"
        self.k_mtm = my_gibbs_sampler_params.k_mtm
        r"""dict[int]: number of candidates in the MTM kernel, denoted :math:`K` in the article"""

        self.compute_correction_term = my_gibbs_sampler_params.compute_correction_term
        r"""bool: wether or not to use the correction term (denoted :math:`\gamma` in the article) during the sampling (only used if `is_stochastic=True`)"""

        # overall
        self.selection_probas = my_gibbs_sampler_params.selection_probas
        r"""Union[dict[xp.ndarray], dict[List[float]]]: vector of selection probabilities for the MTM and PMALA kernels, respectively, i.e., :math:`[p_{MTM}, 1 - p_{MTM}]`"""

        self.stochastic = my_gibbs_sampler_params.is_stochastic
        r"""bool: if True, the algorithm performs sampling, and optimization otherwise"""

        self.compute_derivatives_2nd_order = (
            self.stochastic and self.compute_correction_term
        )
        r"""bool: wether to compute the expensive second order derivatives terms. Only true when the sampler runs a Markov chain (2nd order never used in optimization) and when the correction term denoted :math:`\gamma` is to be computed."""

        self.model_checking_component_name = my_gibbs_sampler_params.model_checking_component_name
        r"""str: name of the component to be used for model checking"""

        self.model_checking_component = None
        r"""ComponentDistribution: component to be used for model checking"""

        self.sample_variables = my_gibbs_sampler_params.sample_variables
        r"""dict[bool]: wether to sample the variables or not"""

        self.pmala_positivity_constraint = my_gibbs_sampler_params.pmala_positivity_constraint
        r"""dict[bool]: wether to enforce positivity constraint on the variables when using the PMALA kernel"""

        self.pmala_add_perturbation = my_gibbs_sampler_params.pmala_add_perturbation
        r"""dict[bool]: wether to add a perturbation to the PMALA proposal. Mainly used for debugging purposes and to validate approaches."""

        self.D = D
        r"""int: total number of physical parameters to reconstruct"""
        self.L = L
        r"""int: number of observables per component :math:`n`"""
        self.N = N
        r"""int: total number of pixels to reconstruct"""

        self.rng = rng
        r"""numpy.random.Generator: random number generator (for reproducibility)"""

        # initialization values, not to be kept during sampling
        self.v = {}
        r"""dict: RMSProp gradient variance vector for each target distribution, denoted :math:`v` in the article"""

        self.current = {}
        r"""dict: contains all the data about the current iterate (including the evaluations of the forward map and derivatives, etc.)"""

        self.j_t = {}
        r"""dict: contains the number of iterations since last acceptance for each target distribution"""


    # TODO: implement the model checking for the hierarchical approach and remove this old version.
    def _update_model_check_values(
        self,
        dict_model_check: dict,
        likelihood: ComponentDistribution,
        nll_full: xp.ndarray,
        objective: float,
    ) -> dict:
        count_pval = dict_model_check["count_pval"] * 1
        y_copy = likelihood.y * 1

        if self.stochastic:
            dict_model_check["clppd_online"] *= count_pval / (count_pval + 1)
            dict_model_check["clppd_online"] += xp.exp(-nll_full) / (count_pval + 1)

            y_rep = likelihood.sample_observation_model(
                self.rng,
            )
            likelihood_rep = copy.deepcopy(likelihood)
            likelihood_rep.y = y_rep * 1

            assert xp.allclose(likelihood.y, y_copy), "nooooo"

            likelihood_rep.evaluate_all_nlpdf_utils(
                self.current,
                idx_pix=None,
                compute_derivatives=False,
                compute_derivatives_2nd_order=False,
                mtm=False
            )
            nll_y_rep_full = likelihood_rep.neglog_pdf(
                full=True,
            )

            # p-value per (N, L) with y_rep_{n,ell} <= y_{n,ell}
            dict_model_check["p_values_y"] *= count_pval / (count_pval + 1)
            dict_model_check["p_values_y"] += (y_rep <= likelihood.y) / (count_pval + 1)

            # p-value per (N,) with
            # p(y_rep_n \vert theta_n) <= p(y_n \vert theta_n)
            nll_y = xp.sum(nll_full, axis=1)  # (N,)
            nll_y_rep = xp.sum(nll_y_rep_full, axis=1)  # (N,)

            dict_model_check["p_values_llh"] *= count_pval / (count_pval + 1)
            dict_model_check["p_values_llh"] += (nll_y_rep >= nll_y) / (count_pval + 1)

            dict_model_check["count_pval"] += 1

        else:
            if objective < dict_model_check["best_objective"]:
                dict_model_check["best_objective"] = objective * 1
                dict_model_check["clppd_online"] = xp.exp(-nll_full)

            # p-values are computed at the end of the optimisation process.

        return dict_model_check

    def _finalize_model_check_values(
        self,
        dict_model_check: dict,
    ) -> dict:
        
        # this p-value should be between 0 and 0.5
        dict_model_check["p_values_y"] = xp.where(
            dict_model_check["p_values_y"] > 0.5,
            1 - dict_model_check["p_values_y"],
            dict_model_check["p_values_y"],
        )

        return dict_model_check

    def initialize_model_check_dict(
            self,
        ) -> dict:

        # clppd = computed log point-wise predictive density.
        # if self.stochastic : avg of all pred. likelihood terms (with burn-in)
        # but burn-in values are negligible (0) compared to non burn-in
        # else : predictive likelihood with best param theta
        clppd_online = xp.zeros((self.N, self.L))
        # utilitary variables
        best_objective = xp.infty  # used only if not self.stochastic
        count_pval = 0  # used only if self.stochastic

        # p(y^{rep}_\ell <= y_\ell | y)
        p_values_y = xp.zeros((self.N, self.L))
        # p(y^{rep}_\ell \in [ q_{25\%}(y_\ell), q_{75\%}(y_\ell) ] | y)
        p_values_llh = xp.zeros((self.N,))

        dict_model_check = {
            "clppd_online": clppd_online,
            "best_objective": best_objective,
            "count_pval": count_pval,
            "p_values_y": p_values_y,
            "p_values_llh": p_values_llh,
        }
        return dict_model_check

    def sample(
        self,
        target_distributions: dict[str, TargetDistribution],
        saver: MyGibbsSaver,
        max_iter: int,
        Vars_0: dict[str, Union[None, xp.ndarray]],
        disable_progress_bar: bool = False,
        #
        regu_spatial_N0: Union[int, float] = xp.infty, # sets to infinity by default => no optimization of reg. params.
        regu_spatial_scale: float = 1.0,
        regu_spatial_vmin: float = 1e-8,
        regu_spatial_vmax: float = 1e8,
        #
        T_BI: int = 0,  # used only for clppd
    ) -> None:
        r"""main method of the class, runs the sampler

        Parameters
        ----------
        target_distributions : dict[str, TargetDistribution]
            dict containing the probability distributions to be sampled (key being the name of the variable)
        saver : Saver
            object responsible for progressively saving the Markov chain data during the run
        max_iter : int
            total duration of a Markov chain
        vars_0 : Optional[dict[str, xp.ndarray]], optional
            starting point for each variable, by default None
        disable_progress_bar : bool, optional
            wether to disable the progress bar, by default False
        regu_spatial_N0 : Union[int, float], optional
            number of iterations defining the initial update phase (for spatial regularization weight optimization). xp.infty means that the optimization phase never starts, and that the weight optimization is not applied. by default xp.infty
        regu_spatial_scale : Optional[float], optional
            scale parameter involved in the definition of the projected gradient
            step size (for spatial regularization weight optimization). by default 1.0
        regu_spatial_vmin : Optional[float], optional
            lower limit of the admissible interval (for spatial regularization weight optimization), by default 1e-8
        regu_spatial_vmax : Optional[float], optional
            upper limit of the admissible interval (for spatial regularization weight optimization), by default 1e8
        T_BI : int, optional
            duration of the `Burn-in` phase, by default 0
        """
        var_names = list(target_distributions.keys())

        # gets the component distributions for model checking.
        all_components = dict()
        for target_name in var_names:
            all_components.update(target_distributions[target_name].distribution_components)
        self.model_checking_component = all_components[self.model_checking_component_name]

        # FIXME: temporary solution for Vars_0. We should have a dict with MAP, MLE or None for each variable.
        # assert len(target_distributions) == len(Vars_0), "Vars_0 should have the same number of variables as target_distributions. If no initial value is provided for a variable it should be set to None in the Vars_0 dict."
        # assert set(target_distributions.keys()) == set(Vars_0.keys()), "Vars_0 should have the same variable names (keys of dict) as target_distributions"
        self.current = {key: dict() for key in var_names}
        for key in var_names:
            if Vars_0[key] is None:
                self.current[key]["var"] = self.generate_random_start_Var(target_distributions[key], type=self.generate_random_start_type[key])  # (N, D)
            else:
                self.current[key]["var"] = Vars_0[key] * 1

        for current_var_dict in self.current.values():
            assert None not in current_var_dict['var'], "Vars_0 should not contain None values after generating random start values"
        
        # TODO: change the following to check shape of initial values
        # assert Theta_0.shape == (self.N, self.D)

        additional_sampling_log = {}
        dict_objective = {} #TODO and #FIXME: remove in the final Gibbs sampler. Still need to find a way for the model checking.
        for key in var_names:
            self.current[key] = target_distributions[key].compute_all(
                self.current, compute_derivatives_2nd_order=self.compute_derivatives_2nd_order
            )
            assert xp.isnan(self.current[key]["objective"]) == 0
            assert xp.sum(xp.isnan(self.current[key]["grad"])) == 0

            self.v[key] = self.current[key]["grad"].flatten() ** 2 # the variable v is used in the RMSProp preconditioner of the PMALA kernel. No alpha is applied here since it is the first iteration.
            assert xp.sum(xp.isnan(self.v[key])) == 0.0
            
            # Initialize the j_t variable used in the correction term of the PMALA kernel. It represents the number of iterates since last acceptance.
            self.j_t[key] = xp.zeros(target_distributions[key].var_shape)
            additional_sampling_log[key] = dict()
            dict_objective[key] = dict() 

        rng_state_array, _ = self.get_rng_state()

        regu_weights_optimizer = EBayesMMLELogRate(
            scale=regu_spatial_scale,
            N0=regu_spatial_N0,
            N1=+xp.infty,
            dim=self.D * self.N,
            vmin=regu_spatial_vmin,
            vmax=regu_spatial_vmax,
            homogeneity=2.0,
            exponent=0.8,
        )
        optimize_regu_weights = regu_weights_optimizer.N0 < xp.infty # the condition for optimization is simply based on the the input parameter regu_spatial_N0

        dict_model_check = self.initialize_model_check_dict()
        acceptance_stats = {}

        for t in tqdm(range(1, max_iter + 1), disable=disable_progress_bar):
            
            # TODO: implement the optimization of the spatial regularization weights for hierarchical approach.
            # --- REGULARIZATION WEIGHTS OPTIMIZATION ---
            # if optimize_regu_weights and (self.N > 1):
            #     assert target_distribution.prior_spatial is not None

            #     if t >= regu_weights_optimizer.N0:
            #         tau_t = self.sample_regu_hyperparams(
            #             target_distribution,
            #             regu_weights_optimizer,
            #             t,
            #             self.current["Theta"] * 1,
            #         )

            #         target_distribution.prior_spatial.weights = tau_t * 1

            #         # recompute target_distribution neg log pdf and gradients with
            #         # new spatial regularization parameter
            #         self.current = target_distribution.compute_all(
            #             self.current["Theta"],
            #             self.current["forward_map_evals"],
            #             self.current["nll_utils"],
            #             compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
            #         )

            #     additional_sampling_log["tau"] = target_distribution.prior_spatial.weights * 1
            # ------


            # --- RANDOM CHOICE PMALA / MTM ---
            for key in var_names:
                type_t = xp.argmax(
                self.rng.multinomial(
                    1,
                    pvals=list(self.selection_probas[key]),
                    )
                )
                if not self.sample_variables[key]:
                    acceptance_stats[key] = {"accepted_t": 1, "log_proba_accept_t": 0.0, "type_t": 0} # kernel choice 0 = MTM #FIXME: not ideal, no kernel is used so remove MTM
                    continue

                if type_t == 0:
                    (
                        accepted_t,
                        log_proba_accept_t,
                    ) = self.generate_new_sample_mtm(t, key, target_distributions[key])
                    acceptance_stats[key] = {"accepted_t": accepted_t, "log_proba_accept_t": log_proba_accept_t, "type_t": 0} # kernel choice 0 = MTM
                else:
                    assert type_t == 1
                    (
                        accepted_t,
                        log_proba_accept_t,
                    ) = self.generate_new_sample_pmala_rmsprop(t, key, target_distributions[key])
                    acceptance_stats[key] = {"accepted_t": accepted_t, "log_proba_accept_t": log_proba_accept_t, "type_t": 1} # kernel choice 1 = PMALA


            # * if the memory is empty : initialize it
            nll_full = xp.zeros((self.N, self.L,))
            if saver.memory == {}:
                for key in var_names:
                    additional_sampling_log[key]["v"] = self.v[key].reshape(self.current[key]["grad"].shape) * 1
                    additional_sampling_log[key]["type_t"] = acceptance_stats[key]["type_t"]
                    additional_sampling_log[key]["accepted_t"] = acceptance_stats[key]["accepted_t"]
                    additional_sampling_log[key]["log_proba_accept_t"] = acceptance_stats[key]["log_proba_accept_t"]

                    dict_objective[key], nll_full_temp = target_distributions[key].compute_all_for_saver( # Update nll_full
                        self.current,
                        self.model_checking_component_name
                    )
                    nll_full += nll_full_temp
                
                components = set([component for target_distribution in target_distributions.values() for component in target_distribution.distribution_components.values()])
                objective_true = 0
                for comp in components:
                    # FIXME: find a way toa void recomuting utils here. Too costly.
                    comp.evaluate_all_nlpdf_utils(current=self.current, idx_pix=None, compute_derivatives_2nd_order=False, compute_derivatives=False, mtm=False)
                    objective_true += comp.neglog_pdf()
                dict_objective['objective_global'] = objective_true * 1

                if t > T_BI:
                    dict_model_check = self._update_model_check_values(
                        dict_model_check,
                        self.model_checking_component,
                        nll_full,
                        objective=None,) # Objective is used for optimization (self.stochastic == False)
                    
                saver.initialize_memory(
                    max_iter,
                    t,
                    current=self.current,
                    nlpdf_utils={var_name: dict() for var_name in var_names}, # FIXME: to change if we to store variables related to some component distributions. Previously it was used to keep track of parameters from the approximate likelihood.
                    dict_objective=dict_objective,
                    additional_sampling_log=additional_sampling_log,
                    sample_variables=self.sample_variables,
                )

                rng_state_array, rng_inc_array = self.get_rng_state()

                saver.update_memory(
                    t,
                    current=self.current,
                    nlpdf_utils={var_name: dict() for var_name in var_names}, # FIXME: to change if we to store variables related to some component distributions. Previously it was used to keep track of parameters from the approximate likelihood.
                    dict_objective=dict_objective,
                    additional_sampling_log=additional_sampling_log,
                    rng_state_array=rng_state_array,
                    rng_inc_array=rng_inc_array,
                    sample_variables=self.sample_variables,
                )

            elif saver.check_need_to_update_memory(t):
                for key in var_names:
                    additional_sampling_log[key]["v"] = self.v[key].reshape(self.current[key]["grad"].shape) * 1
                    additional_sampling_log[key]["type_t"] = acceptance_stats[key]["type_t"]
                    additional_sampling_log[key]["accepted_t"] = acceptance_stats[key]["accepted_t"]
                    additional_sampling_log[key]["log_proba_accept_t"] = acceptance_stats[key]["log_proba_accept_t"]

                    dict_objective[key], nll_full_temp = target_distributions[key].compute_all_for_saver(
                        self.current,
                        self.model_checking_component_name
                    )
                    nll_full += nll_full_temp
                
                components = set([component for target_distribution in target_distributions.values() for component in target_distribution.distribution_components.values()])
                objective_true = 0
                for comp in components:
                    # FIXME: find a way toa void recomuting utils here. Too costly.
                    comp.evaluate_all_nlpdf_utils(current=self.current, idx_pix=None, compute_derivatives_2nd_order=False, compute_derivatives=False, mtm=False)
                    objective_true += comp.neglog_pdf()
                dict_objective['objective_global'] = objective_true * 1

                if t > T_BI:
                    dict_model_check = self._update_model_check_values(
                        dict_model_check,
                        self.model_checking_component,
                        nll_full,
                        objective=None,) # Objective is used for optimization (self.stochastic == False)

                rng_state_array, rng_inc_array = self.get_rng_state()

                saver.update_memory(
                    t,
                    current=self.current,
                    nlpdf_utils= {var_name: dict() for var_name in var_names}, # FIXME: to change if we to store variables related to some component distributions. Previously it was used to keep track of parameters from the approximate likelihood.
                    dict_objective=dict_objective,
                    additional_sampling_log=additional_sampling_log,
                    rng_state_array=rng_state_array,
                    rng_inc_array=rng_inc_array,
                    sample_variables=self.sample_variables,
                )

            else:
                pass

            if saver.check_need_to_save(t):
                saver.save_to_file()

        # ---------
        dict_model_check = self._finalize_model_check_values(
            dict_model_check,
        )

        saver.save_additional(
            list_arrays=[
                dict_model_check["clppd_online"],
                dict_model_check["p_values_y"],
                dict_model_check["p_values_llh"],
            ],
            list_names=["clppd", "p-values-y", "p-values-llh"],
        )
        return

    def generate_new_sample_pmala_rmsprop(self, t: int, key: str, target_distribution: TargetDistribution):
        """generates a new sample using the position-dependent MALA transition kernel

        Parameters
        ----------
        t : int
            current iteration index
        key : str
            name of the variable to sample
        target_distibution : TargetDistribution
            negative log target distribution class

        Returns
        -------
        accepted : bool
            wether or not the candidate was accepted
        log_proba_accept : float
            log of the acceptance proba
        """
        
        # NOTE: the new_var update was done inside the loop. That is, an expensive call to compute_all of the target distribution was done for each sites group. It does not seem necessary to do it in this way so it has been moved outside the loop as it is done in the MTM kernel.

        new_var = self.current[key]["var"] * 1 # (N, D)

        accept_total = xp.zeros((new_var.shape[0],))
        log_proba_accept_total = xp.zeros((self.N,))

        # * prepare dict with other required variables for weights computation (other variables won't change so we declare it outside)
        current_candidate = {k: {'var': self.current[k]['var']} for k in self.current.keys()}          

        # * define proba of changing each pixel
        # * either uniformly or depending on their respective nll
        # if posterior.prior_spatial is not None:
        # n_sites = len(posterior.dict_sites)
        # idx_site = int(self.rng.integers(0, n_sites))

        list_idx = xp.array(list(target_distribution.dict_sites.keys()))

        for idx_site in list_idx:
            idx_pix = target_distribution.dict_sites[idx_site]
            n_pix = idx_pix.size

            # --- PROPOSAL STEP ---
            grad_t = self.current[key]["grad"][idx_pix] * 1
            v_current = self.v[key].reshape(new_var.shape)[idx_pix, :] * 1

            # generate random
            diag_G_t = 1 / (self.lambda_[key] + xp.sqrt(v_current))  # (n_pix, D)

            assert xp.all(
                diag_G_t > 0
            ), f"{diag_G_t}, {self.lambda_ + xp.sqrt(self.v[key])}, {self.v[key]}"
            
            size_var_sites = (n_pix, new_var.shape[1]) # Shape of the variable to sample restricted to the current sites for the first dimension.

            # bias correction term
            if self.compute_correction_term:
                hess_diag_t = self.current[key]["hess_diag"][idx_pix] * 1
                j_t = self.j_t[key].reshape(new_var.shape)[idx_pix, :] * 1

                correction = (
                    -(1 - self.alpha[key])
                    * self.alpha[key]**j_t
                    * (diag_G_t**2)
                    / xp.sqrt(v_current)
                    * grad_t
                    * hess_diag_t
                )  # (n_pix, D)

                n_inf = xp.sum(~xp.isfinite(correction))
                if n_inf > 0:
                    print(f"num of nan in correction term: {n_inf}")
                correction = xp.nan_to_num(correction)  # ? nécessaire ?
            else: # if no correction term then 0
                correction = xp.zeros(size_var_sites)

            # combination
            mu_current = (
                new_var[idx_pix, :]
                - self.eps0[key] / 2 * diag_G_t * grad_t
                + self.eps0[key] * correction
            )  # (n_pix, D)

            scale = xp.sqrt(self.eps0[key] * diag_G_t)
            if not self.pmala_positivity_constraint[key]:
                z_t = self.rng.standard_normal(size=size_var_sites)
                z_t *= scale  # (n_pix, D)
            else:
                u = self.rng.random(size=size_var_sites)
                Phi_0 = norm.cdf(xp.zeros(size_var_sites), loc=mu_current, scale=scale)
                u = u * (1 - Phi_0) + Phi_0
                z_t = norm.ppf(u, scale=scale)

            candidate = mu_current
            if self.pmala_add_perturbation[key]:
                """
                # Noise removed for trials. Should be added back for true sampling.
                # (n_pix, D), simulates Gaussian with mean mu_current and std dev sqrt(eps0[key] * diag_G_t). It is the PMALA proposal (Langevin step)
                """
                candidate += z_t  

            # --- ACCEPT/REJECT STEP ---
            # * compute log_q of candidate given current
            log_q_candidate_given_current = -1 / 2 * xp.sum(
                xp.log(diag_G_t), axis=1
            ) - 1 / (2 * self.eps0[key]) * xp.sum(
                (candidate - mu_current) ** 2 / diag_G_t, axis=1 
            )  # (n_pix,)

            shape_q = log_q_candidate_given_current.shape
            assert shape_q == (n_pix,), f"{shape_q}"

            # * compute log_q of current given candidate
            candidate_pixel = new_var * 1
            # candidate_pixel[idx_pix, :] = mu_current * 1 # TODO: reset this after debugging
            candidate_pixel[idx_pix, :] = candidate * 1

            current_candidate[key] = {'var': candidate_pixel}
            target_distribution.update_nlpdf_utils(current_candidate, idx_pix, compute_derivatives=True, compute_derivatives_2nd_order=self.compute_correction_term)
            candidate_all = {}
            candidate_all['objective_pix'] = target_distribution.neglog_pdf(pixelwise=True, update_nlpdf_utils=False, idx_pix=idx_pix)
            candidate_all['grad'] = target_distribution.grad_neglog_pdf(update_nlpdf_utils=False, idx_pix=idx_pix) # (N, D)
            candidate_all["hess_diag"] = target_distribution.hess_diag_neglog_pdf(update_nlpdf_utils=False, idx_pix=idx_pix) if self.compute_correction_term else None# (N, D)

            assert xp.isnan(candidate_all["objective_pix"]).sum() == 0
            assert xp.isnan(candidate_all["grad"]).sum() == 0

            grad_cand = candidate_all["grad"] * 1
            v_cand = (
                self.alpha[key] * v_current + (1 - self.alpha[key]) * grad_cand**2
            )  # (n_pix, D)
            diag_G_cand = 1 / (self.lambda_[key] + xp.sqrt(v_cand))  # (n_pix, D)

            if self.compute_correction_term:
                hess_diag_cand = candidate_all["hess_diag"] * 1

                correction_cand = -(
                    (1 - self.alpha[key])
                    * diag_G_cand**2
                    / (2*xp.sqrt(v_cand)) # NOTE: the factor 2 was missing in the initial approach. To be confirmed.
                    * grad_cand
                    * hess_diag_cand
                )
            else:
                correction_cand = xp.zeros(size_var_sites)

            mu_cand = (
                candidate
                - self.eps0[key] / 2 * diag_G_cand * grad_cand
                + self.eps0[key] * correction_cand
            )  # (n_pix, D)

            log_q_current_given_candidate = -1 / 2 * xp.sum(
                xp.log(diag_G_cand), axis=1
            ) - 1 / (2 * self.eps0[key]) * xp.sum(
                (new_var[idx_pix, :] - mu_cand) ** 2 / diag_G_cand, axis=1
            )  # (n_pix,)

            shape_q = log_q_current_given_candidate.shape
            assert shape_q == (n_pix,), f"{shape_q}"

            # * compute proba accept
            logpdf_current = -self.current[key]["objective_pix"][idx_pix]
            logpdf_candidate = -candidate_all["objective_pix"]

            shape_1 = logpdf_current.shape
            shape_2 = logpdf_candidate.shape
            assert shape_1 == (n_pix,), f"{shape_1}"
            assert shape_2 == (n_pix,), f"{shape_2}"

            log_proba_accept = (
                logpdf_candidate
                - logpdf_current
                + log_q_current_given_candidate
                - log_q_candidate_given_current
            )
            assert log_proba_accept.shape == (n_pix,)

            log_u = xp.log(self.rng.uniform(0, 1, size=n_pix))
            accept_arr = log_u < log_proba_accept

            new_var[idx_pix, :] = xp.where(
                accept_arr[:, None] * xp.ones(size_var_sites),
                candidate,  # (n_pix, D)
                new_var[idx_pix, :],  # (n_pix, D)
            )

            accept_total[idx_pix] = accept_arr * 1
            log_proba_accept_total[idx_pix] = log_proba_accept * 1

            # update v and j
            v = self.v[key].reshape(new_var.shape) * 1
            v[idx_pix, :] = v_cand * 1
            self.v[key] = v.flatten()

            j = self.j_t[key].reshape(new_var.shape) * 1
            j[idx_pix, :] = xp.where(
                accept_arr[:, None],
                0.0,  # reset to 0 if accept
                j[idx_pix, :] + 1,  # else add 1
            )
            self.j_t[key] = j.flatten()

            if accept_arr.max() > 0:  # if at least one accept and not the last site group
                # update the current variable for the next site group
                self.current[key]['var'] = new_var * 1
                self.current[key] = target_distribution.compute_all(
                    self.current,
                    compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
                )

        # after loop
        return accept_total.mean(), log_proba_accept_total.mean()

    def generate_new_sample_mtm(
        self, t: int, key: str, target_distribution: TargetDistribution  # , idx_site: Union[int, None] = None
    ):
        r"""generates a new sample using the MTM transition kernel

        Parameters
        ----------
        t : int
            current iteration index
        key : str
            name of the variable to sample
        target_distibution : TargetDistribution
            negative log target distribution class

        Returns
        -------
        accepted : bool
            wether or not the candidate was accepted
        log_proba_accept : float
            log of the acceptance proba
        """
        new_var = self.current[key]["var"] * 1  # (N, D) in general

        accept_total = xp.zeros((new_var.shape[0],)) # (N, ) in general
        log_rg_total = xp.zeros((new_var.shape[0],)) # (N, ) in general

        current_candidates = {k: {'var': xp.repeat(self.current[k]["var"][:, xp.newaxis], self.k_mtm[key] + 1, axis=1)} for k in self.current.keys()} # Add a dimension for the candidates.                

        # * define proba of changing each pixel
        # * either uniformly or depending on their respective nll
        # if posterior.prior_spatial is not None:
        # n_sites = len(posterior.dict_sites)
        # idx_site = int(self.rng.integers(0, n_sites))

        # TODO: generalize the use of dict_sites to all variables. Is it always present in the target_distribution?
        list_idx = xp.array(list(target_distribution.dict_sites.keys()))

        for idx_site in list_idx:
            idx_pix = target_distribution.dict_sites[idx_site]
            n_pix = idx_pix.size

            # * generate and evaluate candidates
            candidates = new_var * 1
            candidates = candidates.reshape((new_var.shape[0], 1, *new_var.shape[1:])).repeat(self.k_mtm[key] + 1, axis=1)  # (N, k_mtm, ...)
            candidates[idx_pix, :-1, ...] = self.proposal_distribution_mtm[key].sample(
                new_var, idx_pix, self.k_mtm[key], self.rng,
            )

            # --- COMPUTE WEIGHTS (USING LOG)
            # Compute neglogpdf of candidates

            current_candidates[key] = {'var': candidates}
            
            target_distribution.update_nlpdf_utils(current_candidates, idx_pix, compute_derivatives=False, compute_derivatives_2nd_order=False, mtm=True) # TODO: add the possibility to pass the idx_pix to the neglog_pdf method, improves the performance.
            neglogpdf_candidates = target_distribution.neglog_pdf(pixelwise=True, update_nlpdf_utils=False) # TODO: check if the shape corresponds to the expected one # We don't call the nlpdf_utils here but above since we want to add the mtm=True argument to the update_nlpdf_utils method.
            assert neglogpdf_candidates.shape == (n_pix, self.k_mtm[key] + 1,)
            assert xp.isnan(neglogpdf_candidates).sum() == 0

            # candidates_pix = candidates_pix.reshape((n_pix, self.k_mtm + 1, self.D))
            # assert xp.allclose(candidates_pix[:, -1, :], self.current["Theta"][idx_pix, :]) -> validated

            neglogpdf_proposal_candidates = self.proposal_distribution_mtm[key].neglog_pdf(candidates, idx_pix)

            shape_ = neglogpdf_proposal_candidates.shape
            assert shape_ == (n_pix, self.k_mtm[key] + 1)

            neglogpdf_candidates -= neglogpdf_proposal_candidates # FIXME: originally it was a + instead of the minus but the proposal is in the demoninator in the weight ratio. Check if it is correct.

            neglogpdf_candidates_min = xp.amin(
                neglogpdf_candidates, axis=1, keepdims=True
            )
            neglogpdf_candidates -= neglogpdf_candidates_min

            pdf_candidates = xp.exp(-neglogpdf_candidates)  # (n_pix, k_mtm)

            log_numerators = xp.log(xp.sum(pdf_candidates[:, :-1], axis=1))
            # log_numerators = xp.where(
            #     xp.isinf(log_numerators), -1e15, log_numerators
            # )

            assert log_numerators.shape == (n_pix,), log_numerators.shape
            # assert xp.sum(1 - xp.isfinite(log_numerators)) == 0, log_numerators

            # * choose challenger candidate
            weights = softmax(-neglogpdf_candidates[:, :-1], axis=1)
            assert xp.sum(1 - xp.isfinite(weights)) == 0, weights

            idx_challengers = xp.zeros((n_pix,), dtype=int)
            for i in range(n_pix):
                idx_challengers[i] = self.rng.choice(
                    self.k_mtm[key],
                    p=weights[i],
                )

            challengers = candidates[ #FIXME: check if this is correct. I have chanegd it to match the fact that our candidates array cover all pixels not only the ones in idx_pix. Compare with the original code.
                idx_pix, idx_challengers, :
            ]  # (n_pix, D)
            neglogpdf_challengers = neglogpdf_candidates[
                xp.arange(n_pix), idx_challengers
            ]

            shape_ = neglogpdf_challengers.shape
            assert shape_ == (n_pix,), shape_

            # * denominator
            log_denominators = xp.log(
                xp.sum(pdf_candidates, axis=1) - xp.exp(-neglogpdf_challengers)
            )
            # log_denominators = xp.where(
            #     xp.isinf(log_denominators), -1e15, log_denominators
            # )

            shape_ = log_denominators.shape
            assert shape_ == (n_pix,), shape_

            # assert xp.sum(1 - xp.isfinite(log_numerators)) == 0, log_numerators
            # assert xp.sum(1 - xp.isfinite(log_denominators)) == 0, log_denominators

            # * accept-reject
            log_rg = log_numerators - log_denominators
            log_rg = xp.where(
                xp.isfinite(log_rg), log_rg, 1e-15
            )  # if either log_numerators or log_denominators is not finite, do not accept

            log_u = xp.log(self.rng.uniform(0, 1, size=n_pix))
            accept_arr = log_u < log_rg

            new_var[idx_pix, :] = xp.where(
                accept_arr[:, None] * xp.ones((n_pix, *new_var.shape[1:])),
                challengers,  # (n_pix, *var_shape[1:])
                candidates[idx_pix, -1, :],  # (n_pix, *var_shape[1:])
            )

            accept_total[idx_pix] = accept_arr * 1
            log_rg_total[idx_pix] = log_rg * 1

            # * re-initialize j for new point
            new_j_t = self.j_t[key].reshape(new_var.shape) * 1
            new_j_t[idx_pix, :] = xp.where(
                accept_arr[:, None], 0.0, new_j_t[idx_pix, :]
            )
            self.j_t[key] = new_j_t.flatten()

        # *------
        # * once all sites have been dealt with, update global parameters
        if accept_total.max() > 0:  # if at least one accept
            self.current[key]['var'] = new_var * 1

            self.current[key] = target_distribution.compute_all(
                self.current,
                compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
            )

            new_v = self.v[key].reshape(new_var.shape) * 1
            new_v = xp.where(
                accept_total[:, None],
                self.alpha[key] * new_v + (1 - self.alpha[key]) * self.current[key]["grad"] ** 2,
                new_v,
            )
            self.v[key] = new_v.flatten()
            assert xp.sum(xp.isnan(self.v[key])) == 0.0
            assert xp.sum(xp.isinf(self.v[key])) == 0.0
        else:
            self.current[key] = target_distribution.compute_all(
                self.current,
                compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
            )

        return xp.mean(accept_total), xp.mean(log_rg_total)
