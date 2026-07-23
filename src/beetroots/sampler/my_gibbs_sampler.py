r"""Contains a class of sampler used in the Meudon PDR code Bayesian inversion problems"""

import copy
from typing import Dict, List, Union

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from tqdm.auto import tqdm

from beetroots.modelling.component_distribution import ComponentDistribution
from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.abstract_sampler import Sampler
from beetroots.sampler.kernels.abstract_kernel import Kernel, KernelUtils
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
        rng: xp.random.Generator,
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

        self.kernels_dic: Dict[
            str, Union[Dict[int, Kernel], Dict[str, Dict[int, Kernel]]]
        ] = my_gibbs_sampler_params.kernels_dic
        r"""dict[function]: function to generate random start values for the variables"""

        self.kernels_list: List[Union[KernelUtils, Kernel]] = list(
            self.kernels_dic["joint"].values()
        )
        for dic in self.kernels_dic["disjoint"].values():
            self.kernels_list += list(dic.values())

        self.compute_derivatives_2nd_order = (
            my_gibbs_sampler_params.compute_derivatives_2nd_order
        )
        r"""bool: if True, the algorithm computes the second order derivatives of the target distributions (might be required by some kernels, e.g. PMALA)"""

        self.selection_probas = my_gibbs_sampler_params.selection_probas
        r"""Union[dict[xp.ndarray], dict[List[float]]]: vector of selection probabilities for the different kernels"""

        self.is_stochastic = my_gibbs_sampler_params.is_stochastic
        r"""bool: if True, the algorithm performs sampling, and optimization otherwise"""

        self.model_checking_component_name = (
            my_gibbs_sampler_params.model_checking_component_name
        )
        r"""str: name of the component to be used for model checking"""

        self.param_interest_name = my_gibbs_sampler_params.param_interest_name
        r"""str: name of the physical parameter of interest"""

        self.model_checking_component = None
        r"""ComponentDistribution: component to be used for model checking"""

        self.sample_variables = my_gibbs_sampler_params.sample_variables
        r"""dict[bool]: wether to sample the variables or not"""

        self.rng = rng
        r"""numpy.random.Generator: random number generator (for reproducibility)"""

        self.D = D
        r"""int: total number of physical parameters to reconstruct"""
        self.L = L
        r"""int: number of observables per component :math:`n`"""
        self.N = N
        r"""int: total number of pixels to reconstruct"""

        self.current = {}
        r"""dict: contains all the data about the current iterate (including the evaluations of the forward map and derivatives, etc.)"""

    def _update_model_check_values(
        self,
        dict_model_check: dict,
        likelihood: ComponentDistribution,
        nll_full: xp.ndarray,
        objective: float,
    ) -> dict:
        # TODO: switch that inside likelihood objects (hierarchical likelihood does not fit here)
        count_pval = dict_model_check["count_pval"] * 1
        y_copy = likelihood.y * 1

        if self.is_stochastic:
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
                mtm=False,
            )
            nll_y_rep_full = likelihood_rep.neglog_pdf(
                full=True,
                chromatic_gibbs=False,
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
        likelihood: ComponentDistribution,
        nll_full: xp.ndarray,
    ) -> dict:
        if not self.is_stochastic:
            # optimization p-value computations on estimated \hat{\theta}
            for count_pval in tqdm(range(self.ESS_OPTIM)):
                y_rep = likelihood.sample_observation_model(
                    self.rng,
                )
                likelihood_rep = copy.deepcopy(likelihood)
                likelihood_rep.y = y_rep * 1
                likelihood_rep.evaluate_all_nlpdf_utils(
                    self.current,
                    idx_pix=None,
                    compute_derivatives=False,
                    compute_derivatives_2nd_order=False,
                    mtm=False,
                )
                nll_y_rep_full = likelihood_rep.neglog_pdf(
                    full=True,
                    chromatic_gibbs=False,
                )

                # p-value per (N, L) with y_rep_{n,ell} <= y_{n,ell}
                dict_model_check["p_values_y"] *= count_pval / (count_pval + 1)
                dict_model_check["p_values_y"] += (y_rep <= likelihood.y) / (
                    count_pval + 1
                )

                # p-value per (N,) with
                # p(y_rep_n \vert theta_n) <= p(y_n \vert theta_n)
                nll_y = xp.sum(nll_full, axis=1)  # (N,)
                nll_y_rep = xp.sum(nll_y_rep_full, axis=1)  # (N,)

                dict_model_check["p_values_llh"] *= count_pval / (count_pval + 1)
                dict_model_check["p_values_llh"] += (nll_y_rep >= nll_y) / (
                    count_pval + 1
                )

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
        # if self.is_stochastic : avg of all pred. likelihood terms (with burn-in)
        # but burn-in values are negligible (0) compared to non burn-in
        # else : predictive likelihood with best param theta
        clppd_online = xp.zeros((self.N, self.L))
        # utilitary variables
        best_objective = xp.inf  # used only if not self.is_stochastic
        count_pval = 0  # used only if self.is_stochastic

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
        regu_spatial_N0: Union[
            int, float
        ] = xp.inf,  # sets to infinity by default => no optimization of reg. params.
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
            number of iterations defining the initial update phase (for spatial regularization weight optimization). xp.inf means that the optimization phase never starts, and that the weight optimization is not applied. by default xp.inf
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
        self.all_components = dict()
        for target_name in var_names:
            self.all_components.update(
                target_distributions[target_name].distribution_components
            )
        self.model_checking_component = self.all_components[
            self.model_checking_component_name
        ]

        self.current = {key: dict() for key in var_names}
        for key in var_names:
            if (
                not isinstance(Vars_0[key], xp.ndarray)
                or not Vars_0[key].shape == target_distributions[key].var_shape
            ):
                raise ValueError(
                    "Vars_0 should contain initial values for all variables. It should be instantiated in the inverse problem specific class. Should be an array of the same shape as the variable to be sampled."
                )
            else:
                self.current[key]["var"] = Vars_0[key] * 1

        additional_sampling_log = {}
        dict_objective = {}

        for key in var_names:
            self.current[key] = target_distributions[key].compute_all(
                self.current,
                compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
            )
            assert xp.isnan(self.current[key]["objective_global"]) == 0
            assert xp.sum(xp.isnan(self.current[key]["grad"])) == 0

            additional_sampling_log[key] = dict()
            dict_objective[key] = dict()

        for kernel in self.kernels_list:
            kernel.initialize_hyperparameters(
                current=self.current,
            )

        comp_forward = None
        for comp in self.all_components.values():
            if getattr(comp, "forward_map_evals", None) is not None:
                comp_forward = comp
        assert comp_forward is not None

        rng_state_array, _ = self.get_rng_state()
        is_scalar = True
        regu_weights_optimizer = EBayesMMLELogRate(
            scale=regu_spatial_scale,
            N0=regu_spatial_N0,
            N1=+xp.inf,
            dim=self.N * self.D if is_scalar else self.N,
            vmin=regu_spatial_vmin,
            vmax=regu_spatial_vmax,
            homogeneity=2.0,
            exponent=0.8,
        )

        # check if optimization of regularization weights is possible
        spatial_prior = None
        for comp in self.all_components.values():
            if isinstance(comp, SpatialPrior):
                spatial_prior = comp
                break
        optimize_regu_weights = (
            regu_weights_optimizer.N0 < xp.inf and spatial_prior is not None
        )  # the condition for optimization is simply based on the the input parameter regu_spatial_N0 and the presence of a spatial prior in the model.

        dict_model_check = self.initialize_model_check_dict()
        acceptance_stats = {}

        for t in tqdm(range(1, max_iter + 1), disable=disable_progress_bar):
            # --- REGULARIZATION WEIGHTS OPTIMIZATION ---
            if (
                optimize_regu_weights
                and (self.N > 1)
                and t >= regu_weights_optimizer.N0
            ):
                tau_t = self.sample_regu_hyperparams(
                    spatial_prior,
                    regu_weights_optimizer,
                    t,
                    self.current[self.param_interest_name]["var"] * 1,
                    is_scalar=is_scalar,  # Same regularization for all dimensions d
                )

                spatial_prior.weights = tau_t * 1

                # recompute target_distribution neg log pdf and gradients with
                # new spatial regularization parameter
                self.current[self.param_interest_name] = target_distributions[
                    self.param_interest_name
                ].compute_all(
                    self.current,
                    compute_derivatives=True,
                    compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
                )

                additional_sampling_log["tau"] = spatial_prior.weights * 1
            elif optimize_regu_weights:
                additional_sampling_log["tau"] = spatial_prior.weights * 1

            # ------

            # --- RANDOM CHOICE KERNELS ---
            sampling_joint = self.rng.random() < self.selection_probas["joint"]["proba"]
            sampling_disjoint = not sampling_joint
            if sampling_joint:  # Joint sampling
                index_chosen_kernel: int = self.rng.choice(
                    len(self.selection_probas["joint"]["sub_proba"]),
                    p=self.selection_probas["joint"]["sub_proba"],
                )

                (
                    self.current,
                    accepted_t,
                    log_proba_accept_t,
                ) = self.kernels_dic["joint"][index_chosen_kernel].get_new_sample(
                    self.current,
                    t,
                    target_distributions,
                    self.rng,
                    self.compute_derivatives_2nd_order,
                )

                acceptance_stats = {
                    k: {
                        "accepted_t": accepted_t,
                        "log_proba_accept_t": log_proba_accept_t,
                        "type_t": self.kernels_dic["joint"][
                            index_chosen_kernel
                        ].plot_name,
                    }
                    for k in var_names
                }  # kernel choice = joint (log(U), Theta)
                acceptance_stats["joint"] = {
                    "accepted_t": accepted_t,
                    "log_proba_accept_t": log_proba_accept_t,
                    "type_t": self.kernels_dic["joint"][index_chosen_kernel].plot_name,
                }
            elif sampling_disjoint:  # Disjoint sampling
                for key in var_names:
                    if not self.sample_variables[key]:
                        acceptance_stats[key] = {
                            "accepted_t": 0,
                            "log_proba_accept_t": -xp.inf,
                            "type_t": -1,
                        }
                        continue

                    index_chosen_kernel: int = self.rng.choice(
                        len(self.selection_probas["disjoint"]["sub_proba"][key]),
                        p=self.selection_probas["disjoint"]["sub_proba"][key],
                    )

                    (
                        self.current,
                        accepted_t,
                        log_proba_accept_t,
                    ) = self.kernels_dic["disjoint"][key][
                        index_chosen_kernel
                    ].get_new_sample(
                        self.current,
                        t,
                        target_distributions,
                        self.rng,
                        self.compute_derivatives_2nd_order,
                    )

                    acceptance_stats[key] = {
                        "accepted_t": accepted_t,
                        "log_proba_accept_t": log_proba_accept_t,
                        "type_t": self.kernels_dic["disjoint"][key][
                            index_chosen_kernel
                        ].plot_name,
                    }

            for kernel in self.kernels_list:
                kernel.external_update_hyperparameters(
                    t=t,
                    current=self.current,
                    acceptance_stats=acceptance_stats,
                )

            # SAVING PART:
            # * if the memory is empty : initialize it
            nll_full = xp.zeros(
                (
                    self.N,
                    self.L,
                )
            )
            if saver.memory == {}:
                for key in var_names:
                    # additional_sampling_log[key]["v"] = (
                    #     self.v[key].reshape(self.current[key]["grad"].shape) * 1
                    # )
                    additional_sampling_log[key]["type_t"] = acceptance_stats[key][
                        "type_t"
                    ]
                    additional_sampling_log[key]["accepted_t"] = acceptance_stats[key][
                        "accepted_t"
                    ]
                    additional_sampling_log[key]["log_proba_accept_t"] = (
                        acceptance_stats[key]["log_proba_accept_t"]
                    )

                    dict_objective[key], nll_full_temp = target_distributions[
                        key
                    ].compute_all_for_saver(  # Update nll_full
                        self.current,
                        model_checking_component_name=self.model_checking_component_name,
                    )
                    nll_full += nll_full_temp

                components = set(
                    [
                        component
                        for target_distribution in target_distributions.values()
                        for component in target_distribution.distribution_components.values()
                    ]
                )
                objective_global = 0
                for comp in components:
                    # TODO: find a way to avoid recomuting utils here. Too costly.
                    comp.evaluate_all_nlpdf_utils(
                        current=self.current,
                        idx_pix=None,
                        compute_derivatives_2nd_order=False,
                        compute_derivatives=False,
                        mtm=False,
                    )
                    comp_pdf = comp.neglog_pdf(chromatic_gibbs=False)
                    objective_global += comp_pdf
                dict_objective["objective_global"] = objective_global * 1

                if t > T_BI:
                    dict_model_check = self._update_model_check_values(
                        dict_model_check,
                        self.model_checking_component,
                        nll_full,
                        objective=objective_global,
                    )  # Objective is used for optimization (self.is_stochastic == False)

                saver.initialize_memory(
                    max_iter,
                    t,
                    current=self.current,
                    nlpdf_utils={
                        var_name: dict() for var_name in var_names
                    },  # TODO: to change if we to store variables related to some component distributions. Previously it was used to keep track of parameters from the approximate likelihood.
                    dict_objective=dict_objective,
                    additional_sampling_log=additional_sampling_log,
                    sample_variables=self.sample_variables,
                    forward_map_evals=(
                        comp_forward.forward_map_evals
                        if saver.save_forward_map_evals
                        else None
                    ),
                    kernels_list=self.kernels_list,
                    dict_model_check=dict_model_check,
                )

                rng_state_array, rng_inc_array = self.get_rng_state()

                saver.update_memory(
                    t,
                    current=self.current,
                    nlpdf_utils={
                        var_name: dict() for var_name in var_names
                    },  # TODO: to change if we to store variables related to some component distributions. Previously it was used to keep track of parameters from the approximate likelihood.
                    dict_objective=dict_objective,
                    additional_sampling_log=additional_sampling_log,
                    rng_state_array=rng_state_array,
                    rng_inc_array=rng_inc_array,
                    sample_variables=self.sample_variables,
                    forward_map_evals=(
                        comp_forward.forward_map_evals
                        if saver.save_forward_map_evals
                        else None
                    ),
                    kernels_list=self.kernels_list,
                    dict_model_check=dict_model_check,
                )

            elif saver.check_need_to_update_memory(t):
                for key in var_names:
                    # additional_sampling_log[key]["v"] = (
                    #     self.v[key].reshape(self.current[key]["grad"].shape) * 1
                    # )
                    additional_sampling_log[key]["type_t"] = acceptance_stats[key][
                        "type_t"
                    ]
                    additional_sampling_log[key]["accepted_t"] = acceptance_stats[key][
                        "accepted_t"
                    ]
                    additional_sampling_log[key]["log_proba_accept_t"] = (
                        acceptance_stats[key]["log_proba_accept_t"]
                    )

                    dict_objective[key], nll_full_temp = target_distributions[
                        key
                    ].compute_all_for_saver(
                        self.current,
                        model_checking_component_name=self.model_checking_component_name,
                    )
                    nll_full += nll_full_temp

                components = set(
                    [
                        component
                        for target_distribution in target_distributions.values()
                        for component in target_distribution.distribution_components.values()
                    ]
                )
                objective_global = 0
                for comp in components:
                    # TODO: find a way to avoid recomuting utils here. Too costly.
                    comp.evaluate_all_nlpdf_utils(
                        current=self.current,
                        idx_pix=None,
                        compute_derivatives_2nd_order=False,
                        compute_derivatives=False,
                        mtm=False,
                    )
                    comp_pdf = comp.neglog_pdf(chromatic_gibbs=False)
                    objective_global += comp_pdf
                dict_objective["objective_global"] = objective_global * 1

                if t > T_BI:
                    dict_model_check = self._update_model_check_values(
                        dict_model_check,
                        self.model_checking_component,
                        nll_full,
                        objective=objective_global,
                    )  # Objective is used for optimization (self.is_stochastic == False)

                rng_state_array, rng_inc_array = self.get_rng_state()

                saver.update_memory(
                    t,
                    current=self.current,
                    nlpdf_utils={
                        var_name: dict() for var_name in var_names
                    },  # TODO: to change if we to store variables related to some component distributions. Previously it was used to keep track of parameters from the approximate likelihood.
                    dict_objective=dict_objective,
                    additional_sampling_log=additional_sampling_log,
                    rng_state_array=rng_state_array,
                    rng_inc_array=rng_inc_array,
                    sample_variables=self.sample_variables,
                    forward_map_evals=(
                        comp_forward.forward_map_evals
                        if saver.save_forward_map_evals
                        else None
                    ),
                    kernels_list=self.kernels_list,
                    dict_model_check=dict_model_check,
                )

            else:
                pass

            if saver.check_need_to_save(t):
                saver.save_to_file()

        # ---------
        dict_model_check = self._finalize_model_check_values(
            dict_model_check,
            likelihood=self.model_checking_component,
            nll_full=nll_full,
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
