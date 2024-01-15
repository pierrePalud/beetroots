r"""Contains a class of sampler used in the Meudon PDR code Bayesian inversion problems
"""
import copy
from typing import Optional, Tuple, Union

import numpy as np
from scipy.special import softmax
from tqdm.auto import tqdm

from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood
from beetroots.modelling.posterior import Posterior
from beetroots.sampler.abstract_sampler import Sampler
from beetroots.sampler.saver.abstract_saver import Saver
from beetroots.sampler.utils import utils
from beetroots.sampler.utils.mml import EBayesMMLELogRate
from beetroots.sampler.utils.my_sampler_params import MySamplerParams


class MySampler(Sampler):
    r"""Defines the sampler proposed in :cite:t:`paludEfficientSamplingNon2023` that randomly combines two transition kernels :

    1. a independent MTM-chromatic Gibbs transition kernel
    2. a position-dependent MALA transition kernel with the RMSProp pre-conditioner
    """

    def __init__(
        self,
        my_sampler_params: MySamplerParams,
        D: int,
        L: int,
        N: int,
        rng: np.random.Generator = np.random.default_rng(42),
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
            random number generator (for reproducibility), by default np.random.default_rng(42)
        """

        # P-MALA params
        # ! redefine size of params
        self.eps0 = my_sampler_params.initial_step_size
        r"""float: step size used in the Position-dependent MALA transition kernel, denoted :math:`\epsilon > 0` in the article"""

        self.lambda_ = my_sampler_params.extreme_grad
        r"""float: limit value that avoids division by zero when computing the RMSProp preconditioner, denoted :math:`\eta > 0` in the article"""

        self.alpha = my_sampler_params.history_weight
        r"""float: weight of past values of :math:`v` in the exponential decay (cf RMSProp preconditioner), denoted :math:`\alpha \in ]0,1[` in the article"""

        # MTM params
        # assert np.isclose(
        #     int(pow(my_sampler_params.k_mtm, 1 / D)) ** D, my_sampler_params.k_mtm
        # ), "number of candidates for mtm needs to have an integer D-root"
        self.k_mtm = my_sampler_params.k_mtm
        r"""int: number of candidates in the MTM kernel, denoted :math:`K` in the article"""

        # overall
        self.selection_probas = my_sampler_params.selection_probas
        r"""np.ndarray: vector of selection probabilities for the MTM and PMALA kernels, respectively, i.e., :math:`[p_{MTM}, 1 - p_{MTM}]`"""
        assert (
            np.sum(self.selection_probas) == 1
        ), f"{self.selection_probas} should sum to 1"

        self.stochastic = my_sampler_params.is_stochastic
        r"""bool: if True, the algorithm performs sampling, and optimization otherwise"""

        self.compute_correction_term = my_sampler_params.compute_correction_term
        r"""bool: wether or not to use the correction term (denoted :math:`\gamma` in the article) during the sampling (only used if `is_stochastic=True`)"""

        self.compute_derivatives_2nd_order = (
            self.stochastic and self.compute_correction_term
        )
        r"""bool: wether to compute the expensive second order derivatives terms. Only true when the sampler runs a Markov chain (2nd order never used in optimization) and when the correction term denoted :math:`\gamma` is to be computed."""

        self.D = D
        r"""int: total number of physical parameters to reconstruct"""
        self.L = L
        r"""int: number of observables per component :math:`n`"""
        self.N = N
        r"""int: total number of pixels to reconstruct"""

        self.rng = rng
        r"""numpy.random.Generator: random number generator (for reproducibility)"""

        # initialization values, not to be kept during sampling
        self.v = np.zeros((N * D,))
        r"""np.ndarray: RMSProp gradient variance vector, denoted :math:`v` in the article"""

        self.current = {}
        r"""dict: contains all the data about the current iterate (including the evaluations of the forward map and derivatives, etc.)"""

    def generate_random_start_Theta_1pix(
        self, Theta: np.ndarray, posterior: Posterior, idx_pix: np.ndarray
    ) -> np.ndarray:
        r"""draws a random vectors for components :math:`n` (e.g., a pixel :math:`\theta_n`). The distribution used to draw these vectors is:

        * the smooth indicator prior
        * a combination of the smooth indicator prior and of a Gaussian mixture defined with the set of all combinations of neighbors of component :math:`n`

        Parameters
        ----------
        Theta : np.ndarray
            current iterate
        posterior : Posterior
            contains the lower and upper bounds of the hypercube
        idx_pix : np.ndarray
            indices of the pixels

        Returns
        -------
        np.array of shape (n_pix, self.k_mtm, D)
            random element of the hypercube defined by the lower and upper bounds with uniform distribution

        Raises
        ------
        ValueError : if ``posterior.prior_indicator`` is None
        """
        seed = self.rng.integers(0, 1_000_000_000)
        n_pix = idx_pix.size

        if posterior.prior_indicator is None:
            raise ValueError("The Posterior has no specified smooth indicator prior")

        if posterior.prior_spatial is None:
            # * sample from smooth indicator prior
            return utils.sample_smooth_indicator(
                posterior.prior_indicator.lower_bounds,
                posterior.prior_indicator.upper_bounds,
                posterior.prior_indicator.indicator_margin_scale,
                size=(n_pix * self.k_mtm, self.D),
                seed=seed,
            ).reshape((n_pix, self.k_mtm, self.D))

        else:
            return utils.sample_conditional_spatial_and_indicator_prior(
                Theta,
                posterior.prior_spatial.list_edges,
                posterior.prior_spatial.weights,
                posterior.prior_indicator.lower_bounds,
                posterior.prior_indicator.upper_bounds,
                posterior.prior_indicator.indicator_margin_scale,
                idx_pix=idx_pix,
                k_mtm=self.k_mtm,
                seed=seed,
            )  # (n_pix, self.k_mtm, D)

    def _update_model_check_values(
        self,
        dict_model_check: dict,
        likelihood: Likelihood,
        nll_full: np.ndarray,
        objective: float,
    ) -> dict:
        count_pval = dict_model_check["count_pval"] * 1
        y_copy = likelihood.y * 1

        if self.stochastic:
            dict_model_check["clppd_online"] *= count_pval / (count_pval + 1)
            dict_model_check["clppd_online"] += np.exp(-nll_full) / (count_pval + 1)

            y_rep = likelihood.sample_observation_model(
                self.current["forward_map_evals"],
                self.rng,
            )
            likelihood_rep = copy.deepcopy(likelihood)
            likelihood_rep.y = y_rep * 1

            assert np.allclose(likelihood.y, y_copy), "nooooo"

            nll_utils_rep = likelihood_rep.evaluate_all_nll_utils(
                self.current["forward_map_evals"],
                idx=None,
                compute_derivatives=False,
            )
            nll_y_rep_full = likelihood_rep.neglog_pdf(
                self.current["forward_map_evals"],
                nll_utils_rep,
                full=True,
            )

            # p-value per (N, L) with y_rep_{n,ell} <= y_{n,ell}
            dict_model_check["p_values_y"] *= count_pval / (count_pval + 1)
            dict_model_check["p_values_y"] += (y_rep <= likelihood.y) / (count_pval + 1)

            # p-value per (N,) with
            # p(y_rep_n \vert theta_n) <= p(y_n \vert theta_n)
            nll_y = np.sum(nll_full, axis=1)  # (N,)
            nll_y_rep = np.sum(nll_y_rep_full, axis=1)  # (N,)

            dict_model_check["p_values_llh"] *= count_pval / (count_pval + 1)
            dict_model_check["p_values_llh"] += (nll_y_rep >= nll_y) / (count_pval + 1)

            dict_model_check["count_pval"] += 1

        else:
            if objective < dict_model_check["best_objective"]:
                dict_model_check["best_objective"] = objective * 1
                dict_model_check["clppd_online"] = np.exp(-nll_full)

            # p-values are computed at the end of the optimisation process.

        return dict_model_check

    def _finalize_model_check_values(
        self,
        dict_model_check: dict,
        likelihood: Likelihood,
        forward_map_evals: dict,
        nll_full: np.ndarray,
    ) -> dict:
        if not self.stochastic:
            # optimization p-value computations on estimated \hat{\theta}
            for count_pval in tqdm(range(self.ESS_OPTIM)):
                y_rep = likelihood.sample_observation_model(
                    forward_map_evals,
                    self.rng,
                )
                likelihood_rep = copy.deepcopy(likelihood)
                likelihood_rep.y = y_rep * 1
                nll_utils_rep = likelihood_rep.evaluate_all_nll_utils(
                    forward_map_evals,
                    idx=None,
                    compute_derivatives=False,
                )
                nll_y_rep_full = likelihood_rep.neglog_pdf(
                    forward_map_evals,
                    nll_utils_rep,
                    full=True,
                )

                # p-value per (N, L) with y_rep_{n,ell} <= y_{n,ell}
                dict_model_check["p_values_y"] *= count_pval / (count_pval + 1)
                dict_model_check["p_values_y"] += (y_rep <= likelihood.y) / (
                    count_pval + 1
                )

                # p-value per (N,) with
                # p(y_rep_n \vert theta_n) <= p(y_n \vert theta_n)
                nll_y = np.sum(nll_full, axis=1)  # (N,)
                nll_y_rep = np.sum(nll_y_rep_full, axis=1)  # (N,)

                dict_model_check["p_values_llh"] *= count_pval / (count_pval + 1)
                dict_model_check["p_values_llh"] += (nll_y_rep >= nll_y) / (
                    count_pval + 1
                )

        # this p-value should be between 0 and 0.5
        dict_model_check["p_values_y"] = np.where(
            dict_model_check["p_values_y"] > 0.5,
            1 - dict_model_check["p_values_y"],
            dict_model_check["p_values_y"],
        )

        return dict_model_check

    def sample(
        self,
        posterior: Posterior,
        saver: Saver,
        max_iter: int,
        Theta_0: Optional[np.ndarray] = None,
        disable_progress_bar: bool = False,
        #
        regu_spatial_N0: Union[int, float] = np.infty,
        regu_spatial_scale: float = 1.0,
        regu_spatial_vmin: float = 1e-8,
        regu_spatial_vmax: float = 1e8,
        #
        T_BI: int = 0,  # used only for clppd
    ) -> None:
        r"""main method of the class, runs the sampler

        Parameters
        ----------
        posterior : Posterior
            probability distribution to be sampled
        saver : Saver
            object responsible for progressively saving the Markov chain data during the run
        max_iter : int
            total duration of a Markov chain
        Theta_0 : Optional[np.ndarray], optional
            starting point, by default None
        disable_progress_bar : bool, optional
            wether to disable the progress bar, by default False
        regu_spatial_N0 : Union[int, float], optional
            number of iterations defining the initial update phase (for spatial regularization weight optimization). np.infty means that the optimization phase never starts, and that the weight optimization is not applied. by default np.infty
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
        additional_sampling_log = {}

        if Theta_0 is None:
            print("starting from a random point")
            Theta_0 = self.generate_random_start_Theta(posterior)  # (N, D)

        assert Theta_0 is not None
        assert Theta_0.shape == (self.N, self.D)

        self.current = posterior.compute_all(
            Theta_0, compute_derivatives_2nd_order=self.compute_derivatives_2nd_order
        )

        assert np.isnan(self.current["objective"]) == 0
        assert np.sum(np.isnan(self.current["grad"])) == 0
        # assert (
        #     self.current["forward_map_evals"]["f_Theta"].min() >= 0
        # ), f"{self.current['forward_map_evals']['f_Theta'].min()}"  # {self.current['forward_map_evals']['log_f_Theta'].min()}"

        # if v0 is None:
        #     v_max = (self.current["grad"] ** 2).max(axis=0)
        #     self.v = (v_max[None, :] * np.ones((self.N, self.D))).flatten()
        #     assert np.sum(np.isnan(self.v)) == 0, np.sum(np.isnan(self.v))
        # else:
        #     self.v = v0
        self.v = self.current["grad"].flatten() ** 2
        assert np.sum(np.isnan(self.v)) == 0.0
        assert np.sum(np.isinf(self.v)) == 0.0

        # print(self.v, Theta_0, self.lambda_)

        # self.u = self.current["grad"].flatten() * self.current["hess_diag"].flatten()
        # assert self.u.shape == (self.N * self.D,)

        # if sample_regu_weights and T_BI_reguweights is None:
        #     T_BI_reguweights = 0
        # if not (sample_regu_weights) and T_BI_reguweights is None:
        #     T_BI_reguweights = max_iter * 1

        rng_state_array, _ = self.get_rng_state()

        # self.j_t = 0
        self.j_t = np.zeros((self.N * self.D,))

        # if self.N > 10:
        #     print(f"at start: obj = {self.current['objective']}")

        # n_sites = len(posterior.prior_spatial.dict_sites)

        # n_repetitions_first_mtm = 1
        # list_n_first_samples = list(range(n_sites)) * n_repetitions_first_mtm
        # self.rng.shuffle(list_n_first_samples)

        # if self.N > 1000:
        # list_n_first_samples = []
        # list_n_first_samples = list_n_first_samples[:250]

        regu_weights_optimizer = EBayesMMLELogRate(
            scale=regu_spatial_scale,
            N0=regu_spatial_N0,
            N1=+np.infty,
            dim=self.D * self.N,
            vmin=regu_spatial_vmin,
            vmax=regu_spatial_vmax,
            homogeneity=2.0,
            exponent=0.8,
        )
        optimize_regu_weights = regu_weights_optimizer.N0 < np.infty

        # clppd = computed log point-wise predictive density.
        # if self.stochastic : avg of all pred. likelihood terms (with burn-in)
        # but burn-in values are negligible (0) compared to non burn-in
        # else : predictive likelihood with best param theta
        clppd_online = np.zeros((self.N, self.L))
        # utilitary variables
        best_objective = np.infty  # used only if not self.stochastic
        count_pval = 0  # used only if self.stochastic

        # p(y^{rep}_\ell <= y_\ell | y)
        p_values_y = np.zeros((self.N, self.L))
        # p(y^{rep}_\ell \in [ q_{25\%}(y_\ell), q_{75\%}(y_\ell) ] | y)
        p_values_llh = np.zeros((self.N,))

        dict_model_check = {
            "clppd_online": clppd_online,
            "best_objective": best_objective,
            "count_pval": count_pval,
            "p_values_y": p_values_y,
            "p_values_llh": p_values_llh,
        }

        for t in tqdm(range(1, max_iter + 1), disable=disable_progress_bar):
            if optimize_regu_weights and (self.N > 1):
                assert posterior.prior_spatial is not None

                if t >= regu_weights_optimizer.N0:
                    tau_t = self.sample_regu_hyperparams(
                        posterior,
                        regu_weights_optimizer,
                        t,
                        self.current["Theta"] * 1,
                    )

                    posterior.prior_spatial.weights = tau_t * 1

                    # recompute posterior neg log pdf and gradients with
                    # new spatial regularization parameter
                    self.current = posterior.compute_all(
                        self.current["Theta"],
                        self.current["forward_map_evals"],
                        self.current["nll_utils"],
                        compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
                    )

                additional_sampling_log["tau"] = posterior.prior_spatial.weights * 1
            # ------

            type_t = np.argmax(
                self.rng.multinomial(
                    1,
                    pvals=self.selection_probas,
                )
            )
            if type_t == 0:
                (
                    accepted_t,
                    log_proba_accept_t,
                ) = self.generate_new_sample_mtm(t, posterior)
            else:
                assert type_t == 1
                (
                    accepted_t,
                    log_proba_accept_t,
                ) = self.generate_new_sample_pmala_rmsprop(t, posterior)

            # # check for nan in Theta
            # if np.sum(np.isnan(self.current["Theta"])) > 0:
            #     print(f"type_t : {type_t}")
            #     raise ValueError(
            #         f"Theta contains {np.sum(np.isnan(self.current['Theta']))} nan"
            #     )

            # * if the memory is empty : initialize it
            if saver.memory == {}:
                additional_sampling_log["v"] = self.v.reshape((self.N, self.D)) * 1
                additional_sampling_log["type_t"] = type_t
                additional_sampling_log["accepted_t"] = accepted_t
                additional_sampling_log["log_proba_accept_t"] = log_proba_accept_t

                dict_objective, nll_full = posterior.compute_all_for_saver(
                    self.current["Theta"],
                    self.current["forward_map_evals"],
                    self.current["nll_utils"],
                )

                if t > T_BI:
                    dict_model_check = self._update_model_check_values(
                        dict_model_check,
                        posterior.likelihood,
                        nll_full,
                        dict_objective["objective"] * 1,
                    )

                saver.initialize_memory(
                    max_iter,
                    t,
                    Theta=self.current["Theta"],
                    forward_map_evals=self.current["forward_map_evals"],
                    nll_utils=self.current["nll_utils"],
                    dict_objective=dict_objective,
                    additional_sampling_log=additional_sampling_log,
                )

                rng_state_array, rng_inc_array = self.get_rng_state()

                saver.update_memory(
                    t,
                    Theta=self.current["Theta"],
                    forward_map_evals=self.current["forward_map_evals"],
                    nll_utils=self.current["nll_utils"],
                    dict_objective=dict_objective,
                    additional_sampling_log=additional_sampling_log,
                    rng_state_array=rng_state_array,
                    rng_inc_array=rng_inc_array,
                )

            elif saver.check_need_to_update_memory(t):
                # print(f"updating memory at t={t}")
                additional_sampling_log["v"] = self.v.reshape((self.N, self.D)) * 1
                additional_sampling_log["type_t"] = type_t
                additional_sampling_log["accepted_t"] = accepted_t
                additional_sampling_log["log_proba_accept_t"] = log_proba_accept_t

                dict_objective, nll_full = posterior.compute_all_for_saver(
                    self.current["Theta"],
                    self.current["forward_map_evals"],
                    self.current["nll_utils"],
                )

                if t > T_BI:
                    dict_model_check = self._update_model_check_values(
                        dict_model_check,
                        posterior.likelihood,
                        nll_full,
                        dict_objective["objective"] * 1,
                    )

                rng_state_array, rng_inc_array = self.get_rng_state()

                saver.update_memory(
                    t,
                    Theta=self.current["Theta"],
                    forward_map_evals=self.current["forward_map_evals"],
                    nll_utils=self.current["nll_utils"],
                    dict_objective=dict_objective,
                    additional_sampling_log=additional_sampling_log,
                    rng_state_array=rng_state_array,
                    rng_inc_array=rng_inc_array,
                )

            else:
                pass

            if saver.check_need_to_save(t):
                # print(f"saving memory at t={t}")
                saver.save_to_file()

        # ---------
        dict_model_check = self._finalize_model_check_values(
            dict_model_check,
            likelihood=posterior.likelihood,
            forward_map_evals=self.current["forward_map_evals"],
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

    # def generate_new_sample_pmala_rmsprop(self, t, posterior):
    #     """generates a new sample using the position-dependent MALA transition kernel

    #     Parameters
    #     ----------
    #     t : int
    #         current iteration index
    #     score_model : ScoreModel
    #         negative log posterior class

    #     Returns
    #     -------
    #     accepted : bool
    #         wether or not the candidate was accepted
    #     log_proba_accept : float
    #         log of the acceptance proba
    #     """
    #     grad_t = self.current["grad"].flatten()

    #     # print(self.lambda_ + np.sqrt(self.v))
    #     diag_G_t = 1 / (self.lambda_ + np.sqrt(self.v))

    #     assert np.all(
    #         diag_G_t > 0
    #     ), f"{diag_G_t}, {self.lambda_ + np.sqrt(self.v)}, {self.v}"

    #     # generate random
    #     z_t = self.rng.standard_normal(size=self.N * self.D)
    #     z_t *= np.sqrt(self.eps0 * diag_G_t)

    #     # bias correction term
    #     if self.compute_correction_term:
    #         # recursive version
    #         # correction = -1 / 2 * diag_G_t ** 2 / np.sqrt(self.v) * self.u

    #         # only with corresponding term
    #         hess_diag_t = self.current["hess_diag"].flatten()
    #         correction = (
    #             -(1 - self.alpha)
    #             * self.alpha ** self.j_t
    #             * (diag_G_t ** 2)
    #             / np.sqrt(self.v)
    #             * grad_t
    #             * hess_diag_t
    #         )
    #         if np.sum(~np.isfinite(correction)) > 0:
    #             print(
    #                 f"num of nan in correction term: {np.sum(~np.isfinite(correction))}"
    #             )
    #         correction = np.nan_to_num(correction)  # ? nécessaire ?
    #     else:
    #         correction = np.zeros((self.N * self.D,))

    #     # combination
    #     mu_current = (
    #         self.current["Theta"].flatten()
    #         - self.eps0 / 2 * diag_G_t * grad_t
    #         + self.eps0 * correction
    #     )

    #     if self.stochastic:
    #         candidate = mu_current + z_t  # (N * D,)

    #         log_q_candidate_given_current = -1 / 2 * np.sum(np.log(diag_G_t)) - 1 / (
    #             2 * self.eps0
    #         ) * np.sum((candidate - mu_current) ** 2 / diag_G_t)

    #         # * compute log_q of candidate given current
    #         candidate_all = posterior.compute_all(
    #             candidate.reshape(self.N, self.D),
    #         )
    #         grad_cand = candidate_all["grad"].flatten()
    #         v_cand = self.alpha * self.v + (1 - self.alpha) * grad_cand ** 2
    #         diag_G_cand = 1 / (self.lambda_ + np.sqrt(v_cand))

    #         if self.compute_correction_term:
    #             hess_diag_cand = candidate_all["hess_diag"].flatten()

    #             correction_cand = -(
    #                 (1 - self.alpha)
    #                 * diag_G_cand ** 2
    #                 / np.sqrt(v_cand)
    #                 * grad_cand
    #                 * hess_diag_cand
    #             )
    #         else:
    #             correction_cand = np.zeros((self.N * self.D,))

    #         mu_cand = (
    #             candidate
    #             - self.eps0 / 2 * diag_G_cand * grad_cand
    #             + self.eps0 * correction_cand
    #         )

    #         log_q_current_given_candidate = -1 / 2 * np.sum(np.log(diag_G_cand)) - 1 / (
    #             2 * self.eps0
    #         ) * np.sum((self.current["Theta"].flatten() - mu_cand) ** 2 / diag_G_cand)

    #         # * compute proba accept
    #         logpdf_current = -self.current["objective"] * 1
    #         logpdf_candidate = -candidate_all["objective"] * 1

    #         log_proba_accept = (
    #             logpdf_candidate
    #             - logpdf_current
    #             + log_q_current_given_candidate
    #             - log_q_candidate_given_current
    #         )
    #         log_u = np.log(self.rng.uniform(0, 1))
    #         # print(
    #         #     f"{log_u:.2e}, {log_proba_accept:.4e}, {logpdf_candidate:.4e},, {logpdf_current:.4e}, {log_q_current_given_candidate:.4e}, {log_q_candidate_given_current:.4e}"
    #         # )

    #         if log_u < log_proba_accept:
    #             self.current = copy.copy(candidate_all)
    #             self.v = v_cand * 1
    #             assert np.sum(np.isnan(self.v)) == 0.0
    #             assert np.sum(np.isinf(self.v)) == 0.0

    #             self.j_t = np.zeros((self.N * self.D,))
    #             return True, log_proba_accept
    #         else:
    #             self.j_t += 1
    #             self.v = v_cand * 1
    #             assert np.sum(np.isnan(self.v)) == 0.0
    #             assert np.sum(np.isinf(self.v)) == 0.0
    #             return False, log_proba_accept

    #     # * in case we are doing optimization and not sampling
    #     candidate_all = posterior.compute_all(mu_current.reshape((self.N, self.D)))
    #     if candidate_all["objective"] < self.current["objective"]:
    #         self.current = copy.copy(candidate_all)
    #         accept = True
    #         proba = 1
    #     else:
    #         candidate = mu_current + z_t  # (N * D,)
    #         candidate_all = posterior.compute_all(candidate.reshape((self.N, self.D)))

    #         if candidate_all["objective"] < self.current["objective"]:
    #             self.current = copy.copy(candidate_all)
    #             accept = True
    #             proba = 1
    #         else:
    #             accept = False
    #             proba = 0

    #     grad_tp1 = candidate_all["grad"].flatten()
    #     self.v = self.alpha * self.v + (1 - self.alpha) * grad_tp1 ** 2

    #     assert np.sum(np.isnan(self.v)) == 0.0
    #     assert (
    #         np.sum(np.isinf(self.v)) == 0.0
    #     ), f"{candidate_all['Theta']}, {candidate_all['grad']}"
    #     assert np.sum(np.isnan(self.current["Theta"])) == 0.0

    #     return accept, proba

    def generate_new_sample_pmala_rmsprop(self, t: int, posterior: Posterior):
        """generates a new sample using the position-dependent MALA transition kernel

        Parameters
        ----------
        t : int
            current iteration index
        posterior : ScoreModel
            negative log posterior class

        Returns
        -------
        accepted : bool
            wether or not the candidate was accepted
        log_proba_accept : float
            log of the acceptance proba
        """
        if self.stochastic:
            accept_total = np.zeros((self.N,))
            log_proba_accept_total = np.zeros((self.N,))

            # * define proba of changing each pixel
            # * either uniformly or depending on their respective nll
            # if posterior.prior_spatial is not None:
            # n_sites = len(posterior.dict_sites)
            # idx_site = int(self.rng.integers(0, n_sites))
            list_idx = np.array(list(posterior.dict_sites.keys()))

            for idx_site in list_idx:
                idx_pix = posterior.dict_sites[idx_site]
                n_pix = idx_pix.size

                new_Theta = self.current["Theta"] * 1  # (N, D)
                grad_t = self.current["grad"][idx_pix, :] * 1
                v_current = self.v.reshape((self.N, self.D))[idx_pix, :] * 1

                # generate random
                diag_G_t = 1 / (self.lambda_ + np.sqrt(v_current))  # (n_pix, D)

                assert np.all(
                    diag_G_t > 0
                ), f"{diag_G_t}, {self.lambda_ + np.sqrt(self.v)}, {self.v}"

                z_t = self.rng.standard_normal(size=(n_pix, self.D))
                z_t *= np.sqrt(self.eps0 * diag_G_t)  # (n_pix, D)

                # bias correction term
                if self.compute_correction_term:
                    # recursive version
                    # correction = -1 / 2 * diag_G_t ** 2
                    # / np.sqrt(self.v) * self.u

                    # only with corresponding term
                    hess_diag_t = self.current["hess_diag"][idx_pix, :] * 1
                    j_t = self.j_t.reshape((self.N, self.D))[idx_pix, :] * 1

                    correction = (
                        -(1 - self.alpha)
                        * self.alpha**j_t
                        * (diag_G_t**2)
                        / np.sqrt(v_current)
                        * grad_t
                        * hess_diag_t
                    )  # (n_pix, D)
                    if np.sum(~np.isfinite(correction)) > 0:
                        n_inf = np.sum(~np.isfinite(correction))
                        print(f"num of nan in correction term: {n_inf}")
                    correction = np.nan_to_num(correction)  # ? nécessaire ?
                else:
                    correction = np.zeros((n_pix, self.D))

                # combination
                mu_current = (
                    new_Theta[idx_pix, :]
                    - self.eps0 / 2 * diag_G_t * grad_t
                    + self.eps0 * correction
                )  # (n_pix, D)

                candidate = mu_current + z_t  # (n_pix, D)

                log_q_candidate_given_current = -1 / 2 * np.sum(
                    np.log(diag_G_t), axis=1
                ) - 1 / (2 * self.eps0) * np.sum(
                    (candidate - mu_current) ** 2 / diag_G_t, axis=1
                )  # (n_pix,)

                shape_q = log_q_candidate_given_current.shape
                assert shape_q == (n_pix,), f"{shape_q}"

                # * compute log_q of candidate given current
                candidate_full = new_Theta * 1
                candidate_full[idx_pix, :] = mu_current * 1

                candidate_all = posterior.compute_all(
                    candidate_full,
                    compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
                )
                grad_cand = candidate_all["grad"][idx_pix, :] * 1
                v_cand = (
                    self.alpha * v_current + (1 - self.alpha) * grad_cand**2
                )  # (n_pix, D)
                diag_G_cand = 1 / (self.lambda_ + np.sqrt(v_cand))  # (n_pix, D)

                if self.compute_correction_term:
                    hess_diag_cand = candidate_all["hess_diag"][idx_pix, :] * 1

                    correction_cand = -(
                        (1 - self.alpha)
                        * diag_G_cand**2
                        / np.sqrt(v_cand)
                        * grad_cand
                        * hess_diag_cand
                    )
                else:
                    correction_cand = np.zeros((n_pix, self.D))

                mu_cand = (
                    candidate
                    - self.eps0 / 2 * diag_G_cand * grad_cand
                    + self.eps0 * correction_cand
                )  # (n_pix, D)

                log_q_current_given_candidate = -1 / 2 * np.sum(
                    np.log(diag_G_cand), axis=1
                ) - 1 / (2 * self.eps0) * np.sum(
                    (new_Theta[idx_pix, :] - mu_cand) ** 2 / diag_G_cand, axis=1
                )  # (n_pix,)

                shape_q = log_q_current_given_candidate.shape
                assert shape_q == (n_pix,), f"{shape_q}"

                # * compute proba accept
                logpdf_current = -self.current["objective_pix"][idx_pix]
                logpdf_candidate = -candidate_all["objective_pix"][idx_pix]

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

                log_u = np.log(self.rng.uniform(0, 1, size=n_pix))
                accept_arr = log_u < log_proba_accept

                new_Theta[idx_pix, :] = np.where(
                    accept_arr[:, None] * np.ones((n_pix, self.D)),
                    candidate,  # (n_pix, D)
                    new_Theta[idx_pix, :],  # (n_pix, D)
                )

                accept_total[idx_pix] = accept_arr * 1
                log_proba_accept_total[idx_pix] = log_proba_accept * 1

                # update v and j
                v = self.v.reshape((self.N, self.D)) * 1
                v[idx_pix, :] = v_cand * 1
                self.v = v.flatten()

                j = self.j_t.reshape((self.N, self.D)) * 1
                j[idx_pix, :] = np.where(
                    accept_arr[:, None],
                    0.0,  # reset to 0 if accept
                    j[idx_pix, :] + 1,  # else add 1
                )
                self.j_t = j.flatten()

                if accept_arr.max() > 0:  # if at least one accept
                    self.current = posterior.compute_all(
                        new_Theta,
                        compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
                    )

            # after loop
            return accept_total.mean(), log_proba_accept_total.mean()

        else:  # if optimization
            grad_t = self.current["grad"].flatten()

            # print(self.lambda_ + np.sqrt(self.v))
            diag_G_t = 1 / (self.lambda_ + np.sqrt(self.v))

            assert np.all(
                diag_G_t > 0
            ), f"{diag_G_t}, {self.lambda_ + np.sqrt(self.v)}, {self.v}"

            # generate random
            z_t = self.rng.standard_normal(size=self.N * self.D)
            z_t *= np.sqrt(self.eps0 * diag_G_t)

            # combination
            mu_current = (
                self.current["Theta"].flatten() - self.eps0 / 2 * diag_G_t * grad_t
            )

            candidate_all = posterior.compute_all(
                mu_current.reshape((self.N, self.D)),
                compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
            )
            if candidate_all["objective"] < self.current["objective"]:
                self.current = copy.copy(candidate_all)
                accept = True
                proba = 1
            else:
                candidate = mu_current + z_t  # (N * D,)
                candidate_all = posterior.compute_all(
                    candidate.reshape((self.N, self.D)),
                    compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
                )

                if candidate_all["objective"] < self.current["objective"]:
                    self.current = copy.copy(candidate_all)
                    accept = True
                    proba = 1
                else:
                    accept = False
                    proba = 0

            grad_tp1 = candidate_all["grad"].flatten()
            self.v = self.alpha * self.v + (1 - self.alpha) * grad_tp1**2

            assert np.sum(np.isnan(self.v)) == 0.0
            assert (
                np.sum(np.isinf(self.v)) == 0.0
            ), f"{candidate_all['Theta']}, {candidate_all['grad']}"
            assert np.sum(np.isnan(self.current["Theta"])) == 0.0

            return accept, proba

    def generate_new_sample_mtm(
        self, t: int, posterior: Posterior  # , idx_site: Union[int, None] = None
    ):
        r"""generates a new sample using the MTM transition kernel

        Parameters
        ----------
        t : int
            current iteration index
        posterior : Posterior
            target posterior distribution to sample from

        Returns
        -------
        accepted : bool
            wether or not the candidate was accepted
        log_proba_accept : float
            log of the acceptance proba
        """
        new_Theta = self.current["Theta"] * 1  # (N, D)

        accept_total = np.zeros((self.N,))
        log_rg_total = np.zeros((self.N,))

        # * define proba of changing each pixel
        # * either uniformly or depending on their respective nll
        # if posterior.prior_spatial is not None:
        # n_sites = len(posterior.dict_sites)
        # idx_site = int(self.rng.integers(0, n_sites))
        list_idx = np.array(list(posterior.dict_sites.keys()))

        for idx_site in list_idx:
            idx_pix = posterior.dict_sites[idx_site]
            n_pix = idx_pix.size

            # * generate and evaluate candidates
            candidates_pix = np.zeros((n_pix, self.k_mtm + 1, self.D))
            candidates_pix[:, :-1, :] = self.generate_random_start_Theta_1pix(
                new_Theta, posterior, idx_pix
            )
            candidates_pix[:, -1, :] = self.current["Theta"][idx_pix, :] * 1
            candidates_pix = candidates_pix.reshape(
                (n_pix * (self.k_mtm + 1), self.D)
            )  # (n_pix * (k_mtm+1), D)

            neglogpdf_candidates = posterior.likelihood.neglog_pdf_candidates(
                candidates_pix,
                idx=idx_pix,
                Theta_t=new_Theta * 1,  # self.current["Theta"] * 1
            )  # (n_pix * (k_mtm+1),)
            assert neglogpdf_candidates.shape == (n_pix * (self.k_mtm + 1),)

            candidates_pix = candidates_pix.reshape((n_pix, self.k_mtm + 1, self.D))
            # assert np.allclose(candidates_pix[:, -1, :], self.current["Theta"][idx_pix, :]) -> validated

            neglogpdf_candidates = neglogpdf_candidates.reshape((n_pix, self.k_mtm + 1))

            # * if optimization: define challenger with conditional posterior
            # * instead of likelihood, and only keep if better than current
            if not self.stochastic:
                neglogpdf_candidates += posterior.partial_neglog_pdf_priors(
                    new_Theta.copy(), idx_pix, candidates_pix
                )  # (n_pix, k_mtm)
                idx_challengers = np.argmin(
                    neglogpdf_candidates[:, :-1], axis=1
                )  # (n_pix,)
                assert idx_challengers.shape == (n_pix,)

                neglogpdf_candidates_challengers = np.zeros((n_pix,))
                challengers = np.zeros((n_pix, self.D))
                for i in range(n_pix):
                    neglogpdf_candidates_challengers[i] = neglogpdf_candidates[
                        i, idx_challengers[i]
                    ]
                    challengers[i, :] = candidates_pix[i, idx_challengers[i], :]
                # neglogpdf_candidates_challengers = neglogpdf_candidates[
                #     np.arange(len(candidates_pix)), idx_challengers
                # ]
                assert neglogpdf_candidates_challengers.shape == (
                    n_pix,
                ), neglogpdf_candidates_challengers.shape

                # challengers = candidates_pix[
                #     np.arange(len(candidates_pix)), idx_challengers, :
                # ]
                assert challengers.shape == (n_pix, self.D), challengers.shape

                # * compute values of corresponding pixels in current x
                candidates_already_Theta = candidates_pix[:, -1, :] * 1
                neglogpdf_already_Theta = neglogpdf_candidates[:, -1] * 1
                assert candidates_already_Theta.shape == (n_pix, self.D)
                assert neglogpdf_already_Theta.shape == (n_pix,)

                # * select best pixels
                accept_arr = (
                    (neglogpdf_candidates_challengers < neglogpdf_already_Theta)
                    & np.isfinite(neglogpdf_candidates_challengers)
                    & np.isfinite(neglogpdf_already_Theta)
                )

                new_Theta[idx_pix, :] = np.where(
                    accept_arr[:, None] * np.ones((n_pix, self.D)),
                    challengers,  # (n_pix, D)
                    candidates_already_Theta,  # (n_pix, D)
                )

                # * save which pixels were accepted
                accept_total[idx_pix] = accept_arr * 1

            # *------
            # * if sampling
            else:
                if posterior.prior_spatial is not None:
                    nlratio_prior_proposal = utils.compute_nlratio_prior_proposal(
                        new_Theta * 1,
                        posterior.prior_spatial.list_edges,
                        posterior.prior_spatial.weights,
                        idx_pix,
                        candidates_pix,
                    )
                    shape_ = nlratio_prior_proposal.shape
                    assert shape_ == (n_pix, self.k_mtm + 1)
                    neglogpdf_candidates += nlratio_prior_proposal

                neglogpdf_candidates_min = np.amin(
                    neglogpdf_candidates, axis=1, keepdims=True
                )
                neglogpdf_candidates -= neglogpdf_candidates_min

                pdf_candidates = np.exp(-neglogpdf_candidates)  # (n_pix, k_mtm)

                log_numerators = np.log(np.sum(pdf_candidates[:, :-1], axis=1))
                # log_numerators = np.where(
                #     np.isinf(log_numerators), -1e15, log_numerators
                # )

                assert log_numerators.shape == (n_pix,), log_numerators.shape
                # assert np.sum(1 - np.isfinite(log_numerators)) == 0, log_numerators

                # * choose challenger candidate
                weights = softmax(-neglogpdf_candidates[:, :-1], axis=1)
                assert np.sum(1 - np.isfinite(weights)) == 0, weights

                idx_challengers = np.zeros((n_pix,), dtype=int)
                for i in range(n_pix):
                    idx_challengers[i] = self.rng.choice(
                        self.k_mtm,
                        p=weights[i],
                    )

                challengers = candidates_pix[
                    np.arange(n_pix), idx_challengers, :
                ]  # (n_pix, D)
                neglogpdf_challengers = neglogpdf_candidates[
                    np.arange(n_pix), idx_challengers
                ]

                shape_ = neglogpdf_challengers.shape
                assert shape_ == (n_pix,), shape_

                # * denominator
                log_denominators = np.log(
                    np.sum(pdf_candidates, axis=1) - np.exp(-neglogpdf_challengers)
                )
                # log_denominators = np.where(
                #     np.isinf(log_denominators), -1e15, log_denominators
                # )

                shape_ = log_denominators.shape
                assert shape_ == (n_pix,), shape_

                # assert np.sum(1 - np.isfinite(log_numerators)) == 0, log_numerators
                # assert np.sum(1 - np.isfinite(log_denominators)) == 0, log_denominators

                # * accept-reject
                log_rg = log_numerators - log_denominators
                log_rg = np.where(
                    np.isfinite(log_rg), log_rg, 1e-15
                )  # if either log_numerators or log_denominators is not finite, do not accept

                log_u = np.log(self.rng.uniform(0, 1, size=n_pix))
                accept_arr = log_u < log_rg

                new_Theta[idx_pix, :] = np.where(
                    accept_arr[:, None] * np.ones((n_pix, self.D)),
                    challengers,  # (n_pix, D)
                    candidates_pix[:, -1, :],  # (n_pix, D)
                )

                accept_total[idx_pix] = accept_arr * 1
                log_rg_total[idx_pix] = log_rg * 1

                # * re-initialize j for new point
                new_j_t = self.j_t.reshape((self.N, self.D))
                new_j_t[idx_pix, :] = np.where(
                    accept_arr[:, None], 0.0, new_j_t[idx_pix, :]
                )
                self.j_t = new_j_t.flatten()  # (ND,)

        # *------
        # * once all sites have been dealt with, update global parameters
        if accept_total.max() > 0:  # if at least one accept
            self.current = posterior.compute_all(
                new_Theta,
                compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
            )

            new_v = self.v.reshape((self.N, self.D))
            new_v = np.where(
                accept_total[:, None],
                self.alpha * new_v + (1 - self.alpha) * self.current["grad"] ** 2,
                new_v,
            )
            self.v = new_v.flatten()
            assert np.sum(np.isnan(self.v)) == 0.0
            assert np.sum(np.isinf(self.v)) == 0.0

        if not self.stochastic:
            return np.mean(accept_total), np.mean(accept_total)
        else:
            return np.mean(accept_total), np.mean(log_rg_total)
