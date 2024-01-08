r"""Contains a class of sampler used in the Meudon PDR code Bayesian inversion problems
"""
import copy
import os
from sys import byteorder
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, softmax
from tqdm.auto import tqdm

from beetroots.modelling.posterior import Posterior
from beetroots.sampler.abstract_sampler import Sampler
from beetroots.sampler.saver.hierarchical_saver import HierarchicalSaver
from beetroots.sampler.utils import utils
from beetroots.sampler.utils.mml import EBayesMMLELogRate
from beetroots.sampler.utils.psgldparams import PSGLDParams


class HierarchicalSampler(Sampler):
    r"""Defines a sampler that randomly combines two transition kernels :

    1. a independent MTM variable-at-a-time transition kernel
    2. a position-dependent MALA transition kernel with the RMSProp pre-conditioner
    """

    def __init__(
        self,
        psgld_params: List[
            PSGLDParams
        ],  # ! one set of parameters per conditional distribution
        D: int,
        L: int,
        N: int,
        rng: np.random.Generator = np.random.default_rng(42),
    ):
        r"""

        Parameters
        ----------
        psgld_params : PSGLDParams
            contains the main parameters of the algorithm
        D : int
            total number of physical parameters to reconstruct
        N : int
            total number of pixels to reconstruct
        selection_probas : float
            probability of running the MTM transition kernel at a given iteration
        proba_unif : float
            probability to choose a pixel to sample using a uniform distribution. The other option is via a softmax that gives more weight to pixels with higher negative log likelihood (only used when the MTM transition kernel is run)
        k_mtm : int
            number of candidates to simulate and evaluate in the MTM transition kernel
        rng : numpy.random.Generator, optional
            random number generator (for reproducibility), by default np.random.default_rng(42)
        """
        # P-MALA params
        self.eps0 = np.empty((len(psgld_params),), dtype="d")
        self.lambda_ = np.empty((len(psgld_params),), dtype="d")
        self.alpha = np.empty((len(psgld_params),), dtype="d")

        # MTM params
        self.k_mtm = np.empty((len(psgld_params),), dtype="i")

        # overall
        self.selection_probas = np.empty((len(psgld_params), 2), dtype="d")
        self.stochastic = np.full((len(psgld_params),), False, dtype=bool)
        self.compute_correction_term = np.full((len(psgld_params),), False, dtype=bool)

        for k in range(len(psgld_params)):
            self.eps0[k] = psgld_params[k].initial_step_size
            self.lambda_[k] = psgld_params[k].extreme_grad
            self.alpha[k] = psgld_params[k].history_weight

            self.k_mtm[k] = psgld_params[k].k_mtm

            self.selection_probas[k] = psgld_params[k].selection_probas
            self.stochastic[k] = psgld_params[k].is_stochastic
            self.compute_correction_term[k] = psgld_params[k].compute_correction_term

        self.D = D
        self.L = L
        self.N = N

        self.rng = rng

        # initialization values, not to be kept during sampling
        # self.v_u = np.zeros((N * L,))
        self.v_Theta = np.zeros((N * D,))
        self.current_Theta = {}
        self.current_u = {}
        # self.additional_sampling_log = {}

        self.list_lower_bounds_mcs = None
        self.list_upper_bounds_mcs = None

    # TODO: P.-A.: to be deprecated? (not sure this is still useful)
    def generate_random_start_Theta_1pix(
        self, x: np.ndarray, conditional: Posterior, idx_pix: np.ndarray, k_mtm
    ) -> np.ndarray:
        """generates a random element of the hypercube defined by the lower and upper bounds with stratified sampling

        Parameters
        ----------
        conditional : Posterior
            contains the lower and upper bounds of the hypercube

        Returns
        -------
        x : np.array of shape (n_pix, self.k_mtm, D)
            random element of the hypercube defined by the lower and upper bounds with uniform distribution
        """
        seed = self.rng.integers(0, 1_000_000_000)
        n_pix = idx_pix.size

        if (
            conditional.prior_indicator is not None
            and conditional.prior_spatial is None
        ):
            # * sample from smooth indicator prior
            return utils.sample_smooth_indicator(
                conditional.prior_indicator.lower_bounds,
                conditional.prior_indicator.upper_bounds,
                conditional.prior_indicator.indicator_margin_scale,
                size=(n_pix * k_mtm, conditional.D),
                seed=seed,
            ).reshape((n_pix, k_mtm, conditional.D))

        if (
            conditional.prior_indicator is not None
            and conditional.prior_spatial is not None
        ):
            return utils.sample_conditional_spatial_and_indicator_prior(
                x,
                conditional.prior_spatial.list_edges,
                conditional.prior_spatial.weights,
                conditional.prior_indicator.lower_bounds,
                conditional.prior_indicator.upper_bounds,
                conditional.prior_indicator.indicator_margin_scale,
                idx_pix=idx_pix,
                k_mtm=k_mtm,
                seed=seed,
            )  # (n_pix, k_mtm, D)

        if conditional.prior_indicator is None:
            raise NotImplementedError(
                "no smooth indicator prior is not yet implemented"
            )

    def sample(
        self,
        conditional: Posterior,
        saver: HierarchicalSaver,
        max_iter: int,
        x0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,  # ! P.-A.: never used? remove?
        u0: Optional[np.ndarray] = None,
        u_v0: Optional[np.ndarray] = None,  # ! P.-A.: never used? remove?
        disable_progress_bar: bool = False,
        regu_spatial_N0: Union[int, float] = np.infty,
        regu_spatial_scale: float = 1.0,
        regu_spatial_vmin: float = 1e-8,
        regu_spatial_vmax: float = 1e8,
    ) -> None:
        """main method of the class, runs the sampler

        Parameters
        ----------
        conditional : Posterior
            probabilistic models
        saver : HierarhcicalSaver
            enables to save the progression of the sampling
        max_iter : int
            maximum size of the markov chain
        x0 : np.array, optional
            starting point of the sampling (if None, it will be sampled randomly), by default None
        v0 : np.array, optional
            initial value of the v vector of RMSProp (if None, it will be initialized used the square of the gradient of the starting point), by default None
        """
        additional_sampling_log = {}
        additional_sampling_log_u = {}

        if x0 is None:
            x0 = self.generate_random_start_Theta(conditional[1])  # (N, D)
            # x0 will need to be passed as hyperparameter for conditional[0].prior
        assert x0.shape == (self.N, self.D)

        if u0 is None:
            # ! Note: conditional[0].N = self.N, conditional[0].D = self.L
            dict_fwm = conditional[1].likelihood.evaluate_all_forward_map(x0, False)
            u0 = self.rng.lognormal(size=(self.N, self.L)) * dict_fwm["f_Theta"]

        assert u0.shape == (self.N, self.L)

        # plug u0 as observations for conditional[1]
        conditional[1].likelihood._update_observations(u0)
        self.current_Theta = conditional[1].compute_all(x0)

        # plug f(x0) as hyperparameter of conditional[0]
        # (field "forward_map_evals" in self.current_Theta)
        self.current_u = conditional[0].compute_all(
            u0,
            update_prior=True,
            theta=self.current_Theta["forward_map_evals"],
            compute_derivatives=False,
        )
        assert np.all(conditional[1].likelihood.y > 0), f"{conditional[1].likelihood.y}"
        assert np.all(conditional[0].prior.y > 0), f"{conditional[0].prior.y}"

        assert np.isnan(self.current_Theta["objective"]) == 0
        assert np.sum(np.isnan(self.current_Theta["grad"])) == 0
        assert np.isnan(self.current_u["objective"]) == 0
        # assert np.sum(np.isnan(self.current_u["grad"])) == 0

        # self.v_u = self.current_u["grad"].flatten() ** 2
        self.v_Theta = 100 * self.current_Theta["grad"].flatten() ** 2

        rng_state_array, _ = self.get_rng_state()

        self.jx_t = np.zeros((self.N * self.D,))

        # only hyperparams for the "second" conditional object need to be
        # updated with MML (if updated)
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

        # type_t_counts = np.empty((2, 2), dtype="i")
        type_t = np.zeros((2,), dtype="int")

        for t in tqdm(range(1, max_iter + 1), disable=disable_progress_bar):

            if optimize_regu_weights and (self.N > 1):

                if t >= regu_weights_optimizer.N0:
                    tau_t = self.sample_regu_hyperparams(
                        conditional[1],
                        regu_weights_optimizer,
                        t,
                        self.current_Theta["Theta"] * 1,
                    )

                    conditional[1].prior_spatial.weights = tau_t * 1

                    # recompute conditional neg log pdf and gradients with
                    # new spatial regularization parameter
                    self.current_Theta = conditional[1].compute_all(
                        self.current_Theta["Theta"],
                        self.current_Theta["forward_map_evals"],
                        self.current_Theta["nll_utils"],
                    )

                additional_sampling_log["tau"] = (
                    conditional[1].prior_spatial.weights * 1
                )

            type_t[1] = np.argmax(
                self.rng.multinomial(1, pvals=self.selection_probas[1])
            )

            if type_t[1] == 0:
                # * MTM: sample joint (u, theta)
                (
                    new_current_Theta,
                    new_current_u,
                    x_accepted_t,
                    x_log_proba_accept_t,
                ) = self.generate_new_sample_mtm(
                    t,
                    conditional,
                    copy.copy(self.current_Theta),
                    copy.copy(self.current_u),
                    self.k_mtm[1],
                    self.alpha[1],
                    self.v_Theta,
                    self.jx_t,
                    self.stochastic[1],
                )
                u_accepted_t = x_accepted_t * 1
                u_log_proba_accept_t = x_log_proba_accept_t * 1

                self.current_Theta = new_current_Theta
                self.current_u = new_current_u

            else:
                assert type_t[1] == 1

                # * sample u
                (
                    new_current_u,
                    u_accepted_t,
                    u_log_proba_accept_t,
                ) = self.generate_new_sample_u(
                    t=t,
                    conditional=conditional[0],
                    current=copy.copy(self.current_u),
                    stochastic=self.stochastic[0],
                    sigma_m=conditional[0].prior.sigma,
                    sigma_a=conditional[0].likelihood.sigma,
                    theta=self.current_Theta["forward_map_evals"],
                )

                self.current_u = new_current_u
                assert np.all(self.current_u["Theta"] > 0)

                # * u updated, update conditional[1]
                # ! define new value of u as observations for conditional[1]
                # ! conditional[0].prior._update_observations(self.current_u["Theta"])
                # ! define new value of u as observations for conditional[1]
                conditional[1].likelihood._update_observations(self.current_u["Theta"])

                # * sample \theta
                (
                    new_current_Theta,
                    x_accepted_t,
                    x_log_proba_accept_t,
                ) = self.generate_new_sample_pmala_rmsprop(
                    t,
                    conditional[1],
                    copy.copy(self.current_Theta),
                    self.lambda_[1],
                    self.v_Theta,
                    self.eps0[1],
                    self.alpha[1],
                    self.jx_t,
                    self.compute_correction_term[1],
                    self.stochastic[1],
                )

            # * if the memory is empty : initialize it
            if saver.memory == {}:
                additional_sampling_log["v"] = (
                    self.v_Theta.reshape((self.N, self.D)) * 1
                )
                additional_sampling_log["type_t"] = type_t[1]
                additional_sampling_log["accepted_t"] = x_accepted_t
                additional_sampling_log["log_proba_accept_t"] = x_log_proba_accept_t

                # additional_sampling_log_u["v"] = (
                #     self.v_u.reshape(self.current_u["Theta"].shape) * 1
                # )
                additional_sampling_log_u["type_t"] = type_t[0]
                additional_sampling_log_u["accepted_t"] = u_accepted_t
                additional_sampling_log_u["log_proba_accept_t"] = u_log_proba_accept_t

                dict_objective_u = conditional[0].compute_all_for_saver(
                    self.current_u["Theta"],
                    self.current_u["forward_map_evals"],
                    self.current_u["nll_utils"],
                    update_prior=False,  # update normally not needed, already done in PMALA
                    theta=self.current_Theta["forward_map_evals"],
                )

                dict_objective_Theta = conditional[1].compute_all_for_saver(
                    self.current_Theta["Theta"],
                    self.current_Theta["forward_map_evals"],
                    self.current_Theta["nll_utils"],
                )

                saver.initialize_memory(
                    max_iter,
                    t,
                    x=self.current_Theta["Theta"],
                    u=self.current_u["Theta"],
                    forward_map_evals=self.current_Theta["forward_map_evals"],
                    forward_map_evals_u=self.current_u["forward_map_evals"],
                    nll_utils=self.current_Theta["nll_utils"],
                    nll_utils_u=self.current_u["nll_utils"],
                    dict_objective=dict_objective_Theta,
                    dict_objective_u=dict_objective_u,
                    additional_sampling_log=additional_sampling_log,
                    additional_sampling_log_u=additional_sampling_log_u,
                )

            if saver.check_need_to_update_memory(t):
                # print(f"updating memory at t={t}")
                additional_sampling_log["v"] = (
                    self.v_Theta.reshape((self.N, self.D)) * 1
                )
                additional_sampling_log["type_t"] = type_t[1]
                additional_sampling_log["accepted_t"] = x_accepted_t
                additional_sampling_log["log_proba_accept_t"] = x_log_proba_accept_t

                # additional_sampling_log_u["v"] = (
                #     self.v_u.reshape(self.current_u["Theta"].shape) * 1
                # )
                additional_sampling_log_u["type_t"] = type_t[0]
                additional_sampling_log_u["accepted_t"] = u_accepted_t
                additional_sampling_log_u["log_proba_accept_t"] = u_log_proba_accept_t

                dict_objective_u = conditional[0].compute_all_for_saver(
                    self.current_u["Theta"],
                    self.current_u["forward_map_evals"],
                    self.current_u["nll_utils"],
                    update_prior=False,  # update normally not needed, already done in PMALA
                    theta=self.current_Theta["forward_map_evals"],
                )

                dict_objective_Theta = conditional[1].compute_all_for_saver(
                    self.current_Theta["Theta"],
                    self.current_Theta["forward_map_evals"],
                    self.current_Theta["nll_utils"],
                )

                rng_state_array, rng_inc_array = self.get_rng_state()

                saver.update_memory(
                    t,
                    x=self.current_Theta["Theta"],
                    u=self.current_u["Theta"],
                    forward_map_evals=self.current_Theta["forward_map_evals"],
                    forward_map_evals_u=self.current_u["forward_map_evals"],
                    nll_utils=self.current_Theta["nll_utils"],
                    nll_utils_u=self.current_u["nll_utils"],
                    dict_objective=dict_objective_Theta,
                    dict_objective_u=dict_objective_u,
                    additional_sampling_log=additional_sampling_log,
                    additional_sampling_log_u=additional_sampling_log_u,
                    rng_state_array=rng_state_array,
                    rng_inc_array=rng_inc_array,
                )

            if saver.check_need_to_save(t):
                # print(f"saving memory at t={t}")
                saver.save_to_file()

    def generate_new_sample_u(
        self,
        t: int,
        conditional,
        current: dict,
        stochastic: bool,
        sigma_m: np.ndarray,
        sigma_a: np.ndarray,
        theta={},
    ):
        r"""generates a new u sample a basic 1D MH step

        Parameters
        ----------
        t : int
            current iteration index
        score_model : ScoreModel
            negative log conditional class
        current
            ... Should be updated in place.
        theta: dict
            Hyperparameters of the conditional distribution to be sampled from.

        Returns
        -------
        accepted : bool
            wether or not the candidate was accepted
        log_proba_accept : float
            log of the acceptance proba
        """
        # * generate random
        zeta = self.rng.uniform(0, 1, size=(self.N, self.L))

        z_t_multi = self.rng.lognormal(-(sigma_m**2) / 2, sigma_m)  # (N, L)

        z_t_add = self.rng.normal(scale=sigma_a)  # (N, L)

        random_seed = self.rng.integers(0, 1_000_000_000)
        z_t_add = utils.correct_add_proposal(
            z_t_add, current["Theta"], sigma_a, random_seed
        )  # (N, L)

        assert z_t_multi.shape == (self.N, self.L)
        assert z_t_add.shape == (self.N, self.L)

        candidate = np.where(
            zeta < 0.5,
            current["Theta"] * z_t_multi,  # (N, L)
            current["Theta"] + z_t_add,  # (N, L)
        )

        # candidate = theta["f_Theta"] * z_t  # (N, L)

        forward_map_evals = conditional.likelihood.evaluate_all_forward_map(
            candidate, False
        )
        nll_utils = conditional.likelihood.evaluate_all_nll_utils(forward_map_evals)
        score_candidate = conditional.neglog_pdf(
            candidate,  # .reshape(conditional.N, conditional.D),
            forward_map_evals,
            nll_utils,
            update_prior=True,
            theta=theta,
            full=True,
        )

        forward_map_evals = conditional.likelihood.evaluate_all_forward_map(
            current["Theta"], False
        )
        nll_utils = conditional.likelihood.evaluate_all_nll_utils(forward_map_evals)
        score_current = conditional.neglog_pdf(
            current["Theta"],  # .reshape(conditional.N, conditional.D),
            forward_map_evals,
            nll_utils,
            update_prior=True,
            theta=theta,
            full=True,
        )

        assert score_candidate.shape == (self.N, self.L)
        assert score_current.shape == (self.N, self.L)

        # * accept reject
        logpdf_current = -score_current  # (N * L,)
        logpdf_candidate = -score_candidate  # (N * L,)

        if stochastic:
            exp_lognormal = (
                np.exp(
                    -0.5
                    * (
                        (
                            np.log(candidate)
                            - np.log(current["Theta"])
                            + sigma_m**2 / 2
                        )
                        / sigma_m
                    )
                    ** 2
                )
                / sigma_m
            )  # (N, L)

            exp_normal = (
                np.exp(-0.5 * ((candidate - current["Theta"]) / sigma_a) ** 2) / sigma_a
            )  # (N, L)

            Z_trunc_cand_from_curr = 1 - ndtr(-current["Theta"] / sigma_a)  # (N, L)
            Z_trunc_curr_from_cand = 1 - ndtr(-candidate / sigma_a)  # (N, L)

            q_cand_from_curr = (
                exp_lognormal / candidate + exp_normal / Z_trunc_curr_from_cand
            )

            q_curr_from_cand = (
                exp_lognormal / candidate + exp_normal / Z_trunc_cand_from_curr
            )

            # log_ratio_transition_proba = np.log(candidate) - np.log(current["Theta"])
            log_ratio_transition_proba = np.log(q_curr_from_cand) - np.log(
                q_cand_from_curr
            )

            log_proba_accept = (
                logpdf_candidate - logpdf_current + log_ratio_transition_proba
            )
            log_u = np.log(self.rng.uniform(0, 1, size=(self.N, self.L)))

            accept_arr = log_u < log_proba_accept  # (N * L,)
        else:
            accept_arr = logpdf_current > logpdf_candidate  # (N * L,)

        new_u = np.where(accept_arr, candidate, current["Theta"])  # (N, L)

        new_current_u = conditional.compute_all(
            new_u,
            update_prior=True,
            theta=theta,
            compute_derivatives=False,
        )

        accepted = (log_u < log_proba_accept).mean() > 0
        log_proba_acept = np.nanmean(log_proba_accept)

        return new_current_u, accepted, log_proba_acept

    def generate_new_sample_pmala_rmsprop(
        self,
        t,
        conditional,
        current,
        lambda_,
        v,
        eps0,
        alpha,
        j_t,
        compute_correction_term,
        stochastic,
        theta={},
    ):
        r"""generates a new sample using the position-dependent MALA transition kernel

        Parameters
        ----------
        t : int
            current iteration index
        score_model : ScoreModel
            negative log conditional class
        current
            ... Should be updated in place.
        lambda_ :
            ...
        v : ...
            ... Should be updated in place.
        eps0: ...
            ...
        alpha: ...
            ...
        j_t: ...
            ... Should be updated in place
        theta: dict
            Hyperparameters of the conditional distribution to be sampled from.

        Returns
        -------
        accepted : bool
            wether or not the candidate was accepted
        log_proba_accept : float
            log of the acceptance proba
        """
        grad_t = current["grad"].flatten()

        diag_G_t = 1 / (lambda_ + np.sqrt(v))
        assert np.all(diag_G_t > 0)

        # generate random
        z_t = self.rng.standard_normal(size=conditional.N * conditional.D)
        z_t *= np.sqrt(eps0 * diag_G_t)

        # bias correction term
        if compute_correction_term:
            # recursive version
            # correction = -1 / 2 * diag_G_t ** 2 / np.sqrt(v) * self.u

            # only with corresponding term
            hess_diag_t = current["hess_diag"].flatten()
            correction = (
                -(1 - alpha)
                * alpha**j_t
                * (diag_G_t**2)
                / np.sqrt(v)
                * grad_t
                * hess_diag_t
            )
        else:
            correction = np.zeros((conditional.N * conditional.D,))

        # combination
        mu_current = (
            current["Theta"].flatten()
            - eps0 / 2 * diag_G_t * grad_t
            + eps0 * correction
        )

        if stochastic:
            candidate = mu_current + z_t  # (N * D,)

            log_q_candidate_given_current = -1 / 2 * np.sum(np.log(diag_G_t)) - 1 / (
                2 * eps0
            ) * np.sum((candidate - mu_current) ** 2 / diag_G_t)

            # * compute log_q of candidate given current
            candidate_all = conditional.compute_all(
                candidate.reshape(conditional.N, conditional.D),
                update_prior=True,
                theta=theta,
            )
            grad_cand = candidate_all["grad"].flatten()
            v_cand = alpha * v + (1 - alpha) * grad_cand**2
            diag_G_cand = 1 / (lambda_ + np.sqrt(v_cand))

            if compute_correction_term:
                hess_diag_cand = candidate_all["hess_diag"].flatten()

                correction_cand = -(
                    (1 - alpha)
                    * diag_G_cand**2
                    / np.sqrt(v_cand)
                    * grad_cand
                    * hess_diag_cand
                )
            else:
                correction_cand = np.zeros((conditional.N * conditional.D,))

            mu_cand = (
                candidate - eps0 / 2 * diag_G_cand * grad_cand + eps0 * correction_cand
            )

            log_q_current_given_candidate = -1 / 2 * np.sum(np.log(diag_G_cand)) - 1 / (
                2 * eps0
            ) * np.sum((current["Theta"].flatten() - mu_cand) ** 2 / diag_G_cand)

            # * compute proba accept
            logpdf_current = -current["objective"] * 1
            logpdf_candidate = -candidate_all["objective"] * 1

            log_proba_accept = (
                logpdf_candidate
                - logpdf_current
                + log_q_current_given_candidate
                - log_q_candidate_given_current
            )
            log_u = np.log(self.rng.uniform(0, 1))
            # print(
            #     f"{log_u:.2e}, {log_proba_accept:.4e}, {logpdf_candidate:.4e},, {logpdf_current:.4e}, {log_q_current_given_candidate:.4e}, {log_q_candidate_given_current:.4e}"
            # )

            if log_u < log_proba_accept:
                v = v_cand * 1
                j_t = np.zeros((conditional.N * conditional.D,))
                return candidate_all, True, log_proba_accept
            else:
                if conditional.prior is not None:
                    # revert prior to initial state
                    conditional.prior._update_observations(current["Theta"])
                j_t += 1
                v = v_cand * 1
                return current, False, log_proba_accept

        # * in case we are doing optimization and not sampling
        candidate_all = conditional.compute_all(
            mu_current.reshape(
                (conditional.N, conditional.D), update_prior=True, theta=theta
            )
        )
        if candidate_all["objective"] < current["objective"]:
            current = copy.copy(candidate_all)
            accept = True
            proba = 1
        else:
            candidate = mu_current + z_t  # (N * D,)
            candidate_all = conditional.compute_all(
                candidate.reshape(
                    (conditional.N, conditional.D), update_prior=True, theta=theta
                )
            )
            if candidate_all["objective"] < current["objective"]:
                current = copy.copy(candidate_all)
                accept = True
                proba = 1
            else:
                if conditional.prior is not None:
                    # revert prior to initial state
                    conditional.prior._update_observations(current["Theta"])
                accept = False
                proba = 0

        grad_tp1 = candidate_all["grad"].flatten()
        v = alpha * v + (1 - alpha) * grad_tp1**2

        assert np.sum(np.isnan(v)) == 0.0
        assert np.sum(np.isnan(current["Theta"])) == 0.0

        return current, accept, proba

    def generate_new_sample_mtm(
        self,
        t: int,
        conditional: list[Posterior],
        current_Theta: dict[str, np.ndarray],
        current_u: dict[str, np.ndarray],
        k_mtm: int,
        alpha: float,
        v: np.ndarray,
        j_t: np.ndarray,
        stochastic: bool,
    ):
        r"""generates a new sample using the MTM transition kernel

        Parameters
        ----------
        t : int
            current iteration index
        conditional : list[Posterior]
            first: u, second: theta
        current : dict
            ...
        k_mtm : int
            number of candiates to be generated by MTM
        alpha : ...
            ...
        v : ...
            ... Should be updated in place.
        j_t : ...
            ... Should be updated in place.

        Returns
        -------
        accepted : bool
            wether or not the candidate was accepted
        log_proba_accept : float
            log of the acceptance proba
        """
        new_Theta = current_Theta["Theta"] * 1  # (N, D)
        new_u = current_u["Theta"] * 1  # (N, D)

        accept_total = np.zeros((self.N,))
        log_rg_total = np.zeros((self.N,))

        list_idx = np.array(list(conditional[1].dict_sites.keys()))
        # self.rng.shuffle(list_idx)

        for idx_site in list_idx:
            idx_pix = conditional[1].dict_sites[idx_site]
            n_pix = idx_pix.size

            # * get relevant sigma_a for u candidates correction
            sigma_a = np.zeros((n_pix, k_mtm, self.L))

            for i_pix in range(n_pix):
                sigma_a[i_pix, :, :] = conditional[0].likelihood.sigma[
                    idx_pix[i_pix], :
                ][None, :] * np.ones((k_mtm, self.L))

            sigma_a = sigma_a.reshape((n_pix * k_mtm, self.L))

            sigma_a /= 5.0  # to generate candidates with increased accept proba

            # * theta: generate candidates
            candidates_pix_theta = np.zeros((n_pix, k_mtm + 1, self.D))
            candidates_pix_theta[:, :-1, :] = self.generate_random_start_Theta_1pix(
                new_Theta, conditional[1], idx_pix, k_mtm
            )
            candidates_pix_theta[:, -1, :] = current_Theta["Theta"][idx_pix, :] * 1
            candidates_pix_theta = candidates_pix_theta.reshape(
                (n_pix * (k_mtm + 1), self.D)
            )  # (n_pix * (k_mtm+1), D)

            # * theta: evaluate likelihood p(u \vert \theta)
            neglogpdf_candidates, forward_map_evals = conditional[
                1
            ].likelihood.neglog_pdf_candidates(
                candidates_pix_theta,
                idx=idx_pix,
                x_t=new_Theta * 1,
                return_forward_map_evals=True,
            )  # (n_pix * (k_mtm+1),)
            assert neglogpdf_candidates.shape == (n_pix * (k_mtm + 1),)

            candidates_pix_theta = candidates_pix_theta.reshape(
                (n_pix, k_mtm + 1, self.D)
            )

            # * u: generate candidates
            f_Theta = (
                forward_map_evals["f_Theta"]
                .reshape((n_pix, k_mtm + 1, self.L))[:, :-1, :]
                .reshape((n_pix * k_mtm, self.L))
            )

            candidates_pix_u = np.zeros((n_pix, k_mtm + 1, self.L))

            random_seed = self.rng.integers(0, 1_000_000_000)
            z_t_add = self.rng.normal(scale=sigma_a)  # (n_pix * k_mtm, L)
            z_t_add = utils.correct_add_proposal(
                z_t_add, f_Theta, sigma_a, random_seed
            )  # (n_pix * k_mtm, L)

            candidates_pix_u[:, :-1, :] = (f_Theta + z_t_add).reshape(
                (n_pix, k_mtm, self.L)
            )
            candidates_pix_u[:, -1, :] = current_u["Theta"][idx_pix, :] * 1

            candidates_pix_u = candidates_pix_u.reshape(
                (n_pix * (k_mtm + 1), self.L)
            )  # (n_pix * (k_mtm+1), D)

            # * u: evaluate likelihood p(y \vert u)
            neglogpdf_candidates_u = conditional[0].likelihood.neglog_pdf_candidates(
                candidates_pix_u,
                idx=idx_pix,
                x_t=new_u * 1,
            )  # (n_pix * (k_mtm+1),)

            candidates_pix_u = candidates_pix_u.reshape((n_pix, k_mtm + 1, self.L))

            neglogpdf_candidates += neglogpdf_candidates_u
            neglogpdf_candidates = neglogpdf_candidates.reshape((n_pix, k_mtm + 1))

            # * if optimization: define challenger with conditional conditional
            # * instead of likelihood, and only keep if better than current
            if not stochastic:
                neglogpdf_candidates += conditional[1].partial_neglog_pdf_priors(
                    new_Theta * 1, idx_pix, candidates_pix_theta
                )  # (n_pix, k_mtm)

                # * choose challengers (1 for each of the n_pix pixels)
                idx_challengers = np.argmin(
                    neglogpdf_candidates[:, :-1], axis=1
                )  # (n_pix,)
                assert idx_challengers.shape == (n_pix,)

                neglogpdf_challengers = neglogpdf_candidates[
                    np.arange(n_pix), idx_challengers
                ]
                assert neglogpdf_challengers.shape == (
                    n_pix,
                ), neglogpdf_challengers.shape

                challengers_theta = candidates_pix_theta[
                    np.arange(n_pix), idx_challengers, :
                ]
                challengers_u = candidates_pix_u[np.arange(n_pix), idx_challengers, :]

                shape_theta = challengers_theta.shape
                shape_u = challengers_u.shape
                assert shape_theta == (n_pix, self.D), shape_theta
                assert shape_u == (n_pix, self.L), shape_u

                # * compute values of corresponding pixels in current x
                current_theta = candidates_pix_theta[:, -1, :] * 1
                current_u = candidates_pix_u[:, -1, :] * 1
                neglogpdf_current = neglogpdf_candidates[:, -1] * 1

                assert current_theta.shape == (n_pix, self.D)
                assert current_u.shape == (n_pix, self.L)
                assert neglogpdf_current.shape == (n_pix,)

                # * select best pixels
                accept_arr = neglogpdf_challengers < neglogpdf_current

                new_Theta[idx_pix, :] = np.where(
                    accept_arr[:, None],
                    challengers_theta,  # (n_pix, D)
                    current_theta,  # (n_pix, D)
                )
                new_Theta[idx_pix, :] = np.where(
                    accept_arr[:, None],
                    challengers_u,  # (n_pix, L)
                    current_u,  # (n_pix, L)
                )

                # * save which pixels were accepted
                accept_total[idx_pix] = accept_arr * 1

            # *------

            else:
                if conditional[1].prior_spatial is not None:
                    nlratio_prior_proposal = utils.compute_nlratio_prior_proposal(
                        new_Theta * 1,
                        conditional[1].prior_spatial.list_edges,
                        conditional[1].prior_spatial.weights,
                        idx_pix,
                        candidates_pix_theta,
                    )
                    assert nlratio_prior_proposal.shape == (n_pix, k_mtm + 1)
                    neglogpdf_candidates += nlratio_prior_proposal

                neglogpdf_candidates_min = np.amin(
                    neglogpdf_candidates, axis=1, keepdims=True
                )
                neglogpdf_candidates -= neglogpdf_candidates_min

                pdf_candidates = np.exp(-neglogpdf_candidates)  # (n_pix, k_mtm)

                log_numerators = np.log(np.sum(pdf_candidates[:, :-1], axis=1))
                assert log_numerators.shape == (n_pix,), log_numerators.shape

                # * choose challenger candidate
                weights = softmax(-neglogpdf_candidates[:, :-1], axis=1)
                idx_challengers = np.zeros((n_pix,), dtype=int)
                for i in range(n_pix):
                    idx_challengers[i] = self.rng.choice(k_mtm, p=weights[i])

                challengers_theta = candidates_pix_theta[
                    np.arange(n_pix), idx_challengers, :
                ]  # (n_pix, D)
                challengers_u = candidates_pix_u[
                    np.arange(n_pix), idx_challengers, :
                ]  # (n_pix, L)
                neglogpdf_challengers = neglogpdf_candidates[
                    np.arange(n_pix), idx_challengers
                ]
                assert neglogpdf_challengers.shape == (
                    n_pix,
                ), neglogpdf_challengers.shape

                # * denominator
                log_denominators = np.log(
                    np.sum(pdf_candidates, axis=1) - np.exp(-neglogpdf_challengers)
                )
                assert log_denominators.shape == (n_pix,), log_denominators.shape

                # * accept-reject
                log_rg = log_numerators - log_denominators
                log_u = np.log(self.rng.uniform(0, 1, size=n_pix))
                accept_arr = log_u < log_rg

                # print("candidate_pix")
                # print(candidates_pix)
                # print("neglogpdf")
                # print(neglogpdf_candidates)
                # print("weights")
                # print(weights)
                # print("challenger")
                # print(idx_challengers, neglogpdf_challengers)
                # print("log_numerators")
                # print(log_numerators)
                # print("log_denominators")
                # print(log_denominators)
                # print("log_rg")
                # print(log_rg)
                # print("accept_arr")
                # print(accept_arr)
                # print()

                new_Theta[idx_pix, :] = np.where(
                    accept_arr[:, None],
                    challengers_theta,  # (n_pix, D)
                    candidates_pix_theta[:, -1, :],  # current, (n_pix, D)
                )
                new_u[idx_pix, :] = np.where(
                    accept_arr[:, None],
                    challengers_u,  # (n_pix, D)
                    candidates_pix_u[:, -1, :],  #  current, (n_pix, D)
                )

                accept_total[idx_pix] = accept_arr * 1
                log_rg_total[idx_pix] = log_rg * 1

                # * theta: re-initialize j for new point
                new_j_t = j_t.reshape((self.N, self.D))
                new_j_t[idx_pix, :] = np.where(
                    accept_arr[:, None], 0.0, new_j_t[idx_pix, :]
                )
                j_t = new_j_t.flatten()  # (ND,)

        # *------
        # * once all sites have been dealt with, update global parameters

        forward_map_evals = conditional[1].likelihood.evaluate_all_forward_map(
            new_Theta, True
        )

        new_current_u = conditional[0].compute_all(
            new_u,
            update_prior=True,
            theta=forward_map_evals,
            compute_derivatives=False,
        )

        conditional[1].likelihood._update_observations(new_current_u["Theta"])

        new_current_Theta = conditional[1].compute_all(
            new_Theta, forward_map_evals=forward_map_evals
        )

        # theta: update v
        new_v = v.reshape((self.N, self.D))
        new_v = np.where(
            accept_total[:, None],
            alpha * new_v + (1 - alpha) * new_current_Theta["grad"] ** 2,
            new_v,
        )
        v = new_v.flatten()

        if not stochastic:
            return (
                new_current_Theta,
                new_current_u,
                accept_total.mean() > 0,
                accept_total.mean(),
            )
        else:
            return (
                new_current_Theta,
                new_current_u,
                accept_total.mean(),
                log_rg_total.mean(),
            )
