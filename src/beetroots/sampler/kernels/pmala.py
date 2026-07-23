try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from typing import Dict, List, Optional

from beetroots.modelling.component_distribution import ComponentDistribution
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.kernels.abstract_kernel import KernelUtils
from beetroots.sampler.kernels.abstract_kernel_metropolis_hastings import (
    KernelMetropolisHastings,
)


class PMALA(KernelUtils):
    __slots__ = (
        "key",
        "is_stochastic",
        "alpha",
        "eps0",
        "lambda_",
        "compute_correction_term",
        "add_perturbation",
        "stochastic",
        "compute_derivatives_2nd_order",
        "v",
        "j_t",
        "chromatic_gibbs",
        "objective_type",
        "var_shape",
        "acceptance_wise_type",
        "plot_name",
        "last_updated",
    )

    def __init__(
        self,
        key: str,
        is_stochastic: bool,
        alpha: float,
        eps0: float,
        lambda_: float,
        compute_correction_term: bool,
        add_perturbation: bool,
        stochastic: bool,
        acceptance_wise_type: str,
        plot_name: str = "PMALA",
        **kwargs,
    ):
        self.key = key
        r"""str: Key for the target distribution."""

        self.is_stochastic = is_stochastic
        r"""bool: If True, the PMALA kernel is stochastic. If False, it is deterministic."""

        if not isinstance(plot_name, str):
            raise TypeError("plot_name should be a string")
        self.plot_name = plot_name

        if not all(isinstance(param, float) for param in [alpha, eps0, lambda_]):
            raise ValueError("Parameters alpha, eps0, and lambda_ must be float.")
        self.alpha = alpha
        self.eps0 = eps0
        self.lambda_ = lambda_

        if not all(
            isinstance(param, bool)
            for param in [compute_correction_term, add_perturbation, stochastic]
        ):
            raise ValueError(
                "Parameters compute_correction_term, add_perturbation, and stochastic must be bool."
            )
        self.compute_correction_term = compute_correction_term
        self.add_perturbation = add_perturbation
        self.stochastic = stochastic

        assert acceptance_wise_type in ["pixelwise", "full", "entire_map"]
        self.acceptance_wise_type = acceptance_wise_type

        self.v = None
        r"""dict: RMSProp gradient variance vector for each target distribution, denoted :math:`v` in the article"""
        self.j_t = None
        r"""dict: contains the number of iterations since last acceptance for each target distribution"""
        self.chromatic_gibbs = None
        self.objective_type = None
        self.var_shape = None
        self.last_updated = -1

    def initialize_hyperparameters(self, current: dict) -> None:
        self.v = current[self.key]["grad"].flatten() ** 2
        assert xp.sum(xp.isnan(self.v)) == 0.0

        self.j_t = xp.zeros(current[self.key]["var"].shape)

    def external_update_hyperparameters(
        self, t: int, current: dict, acceptance_stats: dict
    ) -> None:
        if not t == self.last_updated:
            self.var_shape = current[self.key]["var"].shape

            new_j_t = self.j_t.reshape(self.var_shape) * 1
            new_j_t = xp.where(
                acceptance_stats[self.key]["accepted_t"][:, None], 0.0, new_j_t
            )
            self.j_t = new_j_t.flatten()

            new_v = self.v.reshape(self.var_shape) * 1
            new_v = xp.where(
                acceptance_stats[self.key]["accepted_t"][:, None],
                # self.alpha * new_v + (1 - self.alpha) * current[self.key]["grad"] ** 2,
                current[self.key]["grad"]
                ** 2,  # update v only if accepted, else keep the same value
                new_v,
            )
            self.v = new_v.flatten()
            assert xp.sum(xp.isnan(self.v)) == 0.0
            assert xp.sum(xp.isinf(self.v)) == 0.0
            self.last_updated = t * 1

    def internal_update_hyperparameters(
        self, t: int, v_cand: xp.ndarray, accept_arr: xp.ndarray, idx_pix: xp.ndarray
    ) -> None:
        N, D = self.var_shape

        v = self.v.reshape((N, D)) * 1
        v[idx_pix, :] = v_cand * 1
        self.v = v.flatten()

        j = self.j_t.reshape((N, D)) * 1
        j[idx_pix, :] = xp.where(
            accept_arr[:, None],
            0.0,  # reset to 0 if accept
            j[idx_pix, :] + 1,  # else add 1
        )
        self.j_t = j.flatten()

        self.last_updated = t * 1

    def _dict_parameters_saving(self, **kwargs) -> dict:
        return {"v": self.v, "j_t": self.j_t, "last_updated": self.last_updated}


class KernelPMALA(PMALA, KernelMetropolisHastings):
    r"""PMALA kernel class."""

    def __init__(
        self,
        key: str,
        alpha: float,
        eps0: float,
        lambda_: float,
        is_stochastic: bool,
        compute_correction_term: bool = False,
        add_perturbation: bool = True,
        stochastic: bool = True,
        acceptance_wise_type: str = "pixelwise",
        **kwargs,
    ):
        super().__init__(
            key,
            is_stochastic,
            alpha,
            eps0,
            lambda_,
            compute_correction_term,
            add_perturbation,
            stochastic,
            acceptance_wise_type,
            **kwargs,
        )

        self.compute_derivatives_2nd_order = (
            self.stochastic and self.compute_correction_term
        )

    def sample(
        self,
        current: dict,
        idx_pix: xp.ndarray,
        rng: xp.random.Generator,
    ) -> xp.ndarray:
        current_var = current["var"] * 1  # (N, D)
        n_pix = idx_pix.size

        grad_t = current["grad"][idx_pix] * 1
        v_current = self.v.reshape(current_var.shape)[idx_pix, :] * 1

        # generate random
        diag_G_t = 1 / (self.lambda_ + xp.sqrt(v_current))  # (n_pix, D)

        assert xp.all(diag_G_t > 0), (
            f"{diag_G_t}, {self.lambda_ + xp.sqrt(self.v)}, {self.v}"
        )

        size_var_sites = (
            n_pix,
            *current_var.shape[1:],
        )  # Shape of the variable to sample restricted to the current sites for the first dimension.

        # bias correction term
        if self.compute_correction_term:
            hess_diag_t = current["hess_diag"][idx_pix] * 1
            j_t = self.j_t.reshape(current_var.shape)[idx_pix, :] * 1

            correction = (
                -(1 - self.alpha)
                * self.alpha**j_t
                * (diag_G_t**2)
                / xp.sqrt(v_current)
                * grad_t
                * hess_diag_t
            )  # (n_pix, D)

            n_inf = xp.sum(~xp.isfinite(correction))
            if n_inf > 0:
                print(f"num of nan in correction term: {n_inf}")
            correction = xp.nan_to_num(correction)  # ? nécessaire ?
        else:  # if no correction term then 0
            correction = xp.zeros(size_var_sites)

        # combination
        mu_current = (
            current_var[idx_pix, :]
            - self.eps0 / 2 * diag_G_t * grad_t
            + self.eps0 * correction
        )  # (n_pix, D)

        candidate = mu_current * 1
        if self.add_perturbation:
            """
            # Noise removed for trials. Should be added back for true sampling.
            # (n_pix, D), simulates Gaussian with mean mu_current and std dev sqrt(eps0[ * diag_G_t). It is the PMALA proposal (Langevin step)
            """
            scale = xp.sqrt(self.eps0 * diag_G_t)
            z_t = rng.standard_normal(size=size_var_sites)
            z_t *= scale  # (n_pix, D)
            candidate += z_t

        return candidate, diag_G_t, mu_current

    def neglog_proposal_candidate_given_current(
        self,
        candidate: xp.ndarray,
        mu_current: xp.ndarray,
        diag_G_t: xp.ndarray,
    ) -> xp.ndarray:
        log_proposal_candidate_given_current = -1 / 2 * xp.sum(
            xp.log(diag_G_t), axis=1
        ) - 1 / (2 * self.eps0) * xp.sum(
            (candidate - mu_current) ** 2 / diag_G_t, axis=1
        )  # (n_pix,)

        return -log_proposal_candidate_given_current

    def neglog_proposal_current_given_candidate(
        self,
        current: Dict[str, xp.ndarray],
        candidate: Dict[str, xp.ndarray],
        list_components: Dict[str, ComponentDistribution],
        idx_pix: xp.ndarray,
        compute_derivatives_components: Optional[List[bool]] = None,
    ) -> xp.ndarray:
        self.var_shape = current[self.key]["var"].shape
        if compute_derivatives_components is None:
            compute_derivatives_components = [True] * len(list_components)

        candidate_all = {
            self.objective_type: xp.zeros(
                idx_pix.size,
            ),
            "grad": xp.zeros((idx_pix.size, current[self.key]["var"].shape[1])),
            "hess_diag": xp.zeros((idx_pix.size, current[self.key]["var"].shape[1])),
        }

        for comp, compute_deriv in zip(list_components, compute_derivatives_components):
            comp.evaluate_all_nlpdf_utils(
                candidate,
                idx_pix=idx_pix,
                compute_derivatives=compute_deriv,
                compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
                deriv_var_name=self.key,
            )

            candidate_all[self.objective_type] += comp.neglog_pdf(
                pixelwise=True,
                update_nlpdf_utils=False,
                idx_pix=idx_pix,
                chromatic_gibbs=self.chromatic_gibbs,
            )
            if compute_deriv:
                candidate_all["grad"] += comp.gradient_neglog_pdf(
                    update_nlpdf_utils=False, idx_pix=idx_pix, deriv_var_name=self.key
                )
                if self.compute_correction_term:
                    candidate_all["hess_diag"] += comp.hess_diag_neglog_pdf(
                        update_nlpdf_utils=False, idx_pix=idx_pix
                    )
                else:
                    candidate_all["hess_diag"] = None

        assert xp.isnan(candidate_all[self.objective_type]).sum() == 0
        assert xp.isnan(candidate_all["grad"]).sum() == 0

        grad_cand = candidate_all["grad"]
        v_current = self.v.reshape(self.var_shape)[idx_pix, :] * 1
        v_cand = self.alpha * v_current + (1 - self.alpha) * grad_cand**2  # (n_pix, D)
        diag_G_cand = 1 / (self.lambda_ + xp.sqrt(v_cand))  # (n_pix, D)

        if self.compute_correction_term:
            hess_diag_cand = candidate_all["hess_diag"] * 1

            correction_cand = -(
                (1 - self.alpha)
                * diag_G_cand**2
                / (
                    2 * xp.sqrt(v_cand)
                )  # NOTE: the factor 2 was missing in the initial approach. To be confirmed.
                * grad_cand
                * hess_diag_cand
            )
        else:
            correction_cand = xp.zeros(candidate[self.key]["var"][idx_pix].shape)

        mu_cand = (
            candidate[self.key]["var"][idx_pix]
            - self.eps0 / 2 * diag_G_cand * grad_cand
            + self.eps0 * correction_cand
        )  # (n_pix, D)

        log_q_current_given_candidate = -1 / 2 * xp.sum(
            xp.log(diag_G_cand), axis=1
        ) - 1 / (2 * self.eps0) * xp.sum(
            (current[self.key]["var"][idx_pix, :] - mu_cand) ** 2 / diag_G_cand, axis=1
        )  # (n_pix,)

        shape_q = log_q_current_given_candidate.shape
        assert shape_q == (idx_pix.size,), f"{shape_q}"

        return (
            -log_q_current_given_candidate,
            candidate_all[self.objective_type],
            v_cand,
        )

    def get_new_sample_sampling(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        **kwargs,
    ) -> tuple:
        new_var = current[self.key]["var"] * 1  # (N, D)
        self.var_shape = new_var.shape

        if self.acceptance_wise_type == "pixelwise":
            summing_axes = xp.arange(1, len(self.var_shape))
            accept_total = xp.zeros((self.var_shape[0],))
            log_proba_accept_total = xp.zeros((self.var_shape[0],))
        elif self.acceptance_wise_type == "full":
            summing_axes = xp.array([])
            accept_total = xp.zeros(self.var_shape)
            log_proba_accept_total = xp.zeros(self.var_shape)
        elif self.acceptance_wise_type == "entire_map":
            summing_axes = xp.arange(len(self.var_shape))
            accept_total = xp.zeros((1,))
            log_proba_accept_total = xp.zeros((1,))

        list_idx = xp.array(list(target_distributions[self.key].dict_sites.keys()))
        self.chromatic_gibbs = not (len(list_idx) == 1)
        self.objective_type = (
            "objective_pix_chromatic"
            if self.chromatic_gibbs
            else "objective_pix_global"
        )

        for idx_site in list_idx:
            idx_pix = target_distributions[self.key].dict_sites[idx_site]

            candidate, diag_G_t, mu_current = self.sample(
                current=current[self.key],
                idx_pix=idx_pix,
                rng=rng,
            )

            neglog_proposal_candidate_given_current = (
                self.neglog_proposal_candidate_given_current(
                    candidate=candidate,
                    mu_current=mu_current,
                    diag_G_t=diag_G_t,
                )
            )

            candidate_dict = {k: {"var": current[k]["var"] * 1} for k in current.keys()}
            candidate_dict[self.key]["var"][idx_pix] = candidate * 1

            neglog_proposal_current_given_candidate, neglogpdf_candidate, v_cand = (
                self.neglog_proposal_current_given_candidate(
                    current=current,
                    candidate=candidate_dict,
                    list_components=list(
                        target_distributions[self.key].distribution_components.values()
                    ),
                    idx_pix=idx_pix,
                )
            )

            neglogpdf_current = current[self.key][self.objective_type][idx_pix]

            new_var = current[self.key]["var"] * 1
            candidate_accepted, accept_arr, log_proba_accept = self.accept_reject(
                candidates=candidate,
                current_var=new_var[idx_pix],
                neglogpdf_candidates=neglogpdf_candidate,
                neglogpdf_current=neglogpdf_current,
                neglog_q_current_given_candidate=neglog_proposal_current_given_candidate,
                neglog_q_candidate_given_current=neglog_proposal_candidate_given_current,
                acceptance_wise_type=self.acceptance_wise_type,
                rng=rng,
                joint=False,
            )

            new_var[idx_pix] = candidate_accepted * 1

            accept_total[idx_pix] = accept_arr * 1
            log_proba_accept_total[idx_pix] = log_proba_accept * 1

            self.internal_update_hyperparameters(
                t=t, v_cand=v_cand, accept_arr=accept_arr, idx_pix=idx_pix
            )

            current[self.key]["var"] = new_var * 1
            for k, target_dist in target_distributions.items():
                current[k] = target_distributions[k].compute_all(
                    current,
                    compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
                )

        return current, accept_total, log_proba_accept_total

    def get_new_sample_optim(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        **kwargs,
    ) -> tuple:
        new_var = current[self.key]["var"] * 1
        self.var_shape = new_var.shape

        self.objective_type = "objective_global"
        self.chromatic_gibbs = False

        candidate, _, mu_current = self.sample(
            current=current[self.key],
            idx_pix=xp.arange(new_var.shape[0]),
            rng=rng,
        )

        current_candidate = {k: {"var": current[k]["var"]} for k in current.keys()}
        current_candidate[self.key]["var"] = mu_current * 1

        target_distributions[self.key].update_nlpdf_utils(
            current_candidate,
            idx_pix=None,
            compute_derivatives=True,
            compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
        )

        candidate_all = {}
        candidate_all[self.key] = {"var": mu_current * 1}
        candidate_all[self.key][self.objective_type] = (
            target_distributions[self.key]
            .neglog_pdf(
                pixelwise=True,
                update_nlpdf_utils=False,
                idx_pix=None,
                chromatic_gibbs=self.chromatic_gibbs,
            )
            .sum()
        )
        candidate_all[self.key]["grad"] = target_distributions[
            self.key
        ].grad_neglog_pdf(update_nlpdf_utils=False, idx_pix=None)

        if (
            candidate_all[self.key][self.objective_type]
            < current[self.key][self.objective_type]
        ):
            current = candidate_all
            accept = True
            proba = 1
        else:
            current_candidate[self.key]["var"] = candidate * 1

            target_distributions[self.key].update_nlpdf_utils(
                current_candidate,
                idx_pix=None,
                compute_derivatives=True,
                compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
            )

            candidate_all = {}
            candidate_all[self.key] = {"var": candidate * 1}
            candidate_all[self.key][self.objective_type] = (
                target_distributions[self.key]
                .neglog_pdf(
                    pixelwise=True,
                    update_nlpdf_utils=False,
                    idx_pix=None,
                    chromatic_gibbs=self.chromatic_gibbs,
                )
                .sum()
            )
            candidate_all[self.key]["grad"] = target_distributions[
                self.key
            ].grad_neglog_pdf(update_nlpdf_utils=False, idx_pix=None)
            if (
                candidate_all[self.key][self.objective_type]
                < current[self.key][self.objective_type]
            ):
                current = candidate_all
                accept = True
                proba = 1
            else:
                accept = False
                proba = 0

        grad_tp1 = candidate_all[self.key]["grad"]

        v_cand = (
            self.alpha * self.v.reshape(self.var_shape) + (1 - self.alpha) * grad_tp1**2
        )

        self.internal_update_hyperparameters(
            t=t,
            v_cand=v_cand,
            accept_arr=accept * xp.ones(self.var_shape[0], dtype=bool),
            idx_pix=xp.arange(self.var_shape[0]),
        )

        assert xp.sum(xp.isnan(self.v)) == 0.0
        assert xp.sum(xp.isinf(self.v)) == 0.0, (
            f"{candidate_all[self.key]}, {candidate_all[self.key]['grad']}"
        )
        assert xp.sum(xp.isnan(current[self.key]["var"])) == 0.0

        # update the current variable for the next site group
        current[self.key]["var"] = (
            current_candidate[self.key]["var"] * 1 if accept else new_var * 1
        )
        current[self.key] = target_distributions[self.key].compute_all(
            current,
            compute_derivatives_2nd_order=self.compute_derivatives_2nd_order,
        )

        return (
            current,
            accept * xp.ones(new_var.shape[0], dtype=bool),
            proba * xp.ones(new_var.shape[0]),
        )
