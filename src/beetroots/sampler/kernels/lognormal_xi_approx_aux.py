from typing import Dict, Optional, Tuple

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.kernels.abstract_kernel import KernelUtils
from beetroots.sampler.kernels.abstract_kernel_ind_mtm import KernelIndependentMTM
from beetroots.sampler.kernels.abstract_kernel_metropolis_hastings import (
    KernelMetropolisHastings,
)

VALID_ACCEPTANCE_WISE_TYPES = ["pixelwise", "full", "entire_map"]


class Approximation(KernelUtils):
    __slots__ = (
        "key",
        "is_stochastic",
        "key_forward_var",
        "var_shape",
        "objective_type",
        "chromatic_gibbs",
        "joint",
        "acceptance_wise_type",
        "summing_axes",
        "distribution_with_forward_name",
        "eps_ratio",
        "var_clip_min",
        "var_clip_max",
        "_var_cache",
    )

    def __init__(
        self,
        key: str,
        is_stochastic: bool,
        target_distribution: TargetDistribution,
        joint: bool,
        acceptance_wise_type: str,
        key_forward_var: str,
        eps_ratio: float = 1e-12,
        var_clip_min: float = 1e-8,
        var_clip_max: float = 1e2,
    ):
        self.key = key
        self.is_stochastic = is_stochastic
        self.joint = joint
        assert acceptance_wise_type in VALID_ACCEPTANCE_WISE_TYPES
        self.acceptance_wise_type = acceptance_wise_type
        self.key_forward_var = key_forward_var

        self.summing_axes = None
        self.chromatic_gibbs = None
        self.var_shape = None
        self.distribution_with_forward_name = None

        self.eps_ratio = float(eps_ratio)
        self.var_clip_min = float(var_clip_min)
        self.var_clip_max = float(var_clip_max)
        self._std_cache = None

        self.extract_utils(target_distribution=target_distribution)

    def extract_utils(self, target_distribution):
        distribution_with_forward_name = None
        y = None
        censored = None
        sigma_m2 = None
        sigma_a = None
        sigma_a2 = None

        for comp_name, comp in target_distribution.distribution_components.items():
            if hasattr(comp, "forward_map_evals") and isinstance(
                comp.forward_map_evals, dict
            ):
                distribution_with_forward_name = comp_name
            if hasattr(comp, "y") and isinstance(getattr(comp, "y", None), xp.ndarray):
                y = getattr(comp, "y", None)
            if hasattr(comp, "censored") and isinstance(
                getattr(comp, "censored", None), xp.ndarray
            ):
                censored = getattr(comp, "censored", None)
            if hasattr(comp, "sigma_m") and isinstance(
                getattr(comp, "sigma_m", None), xp.ndarray
            ):
                sigma_m = getattr(comp, "sigma_m", None)
            if hasattr(comp, "sigma_m2") and isinstance(
                getattr(comp, "sigma_m2", None), xp.ndarray
            ):
                sigma_m2 = getattr(comp, "sigma_m2", None)
            if hasattr(comp, "sigma_a") and isinstance(
                getattr(comp, "sigma_a", None), xp.ndarray
            ):
                sigma_a = getattr(comp, "sigma_a", None)
            if hasattr(comp, "sigma_a2") and isinstance(
                getattr(comp, "sigma_a2", None), xp.ndarray
            ):
                sigma_a2 = getattr(comp, "sigma_a2", None)

        if distribution_with_forward_name is None:
            raise ValueError(
                "A component with forward_map_evals is required for xi approximation."
            )
        if y is None or censored is None or sigma_a is None or sigma_a2 is None:
            raise ValueError(
                "Missing observation likelihood metadata for xi approximation."
            )
        if sigma_m2 is None or sigma_m is None:
            raise ValueError(
                "sigma_m2 and sigma_m metadata are required for censored fallback variance."
            )

        self.distribution_with_forward_name = distribution_with_forward_name
        self.y = y
        self.censored = censored
        self.sigma_a = sigma_a
        self.sigma_a2 = sigma_a2
        self.sigma_m2 = sigma_m2
        self.sigma_m = sigma_m

    def _neglog_normal(
        self, x: xp.ndarray, mean: xp.ndarray, std: xp.ndarray
    ) -> xp.ndarray:
        return (
            xp.log(2.0 * xp.pi)
            + xp.log(std)
            + 0.5 * xp.square(x - mean) / xp.square(std)
        )


class KernelLogNormalXiApproximationCensored(KernelMetropolisHastings, Approximation):
    __slots__ = ("plot_name",)

    def __init__(
        self,
        key: str,
        is_stochastic: bool,
        target_distribution: TargetDistribution,
        key_forward_var: str,
        joint: bool = False,
        acceptance_wise_type: str = "full",
        plot_name: str = "LogNormal_xi_approx_censored",
        eps_ratio: float = 1e-12,
        var_clip_min: float = 1e-8,
        var_clip_max: float = 1e2,
        **kwargs,
    ) -> None:
        super().__init__(
            key=key,
            is_stochastic=is_stochastic,
            target_distribution=target_distribution,
            joint=joint,
            acceptance_wise_type=acceptance_wise_type,
            key_forward_var=key_forward_var,
            eps_ratio=eps_ratio,
            var_clip_min=var_clip_min,
            var_clip_max=var_clip_max,
        )
        self.plot_name = plot_name

    def sample(
        self,
        mean_candidate_given_current: xp.ndarray,
        var_candidate_given_current: xp.ndarray,
        rng: xp.random.Generator,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> Tuple[xp.ndarray, xp.ndarray]:
        std = xp.sqrt(var_candidate_given_current)
        samples = mean_candidate_given_current + std * rng.normal(
            loc=0.0,
            scale=1.0,
            size=mean_candidate_given_current.shape,
        )
        weight_lower = xp.zeros_like(samples)
        return samples, weight_lower

    def neglog_proposal_candidate_given_current(
        self,
        candidate: xp.ndarray,
        mean: xp.ndarray,
        log_mean: xp.ndarray,
        var: xp.ndarray,
        weight_lower: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        del log_mean, weight_lower
        std = xp.sqrt(xp.clip(var, self.var_clip_min, self.var_clip_max))

        if idx_pix is not None and mean.shape[0] != candidate.shape[0]:
            mean = mean[idx_pix]
            std = std[idx_pix]

        return self._neglog_normal(candidate, mean, std).sum(axis=self.summing_axes)

    def neglog_proposal_current_given_candidate(
        self,
        current_var: xp.ndarray,
        mean: xp.ndarray,
        log_mean: xp.ndarray,
        var: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        del log_mean
        std = xp.sqrt(xp.clip(var, self.var_clip_min, self.var_clip_max))

        if idx_pix is not None and mean.shape[0] != current_var.shape[0]:
            mean = mean[idx_pix]
            std = std[idx_pix]

        return self._neglog_normal(current_var, mean, std).sum(axis=self.summing_axes)

    def get_mean_proposal(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        forward_map_evals_current: Optional[Dict[str, xp.ndarray]] = None,
        idx_pix: Optional[xp.ndarray] = None,
    ):
        del t
        var_shape = current[self.key]["var"].shape

        if idx_pix is None:
            idx_pix = xp.arange(var_shape[0])

        if forward_map_evals_current is None:
            forward_map_evals = (
                target_distributions[self.key]
                .distribution_components[self.distribution_with_forward_name]
                .forward_map.compute_all(
                    current[self.key_forward_var]["var"],
                    compute_lin=True,
                    compute_log=True,
                    compute_derivatives=False,
                    compute_derivatives_2nd_order=False,
                )
            )
        else:
            forward_map_evals = forward_map_evals_current

        f = forward_map_evals["f_Var"]
        log_f = forward_map_evals["log_f_Var"]
        # if f.shape[0] == var_shape[0]:
        #     f = f[idx_pix]
        #     log_f = log_f[idx_pix]

        y_loc = self.y[idx_pix]
        sigma_a_loc = self.sigma_a[idx_pix]
        sigma_a2_loc = self.sigma_a2[idx_pix]
        sigma_m_loc = self.sigma_m[idx_pix]
        sigma_m2_loc = self.sigma_m2[idx_pix]
        censored_loc = self.censored[idx_pix]

        # std = xp.where(
        #     sigma_a2_loc**0.5 < f * (xp.exp(sigma_m2_loc) - 1) ** 0.5,
        #     sigma_a_loc / f,
        #     sigma_m_loc,
        # )
        std = sigma_a_loc / f

        mean_idx = xp.log(y_loc) - log_f - (std**2) / 2
        var_idx = std**2

        mean_idx = xp.where(censored_loc, -sigma_m2_loc / 2, mean_idx)
        var_idx = xp.where(censored_loc, sigma_m2_loc, var_idx)

        return (
            mean_idx,
            mean_idx,
            var_idx,
        )

    def get_new_sample_sampling(
        self, current, t, target_distributions, rng, compute_derivatives_2nd_order
    ):
        current_var = current[self.key]["var"] * 1
        self.var_shape = current_var.shape

        if self.acceptance_wise_type == "pixelwise":
            summing_axes = xp.arange(1, current_var.ndim)
        elif self.acceptance_wise_type == "full":
            summing_axes = xp.array([])
        elif self.acceptance_wise_type == "entire_map":
            summing_axes = xp.arange(current_var.ndim)
        self.summing_axes = tuple(summing_axes)

        self.chromatic_gibbs = False
        self.objective_type = "objective_pix_global"

        (
            mean_candidate_given_current,
            log_mean_candidate_given_current,
            var_candidate_given_current,
        ) = self.get_mean_proposal(
            current=current,
            t=t,
            target_distributions=target_distributions,
        )

        candidate, weight_lower = self.sample(
            mean_candidate_given_current=mean_candidate_given_current,
            var_candidate_given_current=var_candidate_given_current,
            rng=rng,
        )

        current_for_reverse = {
            k: {"var": current[k]["var"] * 1} for k in current.keys()
        }
        current_for_reverse[self.key]["var"] = candidate * 1

        (
            mean_current_given_candidate,
            log_mean_current_given_candidate,
            var_current_given_candidate,
        ) = self.get_mean_proposal(
            current=current_for_reverse,
            t=t,
            target_distributions=target_distributions,
        )

        neglog_proposal_candidate_given_current = (
            self.neglog_proposal_candidate_given_current(
                candidate=candidate,
                mean=mean_candidate_given_current,
                log_mean=log_mean_candidate_given_current,
                var=var_candidate_given_current,
                weight_lower=weight_lower,
            )
        )

        neglog_proposal_current_given_candidate = (
            self.neglog_proposal_current_given_candidate(
                current_var=current_var,
                mean=mean_current_given_candidate,
                log_mean=log_mean_current_given_candidate,
                var=var_current_given_candidate,
            )
        )

        current_candidate = {
            k: {"var": xp.repeat(current[k]["var"][:, xp.newaxis], 2, axis=1)}
            for k in current.keys()
        }
        current_candidate[self.key]["var"][:, 0] = candidate

        target_distributions[self.key].update_nlpdf_utils(
            current_candidate,
            idx_pix=None,
            compute_derivatives=False,
            compute_derivatives_2nd_order=False,
            mtm=True,
        )

        neglogpdf_candidate_and_current = target_distributions[self.key].neglog_pdf(
            pixelwise=False,
            full=True,
            update_nlpdf_utils=False,
            idx_pix=None,
            chromatic_gibbs=self.chromatic_gibbs,
        )
        neglogpdf_candidate = neglogpdf_candidate_and_current[:, 0].sum(
            axis=self.summing_axes
        )
        neglogpdf_current = neglogpdf_candidate_and_current[:, 1].sum(
            axis=self.summing_axes
        )

        new_var, accept_total, log_proba_accept_total = self.accept_reject(
            candidates=candidate,
            current_var=current_var,
            neglogpdf_candidates=neglogpdf_candidate,
            neglogpdf_current=neglogpdf_current,
            neglog_q_current_given_candidate=neglog_proposal_current_given_candidate,
            neglog_q_candidate_given_current=neglog_proposal_candidate_given_current,
            acceptance_wise_type=self.acceptance_wise_type,
            rng=rng,
            joint=self.joint,
        )

        if accept_total.max() > 0:
            current[self.key]["var"] = new_var * 1
            for k in target_distributions.keys():
                current[k] = target_distributions[k].compute_all(
                    current,
                    compute_derivatives_2nd_order=compute_derivatives_2nd_order,
                )

        if hasattr(accept_total, "ndim") and accept_total.ndim > 1:
            axes = tuple(range(1, accept_total.ndim))
            accept_out = accept_total.mean(axis=axes)
            log_proba_out = log_proba_accept_total.mean(axis=axes)
        else:
            accept_out = accept_total
            log_proba_out = log_proba_accept_total

        return current, accept_out, log_proba_out

    def get_new_sample_optim(
        self, current, t, target_distributions, rng, compute_derivatives_2nd_order
    ):
        raise NotImplementedError(
            "This kernel does not implement get_new_sample_optim yet."
        )


class KernelLogNormalXiApproximationCensoredIMTM(KernelIndependentMTM, Approximation):
    __slots__ = ("plot_name",)

    def __init__(
        self,
        key: str,
        is_stochastic: bool,
        target_distribution: TargetDistribution,
        k_mtm: int,
        key_forward_var: str,
        joint: bool = False,
        acceptance_wise_type: str = "full",
        plot_name: str = "LogNormal_xi_approx_imtm_censored",
        eps_ratio: float = 1e-12,
        **kwargs,
    ) -> None:
        super().__init__(
            key=key,
            is_stochastic=is_stochastic,
            target_distribution=target_distribution,
            joint=joint,
            acceptance_wise_type=acceptance_wise_type,
            key_forward_var=key_forward_var,
            eps_ratio=eps_ratio,
        )
        self.k_mtm = k_mtm
        self.plot_name = plot_name
        self.censored_imtm = xp.repeat(self.censored[:, None], self.k_mtm + 1, axis=1)

    def sample(
        self,
        mean_candidates: xp.ndarray,
        std_candidates: xp.ndarray,
        rng: xp.random.Generator,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> Tuple[xp.ndarray, xp.ndarray]:
        del idx_pix
        samples = mean_candidates + std_candidates * rng.normal(
            loc=0.0,
            scale=1.0,
            size=mean_candidates.shape,
        )
        weight_lower = xp.zeros_like(samples)
        return samples, weight_lower

    def neglog_proposal_candidates_current_selection_bias(
        self,
        candidates: xp.ndarray,
        mean: xp.ndarray,
        log_mean: xp.ndarray,
        weight_lower: xp.ndarray,
        idx_selection: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
    ):
        del log_mean, weight_lower, idx_selection, idx_pix
        std = self._std_cache
        return self._neglog_normal(candidates, mean, std).sum(axis=self.summing_axes)

    def neglog_proposal_candidates_current(
        self,
        candidates: xp.ndarray,
        mean: xp.ndarray,
        log_mean: xp.ndarray,
        weight_lower: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        del log_mean, weight_lower, idx_pix
        std = self._std_cache
        return self._neglog_normal(candidates, mean, std).sum(axis=self.summing_axes)

    def get_means_and_candidate(
        self,
        current: dict,
        idx_pix: xp.ndarray,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_lin: bool = True,
        compute_log: bool = True,
    ):
        del t
        imtm_param_shape = current[self.key_forward_var]["var"].shape
        imtm_aux_shape = (*imtm_param_shape[:2], *current[self.key]["var"].shape[1:])

        forward_map_evals = (
            target_distributions[self.key]
            .distribution_components[self.distribution_with_forward_name]
            .forward_map.compute_all(
                current[self.key_forward_var]["var"].reshape(-1, *imtm_param_shape[2:]),
                compute_lin=compute_lin,
                compute_log=True,
                compute_derivatives=False,
                compute_derivatives_2nd_order=False,
            )
        )
        f = forward_map_evals["f_Var"].reshape(*imtm_aux_shape)
        log_f = forward_map_evals["log_f_Var"].reshape(*imtm_aux_shape)

        sigma_a2_loc = self.sigma_a2[idx_pix, None]
        sigma_a_loc = self.sigma_a[idx_pix, None]
        sigma_m_loc = self.sigma_m[idx_pix, None]
        sigma_m2_loc = self.sigma_m2[idx_pix, None]
        censored_loc = self.censored[idx_pix, None]
        y_loc = self.y[idx_pix, None]

        # sigma = xp.where(
        #     sigma_a2_loc**0.5 < f * (xp.exp(sigma_m2_loc) - 1) ** 0.5,
        #     (sigma_a_loc / f),  # No log since close to 1
        #     sigma_m_loc,
        # )
        sigma = sigma_a_loc / f

        mean_candidates = (
            xp.log(y_loc) - log_f - (sigma**2) / 2
        )  # TODO: also add xp.where here ?
        std_candidates = sigma

        # Broadcast censor fallback along the MTM axis without allocating repeated arrays.
        xp.copyto(mean_candidates, -sigma_m2_loc / 2, where=censored_loc)
        xp.copyto(std_candidates, sigma_m_loc, where=censored_loc)

        self._std_cache = std_candidates

        candidates, weight_lower = self.sample(
            mean_candidates=mean_candidates,
            std_candidates=std_candidates,
            rng=rng,
            idx_pix=idx_pix,
        )

        return (
            candidates,
            mean_candidates,
            mean_candidates,
            weight_lower,
        )

    def get_new_sample_sampling(
        self, current, t, target_distributions, rng, compute_derivatives_2nd_order
    ):
        raise NotImplementedError(
            "This kernel does not implement get_new_sample_sampling yet."
        )

    def get_new_sample_optim(
        self, current, t, target_distributions, rng, compute_derivatives_2nd_order
    ):
        raise NotImplementedError(
            "This kernel does not implement get_new_sample_optim yet."
        )
