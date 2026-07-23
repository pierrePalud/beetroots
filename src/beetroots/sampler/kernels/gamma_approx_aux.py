try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from typing import Dict, Optional, Tuple, Union

import scipy

from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.kernels.abstract_kernel import KernelUtils
from beetroots.sampler.kernels.abstract_kernel_ind_mtm import KernelIndependentMTM
from beetroots.sampler.kernels.abstract_kernel_metropolis_hastings import (
    KernelMetropolisHastings,
)
from beetroots.sampler.kernels.multiplicative_noise import (
    KernelMultiplicativeNoiseIMTM,
    KernelMultiplicativeNoiseMH,
)

VALID_MEAN_TYPES_MH = ["forward", "observations", "current"]
VALID_MEAN_TYPES_IMTM = ["forward", "observations"]
VALID_ACCEPTANCE_WISE_TYPES = ["pixelwise", "full", "entire_map"]
VALID_MIXTURE_METHOD = [
    "censored_mult",
    "censored_add",
]


def log_conditional(u, log_ftheta, sigma_m, sigma_a, y):
    r"""Log of the conditional density of the auxiliary variable."""
    mult = -(
        (xp.log(u) - log_ftheta + sigma_m**2 / 2) ** 2 / (2 * sigma_m**2) + xp.log(u)
    )
    add = -((u - y) ** 2) / (2 * sigma_a**2)
    return mult + add


def ratio_first_second_derivative(u, log_ftheta, sigma_m2, sigma_a2, y):
    r"""Ratio of first and second derivatives of the neglog-conditional density."""
    utils_1 = xp.log(u) - log_ftheta + (3 / 2) * sigma_m2
    utils_2 = u * sigma_m2

    num = utils_1 / utils_2 + (u - y) / sigma_a2
    den = -(utils_1 - 1) / (u * utils_2) + 1 / sigma_a2
    return num / den


def second_order_derivative(u, log_ftheta, sigma_m2, sigma_a2):
    r"""Second derivative of the log-conditional density."""
    mult_deriv2 = (xp.log(u) - log_ftheta + (3 / 2) * sigma_m2 - 1) / (sigma_m2 * u**2)
    add_deriv2 = -1 / sigma_a2
    return mult_deriv2 + add_deriv2


def derivative_log_conditional(u, log_ftheta, sigma_m, sigma_a, y):
    r"""First derivative of the log-conditional density."""
    prior_deriv = -(xp.log(u) - log_ftheta + sigma_m**2 / 2) / (sigma_m**2 * u) - 1 / u
    likelihood_deriv = -(u - y) / (sigma_a**2)
    return prior_deriv + likelihood_deriv


def newton_raphson_method(
    initial_guess,
    log_ftheta,
    sigma_m2,
    sigma_a2,
    y,
    max_iter: int = 5,
):
    r"""Newton-Raphson method to refine the conditional mode estimate."""
    mode = initial_guess
    for _ in range(max_iter):
        ratio = ratio_first_second_derivative(mode, log_ftheta, sigma_m2, sigma_a2, y)
        mode_new = xp.clip(
            mode - ratio,
            xp.minimum(y, xp.exp(log_ftheta) * xp.exp(-(3 / 2) * sigma_m2)),
            xp.maximum(y, xp.exp(log_ftheta) * xp.exp(-(3 / 2) * sigma_m2)),
        )
        mode = mode_new
    return mode


def gamma_approximation_parameters(mode, log_ftheta, sigma_m2, sigma_a2):
    r"""Gamma parameters matching local curvature around the mode."""
    alpha = 1 + mode**2 * xp.abs(
        second_order_derivative(mode, log_ftheta, sigma_m2, sigma_a2)
    )
    theta = mode / (alpha - 1)
    return alpha, theta


def estimate_mode_from_local_grid(
    f_theta,
    log_ftheta,
    sigma_m,
    sigma_a,
    sigma_m2,
    y,
    K: int = 10,
):
    r"""Estimate a mode with local grid search and derivative-based interpolation."""
    u_derivative = xp.logspace(
        xp.log10(xp.minimum(f_theta * xp.exp(-3 / 2 * sigma_m2), y)),
        xp.log10(xp.maximum(f_theta * xp.exp(-3 / 2 * sigma_m2), y)),
        K,
    )

    log_cond_values = log_conditional(
        u_derivative,
        log_ftheta,
        sigma_m,
        sigma_a,
        y,
    )

    argmax = xp.argmax(log_cond_values, axis=0)
    left = xp.maximum(argmax - 1, 0)
    right = xp.minimum(argmax + 1, K - 1)

    left_vals = xp.take_along_axis(log_cond_values, left[None, ...], axis=0)[0]
    right_vals = xp.take_along_axis(log_cond_values, right[None, ...], axis=0)[0]

    idx_max = xp.where(
        argmax == K - 1,
        K - 2,
        xp.where(argmax == 0, 0, xp.where(right_vals >= left_vals, argmax, left)),
    )
    # sum_consec = log_cond_values[:-1, ...] + log_cond_values[1:, ...]
    # idx_max = xp.argmax(sum_consec, axis=0)

    flat_u = u_derivative.reshape((K, -1))
    flat_idx = idx_max.reshape(-1)
    cols = xp.arange(flat_idx.size)
    max2_flat = xp.stack(
        [
            flat_u[flat_idx, cols],
            flat_u[flat_idx + 1, cols],
        ],
        axis=0,
    )
    max2 = max2_flat.reshape((2,) + idx_max.shape)

    der = -derivative_log_conditional(
        max2,
        log_ftheta,
        sigma_m,
        sigma_a,
        y,
    )

    mode = ((max2[0] / xp.abs(der[0])) + (max2[1] / xp.abs(der[1]))) / (
        1.0 / xp.abs(der[0]) + 1.0 / xp.abs(der[1])
    )
    return mode


class Approximation(KernelUtils):
    __slots__ = (
        "key",
        "is_stochastic",
        "key_forward_var",
        "lower_bound",
        "var_shape",
        "objective_type",
        "chromatic_gibbs",
        "joint",
        "acceptance_wise_type",
        "summing_axes",
        "distribution_with_forward_name",
        "which_noise",
    )

    def __init__(
        self,
        key: str,
        is_stochastic: bool,
        target_distribution: TargetDistribution,
        joint: bool,
        acceptance_wise_type: str,
        key_forward_var: str,
    ):
        self.key = key
        self.is_stochastic = is_stochastic
        self.joint = joint
        assert acceptance_wise_type in VALID_ACCEPTANCE_WISE_TYPES
        self.acceptance_wise_type = acceptance_wise_type
        self.key_forward_var = key_forward_var

        self.which_noise = None
        self.summing_axes = None
        self.chromatic_gibbs = None
        self.var_shape = None
        self.y = None
        self.distribution_with_forward_name = None

        self.extract_utils(target_distribution=target_distribution)

    def extract_utils(self, target_distribution):
        r"""Extracts the necessary utilities for the proposal distribution."""
        distribution_with_forward_name = None
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
                sigma_m2 = getattr(comp, "sigma_m2", None)
            if hasattr(comp, "sigma_a") and isinstance(
                getattr(comp, "sigma_a", None), xp.ndarray
            ):
                sigma_a = getattr(comp, "sigma_a", None)
                sigma_a2 = getattr(comp, "sigma_a2", None)

        if distribution_with_forward_name is None:
            raise ValueError(
                "forward_map_evals should be defined in the component distribution."
            )
        if y is None:
            raise ValueError("y should be defined in the component distribution.")
        if censored is None:
            raise ValueError(
                "censored should be defined in the component distribution."
            )
        if sigma_m is None:
            raise ValueError("sigma_m should be defined in the component distribution.")
        if sigma_a is None:
            raise ValueError("sigma_a should be defined in the component distribution.")

        self.distribution_with_forward_name = distribution_with_forward_name
        self.y = y
        self.censored = censored
        self.sigma_m = sigma_m
        self.sigma_m2 = sigma_m2
        self.sigma_a = sigma_a
        self.sigma_a2 = sigma_a2


class KernelGammaApproximationCensored(KernelMetropolisHastings, Approximation):
    r"""Dataclass that implement the proposal distribution based on the approximation of the full conditional."""

    __slots__ = ("plot_name",)

    def __init__(
        self,
        key: str,
        is_stochastic: bool,
        target_distribution: TargetDistribution,
        param_interest_name: str,
        scaling_factor_sigma_a: float = 1.0,
        scaling_factor_sigma_m: float = 1.0,
        joint: bool = False,
        acceptance_wise_type: str = "full",
        plot_name: str = "Gamma_approx_censored",
        lower_bound: float = 1e-100,
        upper_bound: Union[float, int] = 1e100,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            key=key,
            is_stochastic=is_stochastic,
            target_distribution=target_distribution,
            joint=joint,
            acceptance_wise_type=acceptance_wise_type,
            key_forward_var=param_interest_name,
        )
        self.plot_name = plot_name

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.kernel_multiplicative_noise = KernelMultiplicativeNoiseMH(
            key=key,
            is_stochastic=is_stochastic,
            target_distribution=target_distribution,
            mean_type="forward",
            key_forward_var=param_interest_name,
            scaling_factor_sigma_m=scaling_factor_sigma_m,
            joint=joint,
            acceptance_wise_type=acceptance_wise_type,
        )

    def sample(
        self,
        mean_candidate_given_current: xp.ndarray,
        alpha: xp.ndarray,
        theta: xp.ndarray,
        rng: xp.random.Generator,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        if idx_pix is None:
            idx_pix = xp.arange(mean_candidate_given_current.shape[0])
        samples = xp.zeros(mean_candidate_given_current.shape)

        idx_pix_mult = xp.zeros_like(self.censored, dtype=bool)
        idx_pix_mult[idx_pix] = True
        idx_pix_mult = idx_pix_mult & self.censored

        samples[self.censored[idx_pix]] = self.kernel_multiplicative_noise.sample(
            mean=mean_candidate_given_current[self.censored[idx_pix]],
            rng=rng,
            idx_pix=idx_pix_mult,
        )

        samples_additive = rng.gamma(shape=alpha, scale=theta)
        samples[~self.censored[idx_pix]] = samples_additive

        return samples

    def neglog_proposal_candidate_given_current(
        self,
        candidate: xp.ndarray,
        log_mean: xp.ndarray,
        alpha: xp.ndarray,
        theta: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        _nlpdf = xp.zeros_like(candidate)

        if idx_pix is None:
            idx_pix = xp.arange(candidate.shape[0])

        idx_pix_mult = xp.zeros_like(self.censored, dtype=bool)
        idx_pix_mult[idx_pix] = True
        idx_pix_mult = idx_pix_mult & self.censored

        _nlpdf_mult = (
            self.kernel_multiplicative_noise.neglog_proposal_candidate_given_current(
                candidate=candidate[self.censored[idx_pix]],
                log_mean=log_mean[self.censored[idx_pix]],
                idx_pix=idx_pix_mult,
            )
        )

        _nlpdf_add = -scipy.stats.gamma.logpdf(
            candidate[~self.censored[idx_pix]], a=alpha, scale=theta
        )

        _nlpdf[self.censored[idx_pix]] += _nlpdf_mult
        _nlpdf[~self.censored[idx_pix]] += _nlpdf_add

        return _nlpdf.sum(axis=self.summing_axes)

    def neglog_proposal_current_given_candidate(
        self,
        current_var: xp.ndarray,
        log_mean: xp.ndarray,
        alpha: xp.ndarray,
        theta: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        if idx_pix is None:
            idx_pix = xp.arange(current_var.shape[0])

        idx_pix_mult = xp.zeros_like(self.censored, dtype=bool)
        idx_pix_mult[idx_pix] = True
        idx_pix_mult = idx_pix_mult & self.censored

        _nlpdf = xp.zeros_like(current_var)

        _nlpdf_mult = (
            self.kernel_multiplicative_noise.neglog_proposal_current_given_candidate(
                current_var=current_var[self.censored[idx_pix]],
                log_mean=log_mean[self.censored[idx_pix]],
                idx_pix=idx_pix_mult,
            )
        )

        _nlpdf_add = -scipy.stats.gamma.logpdf(
            current_var[~self.censored[idx_pix]],
            a=alpha,
            scale=theta,
        )

        _nlpdf[self.censored[idx_pix]] = _nlpdf_mult
        _nlpdf[~self.censored[idx_pix]] = _nlpdf_add

        return _nlpdf.sum(axis=self.summing_axes)

    def get_mean_proposal(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        forward_map_evals_current: Optional[Dict[str, xp.ndarray]] = None,
        idx_pix: Optional[xp.ndarray] = None,
    ):
        var_shape = current[self.key]["var"].shape

        if idx_pix is None:
            idx_pix = xp.arange(var_shape[0])

        idx_pix_uncensored = xp.zeros_like(self.censored, dtype=bool)
        idx_pix_uncensored[idx_pix] = True
        idx_pix_uncensored = idx_pix_uncensored & ~self.censored

        mean_candidate_given_current = xp.zeros(var_shape)
        log_mean_candidate_given_current = xp.zeros(var_shape)
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

        forward_map_evals["f_Var"] = xp.clip(
            forward_map_evals["f_Var"], a_min=self.lower_bound, a_max=self.upper_bound
        )
        mean_candidate_given_current[self.censored[idx_pix]] = xp.maximum(
            forward_map_evals["f_Var"][self.censored[idx_pix]],
            self.lower_bound,
        )
        log_mean_candidate_given_current[self.censored[idx_pix]] = xp.maximum(
            forward_map_evals["log_f_Var"][self.censored[idx_pix]],
            xp.log(self.lower_bound),
        )

        # Gamma (uncensored)
        # METHOD 1
        # u0_approx = (
        #     self.y[idx_pix_uncensored]
        #     * (xp.exp(self.sigma_m2[idx_pix_uncensored]) - 1)
        #     * forward_map_evals["f_Var"][~self.censored[idx_pix]] ** 2
        # )
        # u0_approx += (
        #     forward_map_evals["f_Var"][~self.censored[idx_pix]]
        #     * xp.exp(-(3 / 2 * self.sigma_m2[idx_pix_uncensored]))
        #     * self.sigma_a2[idx_pix_uncensored]
        # )
        # u0_approx /= (
        #     self.sigma_a2[idx_pix_uncensored]
        #     + (xp.exp(self.sigma_m2[idx_pix_uncensored]) - 1)
        #     * forward_map_evals["f_Var"][~self.censored[idx_pix]] ** 2
        # )
        ## METHOD 2
        # mode_lognormal = forward_map_evals["f_Var"][~self.censored[idx_pix]] * xp.exp(
        #     -(3 / 2) * self.sigma_m2[idx_pix_uncensored]
        # )
        # neglogpdf_mode_lognormal = (
        #     (mode_lognormal - self.y[idx_pix_uncensored]) ** 2
        #     / (2 * self.sigma_a2[idx_pix_uncensored])
        #     + (
        #         xp.log(mode_lognormal)
        #         - forward_map_evals["log_f_Var"][~self.censored[idx_pix]]
        #         + self.sigma_m2[idx_pix_uncensored] / 2
        #     )
        #     ** 2
        #     / (2 * self.sigma_m2[idx_pix_uncensored])
        #     + xp.log(mode_lognormal)
        # )
        # neglogpdf_y_normal = (
        #     xp.log(self.y[idx_pix_uncensored])
        #     - forward_map_evals["log_f_Var"][~self.censored[idx_pix]]
        #     + self.sigma_m2[idx_pix_uncensored] / 2
        # ) ** 2 / (2 * self.sigma_m2[idx_pix_uncensored]) + xp.log(
        #     self.y[idx_pix_uncensored]
        # )
        # u0_approx = xp.where(
        #     neglogpdf_mode_lognormal < neglogpdf_y_normal,
        #     mode_lognormal,
        #     self.y[idx_pix_uncensored],
        # )
        # METHOD 3
        K = 10
        u0_approx = estimate_mode_from_local_grid(
            f_theta=forward_map_evals["f_Var"][~self.censored[idx_pix]],
            log_ftheta=forward_map_evals["log_f_Var"][~self.censored[idx_pix]],
            sigma_m=self.sigma_m[idx_pix_uncensored],
            sigma_a=self.sigma_a[idx_pix_uncensored],
            sigma_m2=self.sigma_m2[idx_pix_uncensored],
            y=self.y[idx_pix_uncensored],
            K=K,
        )

        mode_gamma_approx = newton_raphson_method(
            initial_guess=u0_approx,
            log_ftheta=forward_map_evals["log_f_Var"][~self.censored[idx_pix]],
            sigma_m2=self.sigma_m2[idx_pix_uncensored],
            sigma_a2=self.sigma_a2[idx_pix_uncensored],
            y=self.y[idx_pix_uncensored],
            max_iter=5,
        )
        alpha_candidate_given_current, theta_candidate_given_current = (
            gamma_approximation_parameters(
                mode_gamma_approx,
                forward_map_evals["log_f_Var"][~self.censored[idx_pix]],
                self.sigma_m2[idx_pix_uncensored],
                self.sigma_a2[idx_pix_uncensored],
            )
        )

        return (
            mean_candidate_given_current,
            log_mean_candidate_given_current,
            alpha_candidate_given_current,
            theta_candidate_given_current,
        )

    def get_new_sample_sampling(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        current_var = current[self.key]["var"] * 1  # (N, D)
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
            alpha_candidate_given_current,
            theta_candidate_given_current,
        ) = self.get_mean_proposal(
            current=current,
            t=t,
            target_distributions=target_distributions,
        )

        candidate = self.sample(
            mean_candidate_given_current=mean_candidate_given_current,
            alpha=alpha_candidate_given_current,
            theta=theta_candidate_given_current,
            rng=rng,
        )

        # Same proposal parameters in reverse direction since the physical variable is fixed.
        log_mean_current_given_candidate = log_mean_candidate_given_current * 1
        alpha_current_given_candidate = alpha_candidate_given_current * 1
        theta_current_given_candidate = theta_candidate_given_current * 1

        neglog_proposal_candidate_given_current = (
            self.neglog_proposal_candidate_given_current(
                candidate=candidate,
                log_mean=log_mean_candidate_given_current,
                alpha=alpha_candidate_given_current,
                theta=theta_candidate_given_current,
            )
        )

        neglog_proposal_current_given_candidate = (
            self.neglog_proposal_current_given_candidate(
                current_var=current_var,
                log_mean=log_mean_current_given_candidate,
                alpha=alpha_current_given_candidate,
                theta=theta_current_given_candidate,
            )
        )

        current_candidate = {
            k: {
                "var": xp.repeat(
                    current[k]["var"][:, xp.newaxis],
                    2,
                    axis=1,
                )
            }
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
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        raise NotImplementedError(
            "Optimization is not implemented yet for GammaApproximationCensored."
        )


class KernelGammaApproximationCensoredIMTM(KernelIndependentMTM, Approximation):
    r"""Dataclass that implement the proposal distribution based on the approximation of the full conditional."""

    __slots__ = (
        "plot_name",
        "method_mixture",
    )

    def __init__(
        self,
        key: str,
        is_stochastic: bool,
        target_distribution: TargetDistribution,
        k_mtm: int,
        key_forward_var: str,
        scaling_factor_sigma_m: float = 1.0,
        joint: bool = False,
        acceptance_wise_type: str = "full",
        plot_name: str = "Gaussian_noise_MH",
        lower_bound: float = 1e-100,
        upper_bound: Union[float, int] = 1e100,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            key=key,
            is_stochastic=is_stochastic,
            target_distribution=target_distribution,
            joint=joint,
            acceptance_wise_type=acceptance_wise_type,
            key_forward_var=key_forward_var,
        )
        self.k_mtm = k_mtm
        self.censored_imtm = xp.repeat(self.censored[:, None], self.k_mtm + 1, axis=1)
        self.plot_name = plot_name

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.kernel_multiplicative_noise = KernelMultiplicativeNoiseIMTM(
            key=key,
            k_mtm=k_mtm,
            is_stochastic=is_stochastic,
            target_distribution=target_distribution,
            mean_type="forward",
            key_forward_var=key_forward_var,
            scaling_factor_sigma_m=scaling_factor_sigma_m,
            joint=joint,
            acceptance_wise_type=acceptance_wise_type,
        )

    def sample(
        self,
        mean_candidates: xp.ndarray,
        alpha: xp.ndarray,
        theta: xp.ndarray,
        rng: xp.random.Generator,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        samples = xp.zeros(mean_candidates.shape)

        if idx_pix is None:
            idx_pix = xp.arange(mean_candidates.shape[0])

        idx_pix_selection = self.censored_imtm * True
        idx_pix_selection[~xp.isin(xp.arange(idx_pix_selection.shape[0]), idx_pix)] = (
            False
        )

        samples[self.censored_imtm[idx_pix]] = self.kernel_multiplicative_noise.sample(
            mean=mean_candidates[self.censored_imtm[idx_pix]],
            rng=rng,
            idx_pix=idx_pix_selection,
        )

        samples_additive = rng.gamma(
            shape=alpha,
            scale=theta,
        )
        samples[~self.censored_imtm[idx_pix]] = samples_additive
        assert samples.shape == mean_candidates.shape

        return samples

    def neglog_proposal_candidates_current_selection_bias(
        self,
        candidates: xp.ndarray,
        mean: xp.ndarray,
        log_mean: xp.ndarray,
        alpha: xp.ndarray,
        theta: xp.ndarray,
        idx_selection: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
    ):
        if idx_pix is None:
            idx_pix = xp.arange(candidates.shape[0])
        _nlpdf = xp.zeros_like(candidates)

        _nlpdf_add = -scipy.stats.gamma.logpdf(
            candidates[~self.censored_imtm[idx_pix]], a=alpha, scale=theta
        )

        _nlpdf[~self.censored_imtm[idx_pix]] = _nlpdf_add

        return _nlpdf.sum(axis=self.summing_axes)

    def neglog_proposal_candidates_current(
        self,
        candidates: xp.ndarray,
        mean: xp.ndarray,
        log_mean: xp.ndarray,
        weight_lower: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        raise NotImplementedError(
            "neglog_proposal_candidates_current is not implemented yet for KernelGammaApproximationCensoredIMTM."
        )

    def get_means_proposal(
        self,
        current: dict,
        idx_pix: xp.ndarray,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        forward_map_evals_current: Optional[Dict[str, xp.ndarray]] = None,
        compute_lin: bool = True,
        compute_log: bool = True,
    ):
        imtm_param_shape = current[self.key_forward_var]["var"].shape
        imtm_aux_shape = (*imtm_param_shape[:2], *current[self.key]["var"].shape[1:])

        idx_pix_uncensored = xp.zeros_like(self.censored, dtype=bool)
        idx_pix_uncensored[idx_pix] = True
        idx_pix_uncensored = idx_pix_uncensored & ~self.censored

        mean_candidate_given_current = xp.zeros(imtm_aux_shape)
        log_mean_candidate_given_current = xp.zeros(imtm_aux_shape)

        if forward_map_evals_current is None:
            forward_map_evals = (
                target_distributions[self.key]
                .distribution_components[self.distribution_with_forward_name]
                .forward_map.compute_all(
                    current[self.key_forward_var]["var"].reshape(
                        -1, *imtm_param_shape[2:]
                    ),
                    compute_lin=compute_lin,
                    compute_log=compute_log,
                    compute_derivatives=False,
                    compute_derivatives_2nd_order=False,
                )
            )
        else:
            forward_map_evals = forward_map_evals_current

        forward_map_evals_fvar = forward_map_evals["f_Var"].reshape(*imtm_aux_shape)
        forward_map_evals_logfvar = forward_map_evals["log_f_Var"].reshape(
            *imtm_aux_shape
        )

        forward_map_evals_fvar = xp.clip(
            forward_map_evals_fvar, a_min=self.lower_bound, a_max=self.upper_bound
        )
        mean_candidate_given_current = xp.maximum(
            forward_map_evals_fvar,
            self.lower_bound,
        )
        log_mean_candidate_given_current = xp.maximum(
            forward_map_evals_logfvar,
            xp.log(self.lower_bound),
        )

        # forward_map_evals already restricted to idx_pix → shape (N_pix, K, L)
        f_var = forward_map_evals_fvar
        log_f_var = forward_map_evals_logfvar

        u0_approx = estimate_mode_from_local_grid(
            f_theta=f_var,
            log_ftheta=log_f_var,
            sigma_m=self.sigma_m[idx_pix][:, None, :],
            sigma_a=self.sigma_a[idx_pix][:, None, :],
            sigma_m2=self.sigma_m2[idx_pix][:, None, :],
            y=self.y[idx_pix][:, None, :],
            K=10,
        )

        # Expand sigma_m2 to shape (N_pix, k_mtm+1, L) to match u0_approx
        sigma_m2 = xp.repeat(self.sigma_m2[idx_pix][:, None, :], self.k_mtm + 1, axis=1)
        sigma_a2 = xp.repeat(self.sigma_a2[idx_pix][:, None, :], self.k_mtm + 1, axis=1)
        y = xp.repeat(self.y[idx_pix][:, None, :], self.k_mtm + 1, axis=1)

        # --- Newton refinement ---
        mode_gamma_approx = newton_raphson_method(
            initial_guess=u0_approx[~self.censored_imtm[idx_pix]],
            log_ftheta=log_f_var[~self.censored_imtm[idx_pix]],
            sigma_m2=sigma_m2[~self.censored_imtm[idx_pix]],
            sigma_a2=sigma_a2[~self.censored_imtm[idx_pix]],
            y=y[~self.censored_imtm[idx_pix]],
            max_iter=5,
        )

        # --- Gamma approx parameters ---
        alpha_candidate_given_current, theta_candidate_given_current = (
            gamma_approximation_parameters(
                mode_gamma_approx,
                log_f_var[~self.censored_imtm[idx_pix]],
                sigma_m2[~self.censored_imtm[idx_pix]],
                sigma_a2[~self.censored_imtm[idx_pix]],
            )
        )

        return (
            mean_candidate_given_current,
            log_mean_candidate_given_current,
            alpha_candidate_given_current,
            theta_candidate_given_current,
        )

    def get_new_sample_sampling(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        raise NotImplementedError(
            "Sampling is not implemented yet for KernelGammaApproximationCensoredIMTM."
        )

    def get_new_sample_optim(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        raise NotImplementedError(
            "Optimization is not implemented yet for KernelGammaApproximationCensoredIMTM."
        )
