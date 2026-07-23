try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

import abc
from typing import Dict, Optional, Tuple

from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.kernels.abstract_kernel import KernelUtils
from beetroots.sampler.kernels.abstract_kernel_ind_mtm import KernelIndependentMTM
from beetroots.sampler.kernels.abstract_kernel_metropolis_hastings import (
    KernelMetropolisHastings,
)

VALID_MEAN_TYPES_MH = ["forward", "observations", "current"]
VALID_MEAN_TYPES_IMTM = ["forward", "observations"]
VALID_ACCEPTANCE_WISE_TYPES = ["pixelwise", "full", "entire_map"]


class MultiplicativeNoise(KernelUtils):
    __slots__ = (
        "key",
        "sigma_m",
        "scaling_factor_sigma_m",
        "var_shape",
        "objective_type",
        "chromatic_gibbs",
        "joint",
        "sigma_m2",
        "acceptance_wise_type",
        "summing_axes",
        "distribution_with_forward_name",
        "mean_type",
        "key_forward_var",
    )

    def __init__(
        self,
        key: str,
        target_distribution: TargetDistribution,
        scaling_factor_sigma_m: float,
        joint: bool,
        acceptance_wise_type: str,
        mean_type: str = "current",
        key_forward_var: str = "",
        summing_axes: Tuple = (),
    ):
        self.key = key
        self.scaling_factor_sigma_m = scaling_factor_sigma_m
        self.joint = joint
        assert acceptance_wise_type in VALID_ACCEPTANCE_WISE_TYPES
        self.acceptance_wise_type = acceptance_wise_type

        self.distribution_with_forward_name, self.sigma_m = self.extract_utils(
            target_distribution=target_distribution
        )
        self.sigma_m2 = self.sigma_m**2
        self.mean_type = mean_type
        if self.mean_type and not key_forward_var:
            raise ValueError(
                "key_forward_var should be defined if using forward model as mean for the proposal."
            )
        self.key_forward_var = key_forward_var

        self.summing_axes = summing_axes
        self.chromatic_gibbs = None
        self.var_shape = None

    def extract_utils(self, target_distribution):
        r"""Extracts the necessary utilities for the proposal distribution."""
        distribution_with_forward_name, sigma_m = None, None
        for comp_name, comp in target_distribution.distribution_components.items():
            if hasattr(comp, "forward_map_evals") and isinstance(
                comp.forward_map_evals, dict
            ):
                distribution_with_forward_name = comp_name
            if hasattr(comp, "sigma_m") and comp.sigma_m is not None:
                sigma_m = comp.sigma_m

        if distribution_with_forward_name is None:
            raise ValueError(
                "forward_map_evals should be defined in the component distribution."
            )
        if sigma_m is None:
            raise ValueError("sigma_m should be defined in the component distribution.")

        return distribution_with_forward_name, sigma_m

    @abc.abstractmethod
    def sample(
        self,
        mean=xp.ndarray,
        rng=xp.random.Generator,
    ):
        pass

    def get_means_and_candidate_mh(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
    ):
        if self.mean_type == "forward":
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
            mean_candidate_given_current = forward_map_evals["f_Var"] * 1
            mean_current_given_candidate = forward_map_evals["f_Var"] * 1
            log_mean_candidate_given_current = forward_map_evals["log_f_Var"] * 1
            log_mean_current_given_candidate = forward_map_evals["log_f_Var"] * 1

            candidate = self.sample(
                mean=mean_candidate_given_current,
                rng=rng,
            )
        elif self.mean_type == "current":
            mean_candidate_given_current = current[self.key]["var"]
            log_mean_candidate_given_current = xp.log(mean_candidate_given_current)

            candidate = self.sample(
                mean=mean_candidate_given_current,
                rng=rng,
            )

            mean_current_given_candidate = candidate * 1
            log_mean_current_given_candidate = xp.log(candidate)
        elif self.mean_type == "observations":
            mean_candidate_given_current = self.y * 1
            mean_current_given_candidate = self.y * 1
            log_mean_candidate_given_current = xp.log(mean_candidate_given_current)
            log_mean_current_given_candidate = xp.log(candidate)

            candidate = self.sample(
                mean=mean_candidate_given_current,
                rng=rng,
            )
        else:
            raise ValueError(f"mean_type should be one of {VALID_MEAN_TYPES_MH}.")

        return (
            candidate,
            mean_candidate_given_current,
            log_mean_candidate_given_current,
            mean_current_given_candidate,
            log_mean_current_given_candidate,
        )

    def get_mean_and_candidates_imtm(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
    ):
        if self.mean_type == "forward":
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
            mean = forward_map_evals["f_Var"]
            log_mean = forward_map_evals["log_f_Var"]

            candidates = self.sample(
                mean=mean,
                rng=rng,
            )
        elif self.mean_type == "observations":
            mean_candidate_given_current = current[self.key]["var"]
            log_mean = xp.log(mean_candidate_given_current)

            candidates = self.sample(
                mean=mean_candidate_given_current,
                rng=rng,
            )
        else:
            raise ValueError(f"mean_type should be one of {VALID_MEAN_TYPES_IMTM}.")

        return (
            candidates,
            mean,
            log_mean,
        )


class KernelMultiplicativeNoiseMH(KernelMetropolisHastings, MultiplicativeNoise):
    r"""Dataclass that implement the proposal distribution based on the multiplicative noise prior (linear scale)"""

    __slots__ = "plot_name"

    def __init__(
        self,
        key: str,
        target_distribution: TargetDistribution,
        mean_type: str = "current",
        key_forward_var: str = "",
        scaling_factor_sigma_m: float = 1.0,
        joint: bool = False,
        acceptance_wise_type: str = "pixelwise",
        plot_name: str = "Mult_noise_MH",
        **kwargs: dict,
    ) -> None:
        super().__init__(
            key=key,
            target_distribution=target_distribution,
            scaling_factor_sigma_m=scaling_factor_sigma_m,
            joint=joint,
            acceptance_wise_type=acceptance_wise_type,
            mean_type=mean_type,
            key_forward_var=key_forward_var,
        )
        self.plot_name = plot_name

    def sample(
        self,
        mean: xp.ndarray,
        rng: xp.random.Generator,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        r"""draws a random vectors from a Lognormal distribution.

        Parameters
        ----------
        mean : xp.ndarray
            current iterate, mean of the Gaussian proposal
        rng : xp.random.Generator
            random number generator

        Returns
        -------
        xp.array of shape (n_pix, self.k_mtm, D)
        """
        if idx_pix is None:
            idx_pix = xp.arange(mean.shape[0])
        samples = mean * rng.lognormal(
            mean=-(self.sigma_m2[idx_pix] * self.scaling_factor_sigma_m**2) / 2,
            sigma=(self.sigma_m[idx_pix] * self.scaling_factor_sigma_m),
        )

        return samples

    def neglog_proposal_candidate_given_current(
        self,
        candidate: xp.ndarray,
        log_mean: xp.ndarray,
        return_normalization: Optional[bool] = False,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        r"""Computes the neglog-pdf of the proposal distribution.

        Parameters
        ----------
        candidate : xp.ndarray
            candidate variable
        log_mean : xp.ndarray
            log of the mean of the Gaussian proposal

        Returns
        -------
        xp.ndarray
            neglog-pdf of the proposal distribution
        """
        if idx_pix is None:
            idx_pix = xp.arange(candidate.shape[0])

        _pdf = (
            xp.log(candidate)
            - log_mean
            + (self.sigma_m2[idx_pix] * self.scaling_factor_sigma_m**2) / 2
        ) ** 2
        _pdf /= 2 * (self.scaling_factor_sigma_m**2 * self.sigma_m2[idx_pix])
        _pdf += xp.log(candidate)
        _pdf = _pdf.sum(axis=self.summing_axes)

        if return_normalization:
            assert len(self.summing_axes) == 0
            normalization_constant = self.scaling_factor_sigma_m * self.sigma_m[idx_pix]
            return _pdf, normalization_constant
        else:
            return _pdf

    def neglog_proposal_current_given_candidate(
        self,
        current_var: xp.ndarray,
        log_mean: xp.ndarray,
        return_normalization: bool = False,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        r"""Computes the neglog-pdf of the proposal distribution.

        Parameters
        ----------
        candidate : xp.ndarray
            candidate variable
        log_mean : xp.ndarray
            log of the mean of the Gaussian proposal

        Returns
        -------
        xp.ndarray
            neglog-pdf of the proposal distribution
        """
        if idx_pix is None:
            idx_pix = xp.arange(current_var.shape[0])
        return self.neglog_proposal_candidate_given_current(
            candidate=current_var,
            log_mean=log_mean,
            return_normalization=return_normalization,
            idx_pix=idx_pix,
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
            accept_total = xp.zeros((self.var_shape[0],))
            log_proba_accept_total = xp.zeros((self.var_shape[0],))
        elif self.acceptance_wise_type == "full":
            summing_axes = xp.array([])
            accept_total = xp.zeros(self.var_shape)
            log_proba_accept_total = xp.zeros(self.var_shape)
        elif self.acceptance_wise_type == "entire_map":
            summing_axes = xp.arange(current_var.ndim)
            accept_total = xp.zeros((1,))
            log_proba_accept_total = xp.zeros((1,))
        self.summing_axes = tuple(summing_axes)

        self.chromatic_gibbs = False
        self.objective_type = "objective_pix_global"

        (
            candidate,
            _,
            log_mean_candidate_given_current,
            _,
            log_mean_current_given_candidate,
        ) = self.get_means_and_candidate_mh(
            current=current,
            t=t,
            target_distributions=target_distributions,
            rng=rng,
        )

        neglog_proposal_candidate_given_current = (
            self.neglog_proposal_candidate_given_current(
                candidate=candidate,
                log_mean=log_mean_candidate_given_current,
            )
        )

        neglog_proposal_current_given_candidate = (
            self.neglog_proposal_current_given_candidate(
                current_var=current_var,
                log_mean=log_mean_current_given_candidate,
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
        }  # Add a dimension for the candidates.
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

        if accept_total.max() > 0:  # if at least one accept and not the last site group
            # update the current variable for the next site group
            current[self.key]["var"] = new_var * 1
            current[self.key] = target_distributions[self.key].compute_all(
                current,
                compute_derivatives_2nd_order=compute_derivatives_2nd_order,
            )

        return current, accept_total, log_proba_accept_total

    def get_new_sample_optim(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        raise NotImplementedError(
            "Optimization is not implemented for KernelLogMultiplicativeNoiseMH"
        )


class KernelMultiplicativeNoiseIMTM(KernelIndependentMTM, MultiplicativeNoise):
    r"""Dataclass that implement the proposal distribution based on multiplicative noise (linear scale)"""

    __slots__ = ("k_mtm", "plot_name")

    def __init__(
        self,
        key: str,
        target_distribution: TargetDistribution,
        k_mtm: int,
        key_forward_var: str,
        mean_type: str,
        scaling_factor_sigma_m: float = 1.0,
        joint: bool = False,
        acceptance_wise_type: str = "pixelwise",
        plot_name: str = "Mult_noise_IMTM",
        **kwargs: dict,
    ) -> None:
        super().__init__(
            key=key,
            target_distribution=target_distribution,
            scaling_factor_sigma_m=scaling_factor_sigma_m,
            joint=joint,
            acceptance_wise_type=acceptance_wise_type,
            key_forward_var=key_forward_var,
            mean_type=mean_type,
        )
        self.k_mtm = k_mtm
        self.plot_name = plot_name

    def sample(
        self,
        mean: xp.ndarray,
        rng: xp.random.Generator,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        r"""draws a random vectors from a Lognormal distribution.

        Parameters
        ----------
        mean : xp.ndarray
            current iterate, mean of the Gaussian proposal
        rng : xp.random.Generator
            random number generator

        Returns
        -------
        xp.array of shape (n_pix, self.k_mtm, D)
        """
        if idx_pix is None:
            idx_pix = xp.arange(mean.shape[0])
        sigma_m = self.sigma_m[:, None].repeat(self.k_mtm + 1, axis=1)
        sigma_m2 = self.sigma_m2[:, None].repeat(self.k_mtm + 1, axis=1)

        samples = mean * rng.lognormal(
            mean=-(sigma_m2[idx_pix] * self.scaling_factor_sigma_m**2) / 2,
            sigma=sigma_m[idx_pix] * self.scaling_factor_sigma_m,
        )

        return samples

    def neglog_proposal_candidates_current(
        self,
        candidates: xp.ndarray,
        log_mean: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        if idx_pix is None:
            idx_pix = xp.arange(candidates.shape[0])
        _pdf = (
            xp.log(candidates)
            - log_mean
            + (self.scaling_factor_sigma_m**2 * self.sigma_m2[idx_pix, None]) / 2
        ) ** 2
        _pdf /= 2 * (self.scaling_factor_sigma_m**2 * self.sigma_m2[idx_pix, None])
        _pdf += xp.log(candidates)
        _pdf = _pdf.sum(axis=self.summing_axes)

        return _pdf

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
            accept_total = xp.zeros((self.var_shape[0],))
            log_rg_total = xp.zeros((self.var_shape[0],))
        elif self.acceptance_wise_type == "full":
            summing_axes = xp.array([])
            accept_total = xp.zeros(self.var_shape)
            log_rg_total = xp.zeros(self.var_shape)
        elif self.acceptance_wise_type == "entire_map":
            summing_axes = xp.arange(current_var.ndim)
            accept_total = xp.zeros((1,))
            log_rg_total = xp.zeros((1,))

        summing_axes[1:] += 1
        self.summing_axes = tuple(summing_axes)

        self.chromatic_gibbs = False
        self.objective_type = "objective_pix_global"

        (
            candidates,
            _,
            log_mean,
        ) = self.get_means_and_candidate_mh(
            current=current,
            t=t,
            target_distributions=target_distributions,
            rng=rng,
        )

        # * Evaluate the neglog-pdf of the proposal distribution
        current_candidates = {
            k: {
                "var": xp.repeat(
                    current[k]["var"][:, xp.newaxis],
                    self.k_mtm + 1,
                    axis=1,
                )
            }
            for k in current.keys()
        }  # Add a dimension for the candidates.
        current_candidates[self.key]["var"][:, :-1] = candidates

        neglog_proposal_candidates_current = self.neglog_proposal_candidates_current(
            candidates=current_candidates[self.key]["var"],
            log_mean=log_mean,
        )
        assert len(neglog_proposal_candidates_current.shape) == len(
            self.var_shape
        ) + 1 - len(self.summing_axes)

        # * Evaluate the neglog-pdf of the target distribution
        target_distributions[self.key].update_nlpdf_utils(
            current_candidates,
            idx_pix=None,
            compute_derivatives=False,
            compute_derivatives_2nd_order=False,
            mtm=True,
        )

        neglogpdf_candidates_current = (
            target_distributions[self.key]
            .neglog_pdf(
                full=True,
                update_nlpdf_utils=False,
                idx_pix=None,
                chromatic_gibbs=self.chromatic_gibbs,
            )
            .sum(axis=self.summing_axes)
        )
        assert len(neglogpdf_candidates_current.shape) == len(self.var_shape) + 1 - len(
            self.summing_axes
        )
        assert xp.isnan(neglogpdf_candidates_current).sum() == 0

        # * Accept-reject step
        new_var, accept_total, log_rg_total = self.accept_reject(
            current_candidates[self.key]["var"][:, :-1],
            current_var,
            neglogpdf_candidates_current,
            neglog_proposal_candidates_current,
            self.acceptance_wise_type,
            rng,
            joint=False,
        )

        if accept_total.max() > 0:  # if at least one accept and not the last site group
            # update the current variable for the next site group
            current[self.key]["var"] = new_var * 1
            current[self.key] = target_distributions[self.key].compute_all(
                current,
                compute_derivatives_2nd_order=compute_derivatives_2nd_order,
            )

        return current, accept_total, log_rg_total

    def get_new_sample_optim(
        self, current, t, target_distributions, rng, compute_derivatives_2nd_order
    ):
        raise NotImplementedError(
            "Optimization is not implemented for KernelLogMultiplicativeNoiseIMTM"
        )


class KernelLogMultiplicativeNoiseMH(KernelMetropolisHastings, MultiplicativeNoise):
    r"""Dataclass that implement the proposal distribution based on the multiplicative noise prior (log scale)"""

    __slots__ = "plot_name"

    def __init__(
        self,
        key: str,
        target_distribution: TargetDistribution,
        mean_type: str,
        key_forward_var: str = "",
        scaling_factor_sigma_m: float = 1.0,
        joint: bool = False,
        acceptance_wise_type: str = "pixelwise",
        plot_name: str = "LogMult_noise_MH",
        **kwargs: dict,
    ) -> None:
        super().__init__(
            key=key,
            target_distribution=target_distribution,
            scaling_factor_sigma_m=scaling_factor_sigma_m,
            joint=joint,
            acceptance_wise_type=acceptance_wise_type,
            mean_type=mean_type,
            key_forward_var=key_forward_var,
        )
        self.plot_name = plot_name

    def sample(
        self,
        log_mean: xp.ndarray,
        rng: xp.random.Generator,
    ) -> xp.ndarray:
        r"""draws a random vectors from a Lognormal distribution.

        Parameters
        ----------
        mean : xp.ndarray
            current iterate, mean of the Gaussian proposal
        rng : xp.random.Generator
            random number generator

        Returns
        -------
        xp.array of shape (n_pix, self.k_mtm, D)
        """
        samples = log_mean[:, None] + rng.normal(
            mean=-(self.sigma_m2 * self.scaling_factor_sigma_m**2) / 2,
            sigma=(self.sigma_m * self.scaling_factor_sigma_m),
        )

        return samples

    def neglog_proposal_candidate_given_current(
        self,
        candidate: xp.ndarray,
        log_mean: xp.ndarray,
        return_normalization: bool = False,
    ) -> xp.ndarray:
        _pdf = (
            candidate - log_mean + (self.sigma_m2 * self.scaling_factor_sigma_m**2) / 2
        ) ** 2
        _pdf /= 2 * (self.scaling_factor_sigma_m**2 * self.sigma_m2)
        _pdf = _pdf.sum(axis=self.summing_axes)

        if return_normalization:
            assert len(self.summing_axes) == 0
            normalization_constant = xp.log(self.scaling_factor_sigma_m * self.sigma_m)
            return _pdf, normalization_constant
        else:
            return _pdf

    def neglog_proposal_current_given_candidate(
        self,
        current_var: xp.ndarray,
        log_mean: xp.ndarray,
        return_normalization: bool = False,
    ) -> xp.ndarray:
        return self.neglog_proposal_candidate_given_current(
            candidate=current_var,
            log_mean=log_mean,
            return_normalization=return_normalization,
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
            accept_total = xp.zeros((self.var_shape[0],))
            log_proba_accept_total = xp.zeros((self.var_shape[0],))
        elif self.acceptance_wise_type == "full":
            summing_axes = xp.array([])
            accept_total = xp.zeros(self.var_shape)
            log_proba_accept_total = xp.zeros(self.var_shape)
        elif self.acceptance_wise_type == "entire_map":
            summing_axes = xp.arange(current_var.ndim)
            accept_total = xp.zeros((1,))
            log_proba_accept_total = xp.zeros((1,))
        self.summing_axes = tuple(summing_axes)

        self.chromatic_gibbs = False
        self.objective_type = "objective_pix_global"

        (
            candidate,
            mean_candidate_given_current,
            log_mean_candidate_given_current,
            mean_current_given_candidate,
            log_mean_current_given_candidate,
        ) = self.get_means_and_candidate_mh(
            current=current,
            t=t,
            target_distributions=target_distributions,
            rng=rng,
        )
        if (
            self.mean_type == "current"
        ):  # Because we are dealing with log variables. The location parameter is the not the log of the log. It is just the log.
            log_mean_current_given_candidate = mean_current_given_candidate * 1
            log_mean_candidate_given_current = mean_candidate_given_current * 1

        neglog_proposal_candidate_given_current = (
            self.neglog_proposal_candidate_given_current(
                candidate=candidate,
                log_mean=log_mean_candidate_given_current,
            )
        )

        neglog_proposal_current_given_candidate = (
            self.neglog_proposal_current_given_candidate(
                current_var=current[self.key]["var"],
                log_mean=log_mean_current_given_candidate,
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
        }  # Add a dimension for the candidates.
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

        if accept_total.max() > 0:  # if at least one accept and not the last site group
            # update the current variable for the next site group
            current[self.key]["var"] = new_var * 1
            current[self.key] = target_distributions[self.key].compute_all(
                current,
                compute_derivatives_2nd_order=compute_derivatives_2nd_order,
            )

        return current, accept_total, log_proba_accept_total

    def get_new_sample_optim(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        raise NotImplementedError(
            "Optimization is not implemented for KernelLogMultiplicativeNoiseIMTM"
        )


class KernelLogMultiplicativeNoiseIMTM(KernelIndependentMTM, MultiplicativeNoise):
    r"""Dataclass that implement the proposal distribution based on multiplicative noise (log scale)"""

    __slots__ = ("k_mtm", "plot_name")

    def __init__(
        self,
        key: str,
        k_mtm: int,
        mean_type: str,
        target_distribution: TargetDistribution,
        key_forward_var: str,
        scaling_factor_sigma_m: float = 1.0,
        joint: bool = False,
        acceptance_wise_type: str = "pixelwise",
        plot_name: str = "LogMult_noise_IMTM",
        **kwargs: dict,
    ) -> None:
        super().__init__(
            key=key,
            target_distribution=target_distribution,
            scaling_factor_sigma_m=scaling_factor_sigma_m,
            joint=joint,
            acceptance_wise_type=acceptance_wise_type,
            mean_type=mean_type,
            key_forward_var=key_forward_var,
        )
        self.k_mtm = k_mtm
        self.plot_name = plot_name

    def sample(
        self,
        log_mean: xp.ndarray,
        rng: xp.random.Generator,
    ) -> xp.ndarray:
        r"""draws a random vectors from a Lognormal distribution.

        Parameters
        ----------
        mean : xp.ndarray
            current iterate, mean of the Gaussian proposal
        rng : xp.random.Generator
            random number generator

        Returns
        -------
        xp.array of shape (n_pix, self.k_mtm, D)
        """
        sigma_m = self.sigma_m[:, None].repeat(self.k_mtm, axis=1)
        sigma_m2 = self.sigma_m2[:, None].repeat(self.k_mtm, axis=1)

        samples = log_mean[:, None] + rng.normal(
            mean=-(sigma_m2 * self.scaling_factor_sigma_m**2) / 2,
            sigma=(sigma_m * self.scaling_factor_sigma_m),
        )

        return samples

    def neglog_proposal_candidates_current(
        self,
        candidates: xp.ndarray,
        log_mean: xp.ndarray,
    ) -> xp.ndarray:
        _pdf = (
            candidates
            - log_mean[:, None]
            + (self.scaling_factor_sigma_m**2 * self.sigma_m2) / 2
        ) ** 2
        _pdf /= 2 * (self.scaling_factor_sigma_m**2 * self.sigma_m2)
        _pdf = _pdf.sum(axis=self.summing_axes)

        return _pdf

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
            accept_total = xp.zeros((self.var_shape[0],))
            log_rg_total = xp.zeros((self.var_shape[0],))
        elif self.acceptance_wise_type == "full":
            summing_axes = xp.array([])
            accept_total = xp.zeros(self.var_shape)
            log_rg_total = xp.zeros(self.var_shape)
        elif self.acceptance_wise_type == "entire_map":
            summing_axes = xp.arange(current_var.ndim)
            accept_total = xp.zeros((1,))
            log_rg_total = xp.zeros((1,))
        summing_axes[1:] += 1
        self.summing_axes = tuple(summing_axes)

        self.chromatic_gibbs = False
        self.objective_type = "objective_pix_global"

        # * Sample from the proposal distribution
        (
            candidates,
            _,
            log_mean,
        ) = self.get_means_and_candidate_mh(
            current=current,
            t=t,
            target_distributions=target_distributions,
            rng=rng,
        )

        # * Evaluate the neglog-pdf of the proposal distribution
        current_candidates = {
            k: {
                "var": xp.repeat(
                    current[k]["var"][:, xp.newaxis],
                    self.k_mtm + 1,
                    axis=1,
                )
            }
            for k in current.keys()
        }  # Add a dimension for the candidates.
        current_candidates[self.key]["var"][:, :-1] = candidates

        neglog_proposal_candidates_current = self.neglog_proposal_candidates_current(
            candidate=current_candidates[self.key]["var"],
            log_mean=log_mean,
        )
        assert len(neglog_proposal_candidates_current.shape) == len(self.summing_axes)

        # * Evaluate the neglog-pdf of the target distribution
        target_distributions[self.key].update_nlpdf_utils(
            current_candidates,
            idx_pix=None,
            compute_derivatives=False,
            compute_derivatives_2nd_order=False,
            mtm=True,
        )

        neglogpdf_candidates_current = (
            target_distributions[self.key]
            .neglog_pdf(
                full=True,
                update_nlpdf_utils=False,
                idx_pix=None,
                chromatic_gibbs=self.chromatic_gibbs,
            )
            .sum(axis=self.summing_axes)
        )
        assert len(neglogpdf_candidates_current.shape) == len(self.summing_axes)
        assert xp.isnan(neglogpdf_candidates_current).sum() == 0

        # * Accept-reject step
        new_var, accept_total, log_rg_total = self.accept_reject(
            current_candidates[self.key]["var"][:, :-1],
            current_var,
            neglogpdf_candidates_current,
            neglog_proposal_candidates_current,
            self.acceptance_wise_type,
            rng,
            joint=False,
        )

        if accept_total.max() > 0:  # if at least one accept and not the last site group
            # update the current variable for the next site group
            current[self.key]["var"] = new_var * 1
            current[self.key] = target_distributions[self.key].compute_all(
                current,
                compute_derivatives_2nd_order=compute_derivatives_2nd_order,
            )

        return current, accept_total, log_rg_total

    def get_new_sample_optim(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        raise NotImplementedError(
            "Optimization is not implemented for KernelLogMultiplicativeNoiseIMTM"
        )
