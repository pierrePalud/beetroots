### OUTDATED: has been moved into mixture_neighbors

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from typing import Dict, Optional, Tuple, Union

from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.kernels.abstract_kernel import KernelUtils
from beetroots.sampler.kernels.abstract_kernel_ind_mtm import KernelIndependentMTM
from beetroots.sampler.utils import utils


class MixtureNeighborsIndicator(KernelUtils):
    __slots__ = (
        "key",
        "is_stochastic",
        "k_mtm",
        "list_edges",
        "weights",
        "chromatic_gibbs",
        "objective_type",
        "var_shape",
        "plot_name",
        "acceptance_wise_type",
    )

    def __init__(
        self,
        key: str,
        is_stochastic: str,
        target_distribution: TargetDistribution,
        k_mtm: int,
        weights: Union[xp.ndarray, bool],
        acceptance_wise_type: str,
        plot_name: str = "MTM_mixture_neighbors",
    ):
        self.key = key
        self.is_stochastic = is_stochastic
        if not isinstance(plot_name, str):
            raise TypeError("plot_name should be a string")
        self.plot_name = plot_name

        if not isinstance(k_mtm, int):
            raise TypeError("k_mtm should be an integer")
        self.k_mtm = k_mtm

        if not (isinstance(weights, (xp.ndarray, bool))):
            raise TypeError("weights should be a numpy array or a boolean")
        self.weights = weights

        self.list_edges = False
        self.lower_bounds = False
        self.upper_bounds = False
        self.indicator_margin_scale = False

        for comp in target_distribution.distribution_components.values():
            if self.list_edges is False:
                self.list_edges = getattr(comp, "list_edges", False)
            if self.lower_bounds is False:
                self.lower_bounds = getattr(comp, "lower_bounds", False)
            if self.upper_bounds is False:
                self.upper_bounds = getattr(comp, "upper_bounds", False)
            if self.indicator_margin_scale is False:
                self.indicator_margin_scale = getattr(
                    comp, "indicator_margin_scale", False
                )

        if self.list_edges is False:
            raise ValueError("No list_edges found in the distribution components")
        if self.lower_bounds is False:
            raise ValueError("No lower_bounds found in the distribution components")
        if self.upper_bounds is False:
            raise ValueError("No upper_bounds found in the distribution components")
        if self.indicator_margin_scale is False:
            raise ValueError(
                "No indicator_margin_scale found in the distribution components"
            )

        assert acceptance_wise_type in ["pixelwise", "full", "entire_map"]
        self.acceptance_wise_type = acceptance_wise_type


class KernelMixtureNeighborsIndicator(KernelIndependentMTM, MixtureNeighborsIndicator):
    r"""Dataclass that implement the proposal distribution based on the smooth indicator prior and the spatial prior"""

    def __init__(
        self,
        key: str,
        target_distribution: TargetDistribution,
        k_mtm: int,
        is_stochastic: bool = False,
        acceptance_wise_type: str = "pixelwise",
        weights: Optional[Union[xp.ndarray, bool]] = False,
        **kwargs: dict,
    ) -> None:
        r"""
        Parameters
        ----------
        key : str
        list_edges : List[tuple[int, int]]
            list of edges
        weights : xp.ndarray of shape (n_edges,)
        """
        super().__init__(
            key=key,
            is_stochastic=is_stochastic,
            target_distribution=target_distribution,
            k_mtm=k_mtm,
            weights=weights,
            acceptance_wise_type=acceptance_wise_type,
        )

        self.chromatic_gibbs = None
        self.objective_type = None
        self.var_shape = None

    def sample(
        self,
        current_Var: xp.ndarray,
        idx_pix: Union[dict, xp.ndarray],
        k_mtm,
        rng: xp.random.Generator,
    ) -> xp.ndarray:
        r"""draws a random vectors for components :math:`n` (e.g., a pixel :math:`\theta_n`). The distribution used to draw these vectors is:

        * the smooth indicator prior
        * a combination of the smooth indicator prior and of a Gaussian mixture defined with the set of all combinations of neighbors of component :math:`n`

        Parameters
        ----------
        Theta : xp.ndarray
            current iterate
        idx_pix : xp.ndarray
            indices of the pixels

        Returns
        -------
        xp.array of shape (n_pix, self.k_mtm, D)
            random element of the hypercube defined by the lower and upper bounds with uniform distribution

        Raises
        ------
        ValueError : if ``target_distribution.prior_indicator`` is None
        """
        seed = rng.integers(0, 1_000_000_000)

        if self.weights is False:
            weights = xp.ones(self.var_shape[-1])
        else:
            weights = self.weights

        samples = utils.sample_conditional_spatial_and_indicator_prior(
            current_Var,
            self.list_edges,
            spatial_weights=weights,
            indicator_lower_bounds=self.lower_bounds,
            indicator_upper_bounds=self.upper_bounds,
            indicator_margin_scale=self.indicator_margin_scale,
            idx_pix=idx_pix,
            k_mtm=k_mtm,
            seed=seed,
        )  # (n_pix, self.k_mtm, D)

        return samples

    def neglog_proposal_candidates_current(
        self,
        candidates,
        idx_pix,
    ) -> xp.ndarray:
        n_pix = idx_pix.size
        k_mtm = candidates.shape[1]

        if self.weights is False:
            weights = xp.ones(candidates.shape[-1])
        else:
            weights = self.weights

        _pdf = utils.compute_nlpdf_spatial_indicator_proposal(
            candidates,
            self.list_edges,
            weights,
            idx_pix,
            indicator_lowerbounds=self.lower_bounds,
            indicator_upperbounds=self.upper_bounds,
        )

        return _pdf.reshape((n_pix, k_mtm))

    def get_new_sample_sampling(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        new_var = current[self.key]["var"] * 1  # (N, D)
        self.var_shape = new_var.shape

        if self.acceptance_wise_type == "pixelwise":
            summing_axes = xp.arange(1, len(self.var_shape))
            accept_total = xp.zeros((self.var_shape[0],))
            log_rg_total = xp.zeros((self.var_shape[0],))
        elif self.acceptance_wise_type == "full":
            summing_axes = xp.array([])
            accept_total = xp.zeros(self.var_shape)
            log_rg_total = xp.zeros(self.var_shape)
        elif self.acceptance_wise_type == "entire_map":
            summing_axes = xp.arange(len(self.var_shape))
            accept_total = xp.zeros((1,))
            log_rg_total = xp.zeros((1,))

        target_distribution = target_distributions[self.key]
        list_idx = xp.array(list(target_distribution.dict_sites.keys()))
        self.chromatic_gibbs = not (len(list_idx) == 1)
        self.objective_type = (
            "objective_pix_chromatic"
            if self.chromatic_gibbs
            else "objective_pix_global"
        )

        for idx_site in list_idx:
            idx_pix = target_distribution.dict_sites[idx_site]
            n_pix = idx_pix.size

            current_var = new_var * 1

            candidates = self.sample(
                current_var,
                idx_pix,
                self.k_mtm,
                rng,
            )  # (n_pix, self.k_mtm, ...)

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
            current_candidates[self.key]["var"][idx_pix, :-1] = candidates

            target_distribution.update_nlpdf_utils(
                current_candidates,
                idx_pix,
                compute_derivatives=False,
                compute_derivatives_2nd_order=False,
                mtm=True,
            )
            neglogpdf_candidates_current = target_distribution.neglog_pdf(
                pixelwise=True,
                update_nlpdf_utils=False,
                chromatic_gibbs=self.chromatic_gibbs,
            )
            assert neglogpdf_candidates_current.shape == (
                n_pix,
                self.k_mtm + 1,
            )
            assert xp.isnan(neglogpdf_candidates_current).sum() == 0

            neglog_proposal_candidates_current = (
                self.neglog_proposal_candidates_current(
                    current_candidates[self.key]["var"], idx_pix
                )
            )
            shape_ = neglog_proposal_candidates_current.shape
            assert shape_ == (n_pix, self.k_mtm + 1)

            candidate_accepted, accept_arr, log_rg = self.accept_reject(
                current_candidates[self.key]["var"][idx_pix],
                current_var[idx_pix],
                neglogpdf_candidates_current,
                neglog_proposal_candidates_current,
                self.acceptance_wise_type,
                rng,
                joint=False,
            )
            new_var[idx_pix] = candidate_accepted * 1

            accept_total[idx_pix] = accept_arr * 1
            log_rg_total[idx_pix] = log_rg * 1

        if accept_total.max() > 0:
            current[self.key]["var"] = new_var * 1

            current[self.key] = target_distribution.compute_all(
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
        new_var = current[self.key]["var"] * 1  # (N, D)
        self.var_shape = new_var.shape

        if self.acceptance_wise_type == "pixelwise":
            summing_axes = xp.arange(1, len(self.var_shape))
            accept_total = xp.zeros((self.var_shape[0],))
            log_rg_total = xp.zeros((self.var_shape[0],))
        elif self.acceptance_wise_type == "full":
            summing_axes = xp.array([])
            accept_total = xp.zeros(self.var_shape)
            log_rg_total = xp.zeros(self.var_shape)
        elif self.acceptance_wise_type == "entire_map":
            summing_axes = xp.arange(len(self.var_shape))
            accept_total = xp.zeros((1,))
            log_rg_total = xp.zeros((1,))

        target_distribution = target_distributions[self.key]
        list_idx = xp.array(list(target_distribution.dict_sites.keys()))
        self.chromatic_gibbs = not (len(list_idx) == 1)
        self.objective_type = (
            "objective_pix_chromatic"
            if self.chromatic_gibbs
            else "objective_pix_global"
        )

        for idx_site in list_idx:
            idx_pix = target_distribution.dict_sites[idx_site]
            n_pix = idx_pix.size

            current_var = current[self.key]["var"] * 1

            candidates = self.sample(
                current_var,
                idx_pix,
                self.k_mtm,
                rng,
            )  # (n_pix, self.k_mtm, ...)

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
            current_candidates[self.key]["var"][idx_pix, :-1] = candidates

            target_distribution.update_nlpdf_utils(
                current_candidates,
                idx_pix,
                compute_derivatives=False,
                compute_derivatives_2nd_order=False,
                mtm=True,
            )
            neglogpdf_candidates_current = target_distribution.neglog_pdf(
                pixelwise=True,
                update_nlpdf_utils=False,
                chromatic_gibbs=self.chromatic_gibbs,
            )
            assert neglogpdf_candidates_current.shape == (
                n_pix,
                self.k_mtm + 1,
            )
            assert xp.isnan(neglogpdf_candidates_current).sum() == 0

            idx_challengers = xp.argmin(
                neglogpdf_candidates_current[:, :-1].sum(axis=0)
            )  # integer
            neglogpdf_candidates_challengers = neglogpdf_candidates_current[
                :, idx_challengers
            ]
            challengers = candidates[idx_pix, idx_challengers]

            assert neglogpdf_candidates_challengers.shape == (
                n_pix,
            ), neglogpdf_candidates_challengers.shape

            # challengers = candidates_pix[
            #     np.arange(len(candidates_pix)), idx_challengers, :
            # ]
            assert challengers.shape == (n_pix, self.D), challengers.shape

            # * compute values of corresponding pixels in current x
            candidates_already_Theta = candidates[idx_pix, -1, :] * 1
            neglogpdf_already_Theta = neglogpdf_candidates_current[:, -1] * 1
            assert candidates_already_Theta.shape == (n_pix, self.D)
            assert neglogpdf_already_Theta.shape == (n_pix,)

            # * select best pixels
            accept_arr = (
                (neglogpdf_candidates_challengers < neglogpdf_already_Theta)
                & xp.isfinite(neglogpdf_candidates_challengers)
                & xp.isfinite(neglogpdf_already_Theta)
            )

            new_var[idx_pix, :] = xp.where(
                accept_arr[:, None] * xp.ones((n_pix, self.D)),
                challengers,  # (n_pix, D)
                candidates_already_Theta,  # (n_pix, D)
            )
            accept_total[idx_pix] = accept_arr * 1

        if accept_total.max() > 0:
            current[self.key]["var"] = new_var * 1

            for k, target_dist in target_distributions.items():
                current[k] = target_distributions[k].compute_all(
                    current,
                    compute_derivatives_2nd_order=compute_derivatives_2nd_order,
                )

        return current, accept_total.mean(), accept_total.mean()
