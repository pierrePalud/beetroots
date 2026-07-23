"""(deprecated) Implementation of L22 Laplacian prior. False implementation."""

from typing import Optional

import numba as nb

try:
    import cupy as xp  # type: ignore

    decorator_nb = nb.cuda.jit
except ImportError:
    import numpy as xp

    decorator_nb = nb.njit

from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior
from beetroots.sampler.utils.utils import get_neighboring_pixels


@decorator_nb()
def compute_laplacian(
    Var: xp.ndarray, idx_pix: xp.ndarray, list_edges: xp.ndarray
) -> xp.ndarray:
    """computes the local laplacian only for the one modified pixel

    Parameters
    ----------
    Var : xp.ndarray of shape (N, D)
        current iterate

    n : int
        the index of the pixel to consider (:math:`0 \leq n \leq N - 1`)

    list_pixel_candidates : xp.ndarray of shape (N_candidates, D)
        the list of all candidates for pixel n

    Returns
    -------
    dictionnrary containing the laplacian and the sum of the differences between the pixel and its neighbors
    """
    sum_diffs_ = xp.zeros((idx_pix.size, *Var.shape[1:]))  # (n_pix, k_mtm, D)

    for i, n in enumerate(idx_pix):
        neighbors_ = get_neighboring_pixels(
            Var, list_edges, n
        )  # (N_neighbours, k_mtm, D)
        N_neigh = neighbors_.shape[0]
        diffs_ = xp.expand_dims(Var[n], 0) - neighbors_

        for j in range(N_neigh):
            sum_diffs_[i] += diffs_[j]

    return sum_diffs_


@decorator_nb()
def compute_laplacian_global(Var: xp.ndarray, list_edges: xp.ndarray) -> xp.ndarray:
    """computes the Laplacian operator for all pixels

    Parameters
    ----------
    Var : xp.ndarray of shape (N, D)
        current iterate

    list_pixel_candidates : xp.ndarray of shape (N_candidates, D)
        the list of all candidates for pixel n

    Returns
    -------
    dictionnrary containing the laplacian and the sum of the differences between the pixel and its neighbors
    """
    sum_diffs_ = xp.zeros_like(Var)

    for i, edge in enumerate(list_edges):
        diff_ = Var[edge[0]] - Var[edge[1]]
        sum_diffs_[edge[0]] += diff_
        sum_diffs_[edge[1]] -= diff_

    return sum_diffs_


class L22LaplacianSpatialPrior(SpatialPrior):
    r"""L22 smooth spatial prior, valid for both 1D and 2D tensors. Its pdf is defined as

    .. math::

        \forall d \in [1, D], \quad \pi(\Theta_{\cdot d}) \propto \exp \left[- \tau_d \Vert \Delta \Theta_{\cdot d} \Vert_F^2 \right]

    where  :math:`\Vert \cdot \Vert_F` denotes the Fröbenius norm and :math:`\Delta \Theta_{\cdot d}` is the Laplacian of vector :math:ù\Theta_{\cdot d}`.
    """

    def neglog_pdf(
        self,
        with_weights: bool = True,
        pixelwise: bool = False,
        paramwise: bool = False,
        full: bool = False,
    ) -> xp.ndarray:
        k_mtm = self.nlpdf_utils["k_mtm"]
        n_pix = self.nlpdf_utils["n_pix"]

        assert xp.sum([pixelwise, paramwise, full]) <= 1

        neglog_p = (
            xp.zeros((n_pix, k_mtm, self.D)) if k_mtm > 0 else xp.zeros((n_pix, self.D))
        )

        if self.list_edges.size > 0:
            laplacian_ = self.nlpdf_utils["laplacian"] * 1
            neglog_p += laplacian_**2  # (n_pix,D)

        if with_weights:
            neglog_p *= self.weights

        axes_to_sum = []

        if full:
            return neglog_p  # (n_pix, D) or (n_pix, k_mtm, D)
        elif pixelwise:
            axes_to_sum += list(range(1, neglog_p.ndim))
        elif paramwise:
            axes_to_sum += list(range(0, neglog_p.ndim - 1))
        else:
            axes_to_sum += list(range(0, neglog_p.ndim))
        if k_mtm > 0:
            axes_to_sum = set(axes_to_sum) - {1}

        neglog_p = neglog_p.sum(axis=tuple(axes_to_sum))

        return neglog_p  # (n_pix, k_mtm, D,) or (n_pix, D) or (D,) or (k_mtm, D) depending on the values of pixelwise and k_mtm

    def gradient_neglog_pdf(self, chromatic_gibbs: bool = True, **kwargs) -> xp.ndarray:
        sums_diffs = self.nlpdf_utils["laplacian"] * 1
        factor = 2 if chromatic_gibbs else 4

        if self.list_edges.size > 0:
            idx, counts = xp.unique(self.list_edges.flatten(), return_counts=True)

        assert sums_diffs.shape[0] == self.nlpdf_utils["n_pix"]
        assert sums_diffs.shape[-1] == self.D

        return (
            factor * self.weights * counts[:, None] * sums_diffs[idx]
        )  # (n_pix, k_mtm, D) or (n_pix, D)

    def hess_diag_neglog_pdf(
        self, chromatic_gibbs: bool = True, **kwargs
    ) -> xp.ndarray:
        laplacian_ = self.nlpdf_utils["laplacian"] * 1
        hess_diag = xp.zeros_like(laplacian_, dtype=xp.float64)
        factor = 2 if chromatic_gibbs else 4

        if self.list_edges.size > 0:
            idx, counts = xp.unique(self.list_edges.flatten(), return_counts=True)

            hess_diag[idx, :] += (
                factor
                * (counts**2)[:, None, None]
                * xp.ones((idx.size, *laplacian_.shape[1:]))
            )

        return self.weights * hess_diag

    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
        mtm: bool = False,
        **kwargs,
    ) -> None:
        """Evaluate all utilities for the negative log-pdf and its eventual derivatives"""

        Var = current[self.var_name][
            "var"
        ]  # Should be of full size N not n_pix as in spatial prior there are dependencies between pixels from different idx_pix.

        assert Var.shape[0] == self.N
        assert Var.shape[-1] == self.D

        idx_pix = xp.arange(self.N) if idx_pix is None else idx_pix

        self.nlpdf_utils["mtm"] = mtm
        self.nlpdf_utils["k_mtm"] = Var.shape[1] if mtm else 0
        self.nlpdf_utils["n_pix"] = idx_pix.size
        self.nlpdf_utils["idx_pix"] = idx_pix

        if self.nlpdf_utils["n_pix"] == self.N:
            self.nlpdf_utils["laplacian"] = compute_laplacian_global(
                Var, self.list_edges
            )
        else:
            self.nlpdf_utils["laplacian"] = compute_laplacian(
                Var, idx_pix, self.list_edges
            )  # if we compute the gradient, we need to compute the laplacian for every pixel.            self.nlpdf_utils['laplacian'] = compute_laplacian(Var, idx_pix, self.list_edges) # if we compute the gradient, we need to compute the laplacian for every pixel.
