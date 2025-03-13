"""(deprecated) Implementation of L22 Laplacian prior. False implementation."""

import warnings
from typing import Optional

import numba
import numpy as np
import pandas as pd
from numba.typed import Dict, List

from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior


@numba.njit()
def compute_laplacian(
    Theta: np.ndarray, list_edges: np.ndarray, idx_pix: np.ndarray
) -> np.ndarray:
    r"""evaluates the image Laplacian for each of the D maps, :math:`\Delta \Theta_{\cdot d}`

    Parameters
    ----------
    Theta : np.ndarray of shape (N, D)
        D vectors of N pixels
    list_edges : np.ndarray
        set of edges in the graph induced by the spatial regularization

    Returns
    -------
    np.ndarray of shape (N, D)
        image Laplacian for each of the D maps
    """
    laplacian_ = np.zeros((idx_pix.size, *Theta.shape[1:]))

    for i, n in enumerate(idx_pix):
        mask_i_m = list_edges[:, 1] == n
        mask_i_p = list_edges[:, 0] == n

        laplacian_[i] -= np.sum(
            (Theta[list_edges[mask_i_p, 1], :] - Theta[list_edges[mask_i_p, 0], :]),
            axis=0,
        )
        laplacian_[i] += np.sum(
            (Theta[list_edges[mask_i_m, 1], :] - Theta[list_edges[mask_i_m, 0], :]),
            axis=0,
        )
    return laplacian_  # (N, D)


@numba.njit()
def compute_neglog_chromatic_from_laplacian2(
    laplacian_2: np.ndarray, list_edges: np.ndarray, idx_pix: np.ndarray
) -> np.ndarray:
    r"""evaluates the negative log pdf for each of the D maps, :math:`\Delta \Theta_{\cdot d}`

    Parameters
    ----------
    Theta : np.ndarray of shape (N, D)
        D vectors of N pixels
    dict_neighbors : np.ndarray
        set of neighbors for each pixel
    weights : np.ndarray
        weights for each pixel

    Returns
    -------
    np.ndarray of shape (N, D)
        negative log pdf for each of the D maps
    """
    neglog_p = np.zeros(
        (idx_pix.size, *laplacian_2.shape[1:])
    )  # (n_pix, D) or (n_pix, k_mtm, D)

    for i, n in enumerate(idx_pix):
        mask_i_m = list_edges[:, 1] == n
        mask_i_p = list_edges[:, 0] == n
        neighbors_indices = np.asarray(
            list(list_edges[mask_i_m, 0]) + list(list_edges[mask_i_p, 1]),
            dtype=np.int64,
        )

        neglog_p[i] += laplacian_2[n]
        neglog_p[i] += np.sum(laplacian_2[neighbors_indices, :], axis=0)

    return neglog_p


@numba.njit()
def compute_gradient_from_laplacian(
    laplacian_: np.ndarray, idx_pix: np.ndarray, list_edges: np.ndarray
) -> np.ndarray:
    """evaluates the gradient from the Laplacian matrix

    Parameters
    ----------
    laplacian_ : np.ndarray
        Laplacian matrix
    list_edges : np.ndarray
        array containing all the graph edges, identified with pairs of indices of neighboring pixels

    Returns
    -------
    np.ndarray
        gradient of the prior neg-log pdf
    """
    g = np.zeros((idx_pix.size, *laplacian_.shape[1:]))  # (n_pix, D)

    for i, n in enumerate(idx_pix):
        mask_i_m = list_edges[:, 1] == n
        mask_i_p = list_edges[:, 0] == n
        neighbors_indices = np.asarray(
            list(list_edges[mask_i_m, 0]) + list(list_edges[mask_i_p, 1]),
            dtype=np.int64,
        )

        g[i] = 2 * (
            neighbors_indices.size * laplacian_[n]
            - np.sum(laplacian_[neighbors_indices, :], axis=0)
        )

    return g  # (n_pix, D)


class L22LaplacianSpatialPrior(SpatialPrior):
    r"""L22 smooth spatial prior, valid for both 1D and 2D tensors. Its pdf is defined as

    .. math::

        \forall d \in [1, D], \quad \pi(\Theta_{\cdot d}) \propto \exp \left[- \tau_d \Vert \Delta \Theta_{\cdot d} \Vert_F^2 \right]

    where  :math:`\Vert \cdot \Vert_F` denotes the Fröbenius norm and :math:`\Delta \Theta_{\cdot d}` is the Laplacian of vector :math:ù\Theta_{\cdot d}`.
    """

    def neglog_pdf(
        self,
        Theta: np.ndarray,
        idx_pix: np.ndarray,
        with_weights: bool = True,
        full: bool = False,
        pixelwise: bool = False,
        chromatic_gibbs: bool = False,
        **kwargs,
    ) -> np.ndarray:
        assert np.sum([pixelwise, full]) < 2
        if idx_pix.size < self.N:
            chromatic_gibbs = True

        chromatic_gibbs = False
        warnings.warn(
            "Warning: Chromatic Gibbs is disabled in the L22LaplacianSpatialPrior"
        )

        if self.list_edges.size > 0:
            if not chromatic_gibbs:
                laplacian_ = compute_laplacian(Theta, self.list_edges, idx_pix=idx_pix)
                laplacian_2_ = laplacian_**2  # (N,D)
                neglog_p = laplacian_2_ * 1
            else:
                laplacian_ = compute_laplacian(
                    Theta, self.list_edges, idx_pix=np.arange(self.N)
                )
                laplacian_2_ = laplacian_**2  # (N,D)
                neglog_p = compute_neglog_chromatic_from_laplacian2(
                    laplacian_2_, list_edges=self.list_edges, idx_pix=idx_pix
                )  # (n_pix, D) or (n_pix, k_mtm, D)

        if with_weights:
            neglog_p *= self.weights

        if full:
            return neglog_p
        elif pixelwise:
            return np.sum(neglog_p, axis=1)
        else:
            return np.sum(neglog_p, axis=0)

    def gradient_neglog_pdf(self, Theta: np.ndarray, idx_pix: np.ndarray) -> np.ndarray:
        assert Theta.shape == (self.N, self.D)

        laplacian_ = compute_laplacian(
            Theta, idx_pix=np.arange(self.N), list_edges=self.list_edges
        )

        g = compute_gradient_from_laplacian(
            laplacian_, idx_pix, list_edges=self.list_edges
        )  # (N, D)
        # g /= self.N * self.D
        return self.weights[None, :] * g  # (N, D)

    def hess_diag_neglog_pdf(
        self, Theta: np.ndarray, idx_pix: np.ndarray
    ) -> np.ndarray:
        hess_diag = np.zeros((idx_pix.size, self.D))  # (n_pix, D)

        if self.list_edges.size > 0:
            idx, counts = np.unique(self.list_edges.flatten(), return_counts=True)
            indices_kept_1 = np.isin(idx_pix, idx, assume_unique=True)
            indices_kept_2 = np.isin(idx, idx_pix, assume_unique=True)

            hess_diag[indices_kept_1, :] += (
                2
                * (counts * (counts + 1))[indices_kept_2, None]
                * np.ones((indices_kept_1.sum(), self.D))
            )

        # hess_diag /= self.N * self.D
        return self.weights[None, :] * hess_diag  # (N, D)
