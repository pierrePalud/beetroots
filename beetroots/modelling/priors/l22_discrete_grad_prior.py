from typing import Optional

import numba
import numpy as np

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
    np.ndarray of shape (n_pix, D)
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


def compute_hadamard_discrete_gradient(
    Theta: np.ndarray, list_edges: np.ndarray, idx_pix: np.ndarray
) -> np.ndarray:
    r"""evaluates the sum of squared differences for each of the D maps, :math:`\Delta \Theta_{\cdot d}`

    Parameters
    ----------
    Theta : np.ndarray of shape (N, D)
        D vectors of N pixels
    list_edges : np.ndarray
        set of edges in the graph induced by the spatial regularization
    idx_pix : np.ndarray
        array of the indices of the pixels to consider

    Returns
    -------
    np.ndarray of shape (n_pix, D)
        image of  for each of the D maps
    """
    hadamard_gradient_ = np.zeros((idx_pix.size, *Theta.shape[1:]))

    for i, n in enumerate(idx_pix):
        mask_i_m = list_edges[:, 1] == n
        mask_i_p = list_edges[:, 0] == n

        hadamard_gradient_[i] += np.sum(
            (Theta[list_edges[mask_i_p, 1], :] - Theta[list_edges[mask_i_p, 0], :])
            ** 2,
            axis=0,
        )
        hadamard_gradient_[i] += np.sum(
            (Theta[list_edges[mask_i_m, 1], :] - Theta[list_edges[mask_i_m, 0], :])
            ** 2,
            axis=0,
        )
    return hadamard_gradient_  # (N, D)


class L22DiscreteGradSpatialPrior(SpatialPrior):
    r"""L22 smooth spatial prior, valid for both 1D and 2D tensors. Its pdf is defined as

    .. math::

        \forall d \in [1, D], \quad \pi(\Theta_{\cdot d}) \propto \exp \left[- \tau_d \Vert \Delta \Theta_{\cdot d} \Vert_F^2 \right]

    where  :math:`\Vert \cdot \Vert_F` denotes the Fröbenius norm and :math:`\Delta \Theta_{\cdot d}` is the Laplacian of vector :math:ù\Theta_{\cdot d}`.

    It does handle the MTM case with an additional dimension.
    """

    def neglog_pdf(
        self,
        Theta: np.ndarray,
        idx_pix: np.ndarray,
        with_weights: bool = True,
        full: bool = False,
        pixelwise: bool = False,
        chromatic_gibbs: bool = False,
    ) -> np.ndarray:
        assert np.sum([pixelwise, full]) < 2
        if idx_pix.size < self.N:
            chromatic_gibbs = True
        factor = 2 if chromatic_gibbs else 1

        if self.list_edges.size > 0:
            hadamard_gradient_ = compute_hadamard_discrete_gradient(
                Theta, self.list_edges, idx_pix
            )
            neglog_p = factor * hadamard_gradient_
        else:
            neglog_p = np.zeros((idx_pix.size, *Theta.shape[1:]))

        if with_weights:
            neglog_p = self.weights * neglog_p

        if full:
            return neglog_p
        elif pixelwise:
            neglog_p = np.sum(neglog_p, axis=1)
        else:
            neglog_p = np.sum(neglog_p, axis=0)

        # neglog_p /= self.N * self.D
        return neglog_p  # (D,) if not pixelwise or (N, D) if pixelwise

    def gradient_neglog_pdf(self, Theta: np.ndarray, idx_pix: np.ndarray) -> np.ndarray:
        assert Theta.shape == (self.N, self.D)

        laplacian_ = compute_laplacian(Theta, self.list_edges, idx_pix)

        return self.weights[None, :] * 4 * laplacian_  # (N, D)

    def hess_diag_neglog_pdf(
        self, Theta: np.ndarray, idx_pix: np.ndarray
    ) -> np.ndarray:
        hess_diag = np.zeros((idx_pix.size, *Theta.shape[1:]), dtype=np.float64)

        if self.list_edges.size > 0:
            idx, counts = np.unique(self.list_edges.flatten(), return_counts=True)
            indices_kept_1 = np.isin(idx_pix, idx, assume_unique=True)
            indices_kept_2 = np.isin(idx, idx_pix, assume_unique=True)

            hess_diag[indices_kept_1, :] += counts[indices_kept_2, None] * np.ones(
                (indices_kept_1.sum(), self.D)
            )

        # hess_diag /= self.N * self.D
        return self.weights[None, :] * 4 * hess_diag  # (N, D)
