from typing import Optional

import numba as nb

try:
    import cupy as xp  # type: ignore

    decorator_nb = nb.cuda.jit
except ImportError:
    import numpy as xp

    decorator_nb = nb.njit

from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior


@decorator_nb()
def compute_laplacian(
    Theta: xp.ndarray, list_edges: xp.ndarray, idx_pix: xp.ndarray
) -> xp.ndarray:
    r"""evaluates the image Laplacian for each of the D maps, :math:`\Delta \Theta_{\cdot d}`

    Parameters
    ----------
    Theta : xp.ndarray of shape (N, D)
        D vectors of N pixels
    list_edges : xp.ndarray
        set of edges in the graph induced by the spatial regularization

    Returns
    -------
    xp.ndarray of shape (n_pix, D)
        image Laplacian for each of the D maps
    """
    laplacian_ = xp.zeros((idx_pix.size, *Theta.shape[1:]))

    for i, n in enumerate(idx_pix):
        mask_i_m = list_edges[:, 1] == n
        mask_i_p = list_edges[:, 0] == n

        laplacian_[i] -= xp.sum(
            (Theta[list_edges[mask_i_p, 1], :] - Theta[list_edges[mask_i_p, 0], :]),
            axis=0,
        )
        laplacian_[i] += xp.sum(
            (Theta[list_edges[mask_i_m, 1], :] - Theta[list_edges[mask_i_m, 0], :]),
            axis=0,
        )
    return laplacian_  # (N, D)


@decorator_nb()
def compute_hadamard_discrete_gradient(
    Theta: xp.ndarray, list_edges: xp.ndarray, idx_pix: xp.ndarray
) -> xp.ndarray:
    r"""evaluates the sum of squared differences for each of the D maps, :math:`\Delta \Theta_{\cdot d}`

    Parameters
    ----------
    Theta : xp.ndarray of shape (N, D)
        D vectors of N pixels
    list_edges : xp.ndarray
        set of edges in the graph induced by the spatial regularization
    idx_pix : xp.ndarray
        array of the indices of the pixels to consider

    Returns
    -------
    xp.ndarray of shape (n_pix, D)
        image of  for each of the D maps
    """
    hadamard_gradient_ = xp.zeros((idx_pix.size, *Theta.shape[1:]))

    for i, n in enumerate(idx_pix):
        mask_i_m = list_edges[:, 1] == n
        mask_i_p = list_edges[:, 0] == n

        hadamard_gradient_[i] += xp.sum(
            (Theta[list_edges[mask_i_p, 1], :] - Theta[list_edges[mask_i_p, 0], :])
            ** 2,
            axis=0,
        )
        hadamard_gradient_[i] += xp.sum(
            (Theta[list_edges[mask_i_m, 1], :] - Theta[list_edges[mask_i_m, 0], :])
            ** 2,
            axis=0,
        )
    return hadamard_gradient_  # (N, D)


class L22DiscreteGradSpatialPrior(SpatialPrior):
    r"""L22 smooth spatial prior, valid for both 1D and 2D tensors. Its pdf is defined as

    .. math::

        \forall d \in [1, D], \quad \pi(\Theta_{\cdot d}) \propto \exp \left[- \tau_d \Vert D \Theta_{\cdot d} \Vert_F^2 \right] = \exp \left[- \frac{\tau_d}{2} \sum_{n \in \mathcal{N}} \sum_{i\in V_n}\left( \Theta_{n, d} - \Theta_{i, d} \right)^2 \right]

    where  :math:`\Vert \cdot \Vert_F` denotes the Fröbenius norm and :math:`D\Theta_{\cdot d}` is the discrete gradient operator for vector :math:\Theta_{\cdot d}`.

    It does handle the MTM case with an additional dimension (thourgh the `full` dimension).
    """

    def neglog_pdf(
        self,
        with_weights: bool = True,
        full: bool = False,
        pixelwise: bool = False,
        paramwise: bool = False,
        chromatic_gibbs: bool = False,
        **kwargs,
    ) -> xp.ndarray:
        assert xp.sum([pixelwise, full, paramwise]) < 2
        if self.nlpdf_utils["idx_pix"].size < self.N:
            chromatic_gibbs = True
        factor = 1 if chromatic_gibbs else 1 / 2

        hadamard_gradient_ = self.nlpdf_utils["hadamard_discrete_gradient"] * 1
        neglog_p = hadamard_gradient_

        if with_weights:
            neglog_p = self.weights * neglog_p

        if pixelwise:
            neglog_p = xp.sum(neglog_p, axis=-1)
        elif paramwise:
            neglog_p = xp.sum(neglog_p, axis=0)
        elif not full:
            if not self.nlpdf_utils["is_mtm"]:
                neglog_p = xp.sum(neglog_p)
            else:
                xp.sum(neglog_p.swapaxes(0, 1), axis=tuple(range(1, neglog_p.ndim)))

        if isinstance(chromatic_gibbs, bool):
            return factor * neglog_p  # (D,) if not pixelwise or (N, D) if pixelwise
        elif chromatic_gibbs == "both":
            return 1 * neglog_p, 1 / 2 * neglog_p
        else:
            raise ValueError(
                'wrong value for "chromatic_gibbs", should be a boolean or the string "both"'
            )

    def gradient_neglog_pdf(self, **kwargs) -> xp.ndarray:
        laplacian_ = self.nlpdf_utils["laplacian"] * 1

        return self.weights[None, :] * 2 * laplacian_  # (n_pix, D)

    def hess_diag_neglog_pdf(self, **kwargs) -> xp.ndarray:
        hess_diag = xp.zeros(self.nlpdf_utils["hadamard_discrete_gradient"].shape)

        if self.list_edges.size > 0:
            idx, counts = xp.unique(self.list_edges.flatten(), return_counts=True)
            indices_kept_1 = xp.isin(
                self.nlpdf_utils["idx_pix"], idx, assume_unique=True
            )
            indices_kept_2 = xp.isin(
                idx, self.nlpdf_utils["idx_pix"], assume_unique=True
            )

            hess_diag[indices_kept_1, :] += counts[indices_kept_2, None] * xp.ones(
                (indices_kept_1.sum(), self.D)
            )

        return self.weights[None, :] * 2 * hess_diag  # (n_pix, D)

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

        self.nlpdf_utils["idx_pix"] = idx_pix
        self.nlpdf_utils["is_mtm"] = mtm

        self.nlpdf_utils["hadamard_discrete_gradient"] = (
            (compute_hadamard_discrete_gradient(Var, self.list_edges, idx_pix))
            if self.list_edges.size > 0
            else xp.zeros((idx_pix.size, *Var.shape[1:]))
        )

        if compute_derivatives:
            self.nlpdf_utils["laplacian"] = (
                compute_laplacian(Var, self.list_edges, idx_pix)
                if self.list_edges.size > 0
                else xp.zeros((idx_pix.size, *Var.shape[1:]))
            )
