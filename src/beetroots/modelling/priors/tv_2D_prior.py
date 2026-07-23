"""Defines the Total variation spatial regularization and its derivatives"""

import numba as nb

try:
    import cupy as xp  # type: ignore

    decorator_nb = nb.cuda.jit
except ImportError:
    import numpy as xp

    decorator_nb = nb.njit

from typing import Optional

import pandas as pd

from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior


@decorator_nb(nopython=True)  # , cache=True)
def compute_tv_matrix(
    Var: xp.ndarray, list_edges: xp.ndarray, eps: float
) -> xp.ndarray:
    tv_matrix_ = xp.zeros_like(Var) + eps

    N = tv_matrix_.shape[0]
    for i in range(N):
        # mask_i_m = list_edges[:, 1] == i
        mask_i_p = list_edges[:, 0] == i

        tv_matrix_[i] += xp.sum(
            (Var[list_edges[mask_i_p, 1], :] - Var[list_edges[mask_i_p, 0], :]) ** 2,
            axis=0,
        )
    return xp.sqrt(tv_matrix_)  # (N, D)


@decorator_nb(nopython=True)
def compute_gradient_from_tv_matrix(
    Var: xp.ndarray, tv_matrix_: xp.ndarray, list_edges: xp.ndarray
) -> xp.ndarray:
    grad_ = xp.zeros_like(Var, dtype=xp.float64)
    for edge in list_edges:
        val = (Var[edge[1]] - Var[edge[0]]) / tv_matrix_[edge[0]]  # (D,)
        grad_[edge[0], :] -= val
        grad_[edge[1], :] += val

    return grad_


class TVeps2DSpatialPrior(SpatialPrior):
    def __init__(
        self,
        D: int,
        N: int,
        df: pd.DataFrame,
        weights: xp.ndarray = None,
        eps: float = 1e-3,
    ):
        super().__init__(D, N, df, weights)
        self.eps = eps

    def neglog_pdf(
        self,
        Var: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
        with_weights: bool = True,
        **kwargs,
    ) -> xp.ndarray:
        assert Var.shape == (self.N, self.D)

        if self.list_edges.size == 0:
            return xp.zeros((self.D,))

        tv_matrix_ = compute_tv_matrix(Var, self.list_edges, self.eps)
        nlpdf = xp.sum(tv_matrix_, axis=0)

        if with_weights:
            nlpdf *= self.weights

        return nlpdf  # (D,)

    def gradient_neglog_pdf(self, Var: xp.ndarray) -> xp.ndarray:
        assert Var.shape == (self.N, self.D)
        if self.list_edges.size == 0:
            return xp.zeros_like(Var, dtype=xp.float64)

        tv_matrix_ = compute_tv_matrix(Var, self.list_edges, self.eps)  # (N, D)

        grad_ = compute_gradient_from_tv_matrix(Var, tv_matrix_, self.list_edges)
        grad_ = grad_ * self.weights[None, :]
        return grad_  # (N, D)

    def hess_diag_neglog_pdf(self, Var: xp.ndarray) -> xp.ndarray:
        # TODO
        assert Var.shape == (self.N, self.D)

        hess_diag = xp.zeros_like(Var, dtype=xp.float64)

        #! unfinished
        # for edge in self.list_edges:
        #     # val = 1 / xp.sqrt((x[edge[1]] - x[edge[0]]) ** 2 + self.eps) - (
        #     #     x[edge[1]] - x[edge[0]]
        #     # ) ** 2 * ((x[edge[1]] - x[edge[0]]) ** 2 + self.eps) ** (
        #     #     -3 / 2
        #     # )  # (D,)
        #     delta_x = x[edge[1]] - x[edge[0]]
        #     val = 2 * self.eps / (delta_x ** 2 + self.eps) ** (3 / 2)
        #     hess_diag[edge[0], :] += val
        #     hess_diag[edge[1], :] += val

        hess_diag = hess_diag * self.weights[None, :]
        return hess_diag
