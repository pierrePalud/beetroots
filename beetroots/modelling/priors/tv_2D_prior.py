"""Defines the Total variation spatial regularization and its derivatives
"""
import numba
import numpy as np
import pandas as pd

from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior


@numba.jit(nopython=True)  # , cache=True)
def compute_tv_matrix(
    Theta: np.ndarray, list_edges: np.ndarray, eps: float
) -> np.ndarray:
    tv_matrix_ = np.zeros_like(Theta) + eps

    N = tv_matrix_.shape[0]
    for i in range(N):
        # mask_i_m = list_edges[:, 1] == i
        mask_i_p = list_edges[:, 0] == i

        tv_matrix_[i] += np.sum(
            (Theta[list_edges[mask_i_p, 1], :] - Theta[list_edges[mask_i_p, 0], :])
            ** 2,
            axis=0,
        )
    return np.sqrt(tv_matrix_)  # (N, D)


@numba.jit(nopython=True)
def compute_gradient_from_tv_matrix(
    Theta: np.ndarray, tv_matrix_: np.ndarray, list_edges: np.ndarray
) -> np.ndarray:
    grad_ = np.zeros_like(Theta, dtype=np.float64)
    for edge in list_edges:
        val = (Theta[edge[1]] - Theta[edge[0]]) / tv_matrix_[edge[0]]  # (D,)
        grad_[edge[0], :] -= val
        grad_[edge[1], :] += val

    return grad_


class TVeps2DSpatialPrior(SpatialPrior):
    def __init__(
        self,
        D: int,
        N: int,
        df: pd.DataFrame,
        weights: np.ndarray = None,
        eps: float = 1e-3,
    ):
        super().__init__(D, N, df, weights)
        self.eps = eps

    def neglog_pdf(self, Theta: np.ndarray, with_weights: bool = True) -> np.ndarray:
        assert Theta.shape == (self.N, self.D)

        if self.list_edges.size == 0:
            return np.zeros((self.D,))

        tv_matrix_ = compute_tv_matrix(Theta, self.list_edges, self.eps)
        nlpdf = np.sum(tv_matrix_, axis=0)

        if with_weights:
            nlpdf *= self.weights

        return nlpdf  # (D,)

    def gradient_neglog_pdf(self, Theta: np.ndarray) -> np.ndarray:
        assert Theta.shape == (self.N, self.D)
        if self.list_edges.size == 0:
            return np.zeros_like(Theta, dtype=np.float64)

        tv_matrix_ = compute_tv_matrix(Theta, self.list_edges, self.eps)  # (N, D)

        grad_ = compute_gradient_from_tv_matrix(Theta, tv_matrix_, self.list_edges)
        grad_ = grad_ * self.weights[None, :]
        return grad_  # (N, D)

    def hess_diag_neglog_pdf(self, Theta: np.ndarray) -> np.ndarray:
        # TODO
        assert Theta.shape == (self.N, self.D)

        hess_diag = np.zeros_like(Theta, dtype=np.float64)

        #! unfinished
        # for edge in self.list_edges:
        #     # val = 1 / np.sqrt((x[edge[1]] - x[edge[0]]) ** 2 + self.eps) - (
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
