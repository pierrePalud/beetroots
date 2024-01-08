"""Defines the Total variation spatial regularization and its derivatives (for 1D signals only)
"""
import numpy as np
import pandas as pd

from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior


class TVeps1DSpatialPrior(SpatialPrior):
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

    def neglog_pdf(self, x: np.ndarray, with_weights: bool = True) -> np.ndarray:
        assert x.shape == (self.N, self.D)

        if self.list_edges.size == 0:
            return np.zeros((self.D,))

        nlpdf = np.sum(
            np.sqrt(
                (x[self.list_edges[:, 1]] - x[self.list_edges[:, 0]]) ** 2 + self.eps
            ),
            axis=0,
        )
        if with_weights:
            nlpdf *= self.weights

        return nlpdf  # (D,)

    def gradient_neglog_pdf(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (self.N, self.D)
        grad_ = np.zeros_like(x, dtype=np.float64)
        if self.list_edges.size == 0:
            return grad_

        for edge in self.list_edges:
            val = (x[edge[1]] - x[edge[0]]) / np.sqrt(
                (x[edge[1]] - x[edge[0]]) ** 2 + self.eps
            )  # (D,)
            grad_[edge[0], :] -= val
            grad_[edge[1], :] += val

        grad_ = grad_ * self.weights[None, :]
        return grad_  # (N, D)

    def hess_diag_neglog_pdf(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (self.N, self.D)

        hess_diag = np.zeros_like(x, dtype=np.float64)

        for edge in self.list_edges:
            # val = 1 / np.sqrt((x[edge[1]] - x[edge[0]]) ** 2 + self.eps) - (
            #     x[edge[1]] - x[edge[0]]
            # ) ** 2 * ((x[edge[1]] - x[edge[0]]) ** 2 + self.eps) ** (
            #     -3 / 2
            # )  # (D,)
            delta_x = x[edge[1]] - x[edge[0]]
            val = 2 * self.eps / (delta_x**2 + self.eps) ** (3 / 2)
            hess_diag[edge[0], :] += val
            hess_diag[edge[1], :] += val

        hess_diag = hess_diag * self.weights[None, :]
        return hess_diag
