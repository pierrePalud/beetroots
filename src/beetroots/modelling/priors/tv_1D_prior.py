"""Defines the Total variation spatial regularization and its derivatives (for 1D signals only)"""

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from typing import Optional

import pandas as pd

from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior


class TVeps1DSpatialPrior(SpatialPrior):
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

        nlpdf = xp.sum(
            xp.sqrt(
                (Var[self.list_edges[:, 1]] - Var[self.list_edges[:, 0]]) ** 2
                + self.eps
            ),
            axis=0,
        )
        if with_weights:
            nlpdf *= self.weights

        return nlpdf  # (D,)

    def gradient_neglog_pdf(self, Var: xp.ndarray) -> xp.ndarray:
        assert Var.shape == (self.N, self.D)
        grad_ = xp.zeros_like(Var, dtype=xp.float64)
        if self.list_edges.size == 0:
            return grad_

        for edge in self.list_edges:
            val = (Var[edge[1]] - Var[edge[0]]) / xp.sqrt(
                (Var[edge[1]] - Var[edge[0]]) ** 2 + self.eps
            )  # (D,)
            grad_[edge[0], :] -= val
            grad_[edge[1], :] += val

        grad_ = grad_ * self.weights[None, :]
        return grad_  # (N, D)

    def hess_diag_neglog_pdf(self, Var: xp.ndarray) -> xp.ndarray:
        assert Var.shape == (self.N, self.D)

        hess_diag = xp.zeros_like(Var, dtype=xp.float64)

        for edge in self.list_edges:
            # val = 1 / xp.sqrt((Var[edge[1]] - Var[edge[0]]) ** 2 + self.eps) - (
            #     Var[edge[1]] - Var[edge[0]]
            # ) ** 2 * ((Var[edge[1]] - Var[edge[0]]) ** 2 + self.eps) ** (
            #     -3 / 2
            # )  # (D,)
            delta_var = Var[edge[1]] - Var[edge[0]]
            val = 2 * self.eps / (delta_var**2 + self.eps) ** (3 / 2)
            hess_diag[edge[0], :] += val
            hess_diag[edge[1], :] += val

        hess_diag = hess_diag * self.weights[None, :]
        return hess_diag
