from typing import List

import numba
import numpy as np

from beetroots.modelling.priors.abstract_prior import PriorProbaDistribution


@numba.jit("float64[:](float64[:,:], float64[:], float64[:], float64)", nopython=True)
def penalty(
    Theta: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    indicator_margin_scale: float,
) -> np.ndarray:
    D = lower_bounds.size

    neglog_p = np.zeros((D,))
    for d in range(D):
        neglog_p[d] = np.sum(
            np.where(
                Theta[:, d] > upper_bounds[d],
                (Theta[:, d] - upper_bounds[d]) / indicator_margin_scale,
                np.where(
                    Theta[:, d] < lower_bounds[d],
                    (lower_bounds[d] - Theta[:, d]) / indicator_margin_scale,
                    0,
                ),
            )
            ** 4
        )
    return neglog_p  # (D,)


@numba.njit()
def penalty_one_pix(
    Theta: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    indicator_margin_scale: float,
) -> np.ndarray:
    neglog_p_full = (
        np.where(
            Theta > np.expand_dims(upper_bounds, 0),
            (Theta - np.expand_dims(upper_bounds, 0)) / indicator_margin_scale,
            np.where(
                Theta < np.expand_dims(lower_bounds, 0),
                (np.expand_dims(lower_bounds, 0) - Theta) / indicator_margin_scale,
                np.zeros_like(Theta),
            ),
        )
        ** 4
    )  # (N_candidates, D)
    return np.sum(neglog_p_full, axis=1)  # (N_candidates,)


@numba.jit("float64[:,:](float64[:,:], float64[:], float64[:], float64)", nopython=True)
def gradient_penalty(
    Theta: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    indicator_margin_scale: float,
) -> np.ndarray:
    D = lower_bounds.size

    g = np.zeros_like(Theta)
    for d in range(D):
        g[:, d] = (
            4
            / indicator_margin_scale**4
            * np.where(
                Theta[:, d] > upper_bounds[d],
                (Theta[:, d] - upper_bounds[d]),
                np.where(
                    Theta[:, d] < lower_bounds[d],
                    (-lower_bounds[d] + Theta[:, d]),
                    0,
                ),
            )
            ** 3
        )
    return g


@numba.jit("float64[:,:](float64[:,:], float64[:], float64[:], float64)", nopython=True)
def hess_diag_penalty(
    Theta: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    indicator_margin_scale: float,
) -> np.ndarray:
    D = lower_bounds.size

    hess_diag = np.zeros_like(Theta)
    for d in range(D):
        hess_diag[:, d] = (
            12
            / indicator_margin_scale**4
            * np.where(
                Theta[:, d] > upper_bounds[d],
                (Theta[:, d] - upper_bounds[d]),
                np.where(
                    Theta[:, d] < lower_bounds[d],
                    (-lower_bounds[d] + Theta[:, d]),
                    0,
                ),
            )
            ** 2
        )
    return hess_diag  # (N, D)


class SmoothIndicatorPrior(PriorProbaDistribution):
    r"""This prior encodes validity intervals :math:`[\underline{\theta}_{d}, \overline{\theta}_{d}]]` the physical parameters :math:`\theta_{nd}` (for :math:`d \in [\![1, D]\!]`).
    The negative log of this prior is

    .. math::

        \forall n, d, \quad \iota^{\Delta}_{[\underline{\theta}_{d}, \overline{\theta}_{d}]}(\theta_{n,d}) = \begin{cases}
        0 \quad \text{ if } \theta_{n,d} \in [\underline{\theta}_{d}, \overline{\theta}_{d}]\\
        \left(\frac{\theta_{n,d} - \underline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} < \underline{\theta}_{d}\\
        \left(\frac{\theta_{n,d} - \overline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} > \overline{\theta}_{d}
        \end{cases}

    with :math:`\Delta > 0` a margin scaling parameter.
    """

    __slots__ = (
        "D",
        "N",
        "indicator_margin_scale",
        "lower_bounds",
        "upper_bounds",
    )

    def __init__(
        self,
        D: int,
        N: int,
        indicator_margin_scale: float,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        list_idx_sampling: List[int],
    ) -> None:
        super().__init__(D, N)
        self.indicator_margin_scale = indicator_margin_scale
        r"""float: scaling parameter :math:`\Delta`"""

        self.lower_bounds_full = lower_bounds
        r"""np.ndarray: validity interval lower bounds of the full set of D physical parameters"""
        self.upper_bounds_full = upper_bounds
        r"""np.ndarray: validity interval upper bounds of the full set of D physical parameters"""

        self.lower_bounds = lower_bounds[list_idx_sampling]
        r"""np.ndarray: validity interval lower bounds of the set of sampled physical parameters"""
        self.upper_bounds = upper_bounds[list_idx_sampling]
        r"""np.ndarray: validity interval upper bounds of the set of sampled physical parameters"""

        assert (
            self.lower_bounds.size == self.D
        ), f"should be {self.D}, is {self.lower_bounds.size}"
        assert (
            self.upper_bounds.size == self.D
        ), f"should be {self.D}, is {self.upper_bounds.size}"

        return

    def neglog_pdf(self, Theta: np.ndarray, pixelwise: bool = False) -> np.ndarray:
        r"""compute the negative log of the prior that approximates the indicator function

        .. math::

            \forall n, d, \quad \iota^{\Delta}_{[\underline{\theta}_{d}, \overline{\theta}_{d}]}(\theta_{n,d}) = \begin{cases}
            0 \quad \text{ if } \theta_{n,d} \in [\underline{\theta}_{d}, \overline{\theta}_{d}]\\
            \left(\frac{\theta_{n,d} - \underline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} < \underline{\theta}_{d}\\
            \left(\frac{\theta_{n,d} - \overline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} > \overline{\theta}_{d}
            \end{cases}

        Parameters
        ----------
        Theta : np.array of shape (N, D)
            current iterate
        pixelwise : bool, optional
            wether to return an aggregated result per pixel (if True) or per map (if False), by default False

        Returns
        -------
        neglog_p : np.ndarray of shape (D,) or (N,)
            negative log of the smooth indicator prior pdf
        """
        assert len(Theta.shape) == 2 and Theta.shape[1] == self.D
        if pixelwise:
            neglog_p = penalty_one_pix(
                Theta,
                self.lower_bounds,
                self.upper_bounds,
                self.indicator_margin_scale,
            )  # (N,)
        else:
            neglog_p = penalty(
                Theta,
                self.lower_bounds,
                self.upper_bounds,
                self.indicator_margin_scale,
            )  # (D,)
        return neglog_p

    def neglog_pdf_one_pix(self, Theta: np.ndarray) -> np.ndarray:
        r"""compute the negative log of the prior that approximates the indicator function

        .. math::

            \forall n, d, \quad \iota^{\Delta}_{[\underline{\theta}_{d}, \overline{\theta}_{d}]}(\theta_{n,d}) = \begin{cases}
            0 \quad \text{ if } \theta_{n,d} \in [\underline{\theta}_{d}, \overline{\theta}_{d}]\\
            \left(\frac{\theta_{n,d} - \underline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} < \underline{\theta}_{d}\\
            \left(\frac{\theta_{n,d} - \overline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} > \overline{\theta}_{d}
            \end{cases}

        Parameters
        ----------
        Theta : np.array of shape (N_candidates, D)
            current iterate

        Returns
        -------
        neglog_p : numpy array of shape (N_candidates,)
            negative log of the smooth indicator prior per map
        """
        assert len(Theta.shape) == 2 and Theta.shape[1] == self.D
        neglog_p = penalty_one_pix(
            Theta,
            self.lower_bounds,
            self.upper_bounds,
            self.indicator_margin_scale,
        )  # (N_candidates,)
        return neglog_p

    def gradient_neglog_pdf(self, Theta: np.ndarray) -> np.ndarray:
        r"""gradient of the negative log pdf of the smooth indicator prior

        Parameters
        ----------
        Theta : np.array of shape (N, D)
            current iterate

        Returns
        -------
        g : np.array of shape (N, D)
            gradient
        """
        assert len(Theta.shape) == 2 and Theta.shape[1] == self.D
        grad_ = gradient_penalty(
            Theta,
            self.lower_bounds,
            self.upper_bounds,
            self.indicator_margin_scale,
        )  # (N, D)
        return grad_  # / (self.N * self.D)

    def hess_diag_neglog_pdf(self, Theta: np.ndarray) -> np.ndarray:
        r"""diagonal of the Hessian of the negative log pdf of the smooth indicator prior

        Parameters
        ----------
        Theta : np.array of shape (N, D)
            current iterate

        Returns
        -------
        hess_diag : np.array of shape (N, D)
            [description]
        """
        assert len(Theta.shape) == 2 and Theta.shape[1] == self.D
        hess_diag = hess_diag_penalty(
            Theta,
            self.lower_bounds,
            self.upper_bounds,
            self.indicator_margin_scale,
        )  # (N, D)
        return hess_diag  # / (self.N * self.D)
