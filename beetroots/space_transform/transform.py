r"""Contains a class that defines the transition between the sampling scale and user friendly / interpretable scale
"""
# from sklearn.preprocessing import StandardScaler
from typing import List, Optional

import numba
import numpy as np

from beetroots.space_transform.abstract_transform import Scaler


@numba.njit()
def _from_scaled_to_lin(
    Theta_scaled: np.ndarray,
    mean_: np.ndarray,
    std_: np.ndarray,
    D: int,
    D_no_kappa: int,
    LOG_10: float,
    list_is_log: List[bool],
) -> np.ndarray:
    # theta : rescale log and go back to linear scale
    Theta_linscale = np.zeros_like(Theta_scaled)

    # kappa : go back to linear scale
    if list_is_log[0]:
        Theta_linscale[:, 0] = np.exp(Theta_scaled[:, 0])
    else:
        Theta_linscale[:, 0] = Theta_scaled[:, 0] * 1

    # other params
    for d in range(1, D):
        rescaled = std_[d - 1] * Theta_scaled[:, d] + mean_[d - 1]
        if list_is_log[d]:
            Theta_linscale[:, d] = np.exp(LOG_10 * rescaled)
        else:
            Theta_linscale[:, d] = rescaled * 1

    return Theta_linscale


@numba.njit()
def _from_lin_to_scaled(
    Theta_linscale: np.ndarray,
    mean_: np.ndarray,
    std_: np.ndarray,
    D: int,
    D_no_kappa: int,
    list_is_log: List[bool],
) -> np.ndarray:
    # theta
    Theta_scaled = np.zeros_like(Theta_linscale)

    # kappa
    if list_is_log[0]:
        Theta_scaled[:, 0] = np.log(Theta_linscale[:, 0])
    else:
        Theta_scaled[:, 0] = Theta_linscale[:, 0] * 1

    # other params
    for d in range(1, D):
        if list_is_log[d]:
            scaled = np.log10(Theta_linscale[:, d])
        else:
            scaled = Theta_linscale[:, d] * 1

        Theta_scaled[:, d] = (scaled - mean_[d - 1]) / std_[d - 1]

    return Theta_scaled


class MyScaler(Scaler):
    r"""Defines the scale used during sampling and the transforms to navigate from one scale to the other"""

    __slots__ = ("D", "D_no_kappa", "mean_", "std_")
    LOG_10 = np.log(10.0)

    def __init__(
        self,
        Theta_grid_lin: Optional[np.ndarray] = None,
        D_no_kappa: Optional[int] = None,
        mean_=None,
        std_=None,
        list_is_log: Optional[List[bool]] = None,
    ):
        r"""

        Parameters
        ----------
        Theta_grid_lin : np.ndarray of shape (-1, D)
            grid of simulations
        D_no_kappa : int
            number of physical parameters that require a standard scaler
        """
        if mean_ is not None and std_ is not None and list_is_log is not None:
            self.D_no_kappa = mean_.size * 1
            r"""int: number of physical parameters that require a standard scaler"""
            self.D = self.D_no_kappa + 1
            r"""int: total number of physical parameters that require a standard scaler, including the scaling factor :math:`\kappa`"""
            self.mean_ = mean_
            r"""np.ndarray of shape (D,): mean of the D components :math:`\theta_d`, used in the data normalization"""
            self.std_ = std_
            r"""np.ndarray of shape (D,): standard deviation of the D components :math:`\theta_d`, used in the data normalization"""

            assert len(list_is_log) == self.D, f"{self.D}, {len(list_is_log)}"
            self.list_is_log = list_is_log

        else:
            self.D = Theta_grid_lin.shape[1]
            self.D_no_kappa = D_no_kappa if D_no_kappa is not None else self.D
            assert self.D_no_kappa <= self.D

            assert isinstance(list_is_log, list) and len(list_is_log) == self.D
            self.list_is_log = list_is_log

            raise NotImplementedError()

            # TODO : correct this condition
            log10_grid_theta = np.log10(Theta_grid_lin[:, (self.D - self.D_no_kappa) :])
            self.mean_ = log10_grid_theta.mean(axis=0)  # (D,)
            self.std_ = log10_grid_theta.std(axis=0)  # (D,)

        assert self.mean_.shape == (
            self.D_no_kappa,
        ), f"{self.D_no_kappa}, {self.mean_.shape}"
        assert self.std_.shape == (
            self.D_no_kappa,
        ), f"{self.D_no_kappa}, {self.std_.shape}"

    def from_scaled_to_lin(self, Theta_scaled: np.ndarray) -> np.ndarray:
        assert len(Theta_scaled.shape) == 2, Theta_scaled.shape
        assert Theta_scaled.shape[1] == self.D, Theta_scaled.shape

        Theta_linscale = _from_scaled_to_lin(
            Theta_scaled,
            self.mean_,
            self.std_,
            self.D,
            self.D_no_kappa,
            self.LOG_10,
            self.list_is_log,
        )
        return Theta_linscale

    def from_lin_to_scaled(self, Theta_linscale: np.ndarray) -> np.ndarray:
        assert len(Theta_linscale.shape) == 2, Theta_linscale.shape
        assert Theta_linscale.shape[1] == self.D, Theta_linscale.shape
        Theta_scaled = _from_lin_to_scaled(
            Theta_linscale,
            self.mean_,
            self.std_,
            self.D,
            self.D_no_kappa,
            self.list_is_log,
        )
        return Theta_scaled
