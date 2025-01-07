r"""Contains a class that defines the transition between the sampling scale and user friendly / interpretable scale
"""
from typing import List

import numba
import numpy as np

from beetroots.space_transform.abstract_transform import Scaler


@numba.njit()
def _from_scaled_to_lin(
    Theta_scaled: np.ndarray,
    mean_: np.ndarray,
    std_: np.ndarray,
    list_is_log: List[bool],
    D: int,
    LOG_10: float,
) -> np.ndarray:
    Theta_linscale = np.zeros_like(Theta_scaled)

    for d in range(D):
        rescaled = std_[d] * Theta_scaled[:, d] + mean_[d]
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
    list_is_log: List[bool],
    D: int,
) -> np.ndarray:
    Theta_scaled = np.zeros_like(Theta_linscale)

    for d in range(D):
        if list_is_log[d]:
            scaled = np.log10(Theta_linscale[:, d])
        else:
            scaled = Theta_linscale[:, d] * 1

        Theta_scaled[:, d] = (scaled - mean_[d]) / std_[d]

    return Theta_scaled


class MyScaler(Scaler):
    r"""Defines the scale used during sampling and the transforms to navigate from one scale to the other. The transformation is a normalization (defined with a mean `mean_` and standard deviation `std_`) for each physical parameter, defined on the log10 scale or on the linear scale depending on `list_is_log`.

    .. note::

        If one of the physical parameters is the scaling factor :math:`\kappa`, its mean is set to 0 and its std to 1 / np.log(10), so that it is not normalized regardless to its sampling scale (log10 or linear).

        The std = 1 / np.log(10) for kappa yields a scaled validity interval that is about [-2.7, 2.7] (for a [0.1, 10] true validity interval), ie comparable to that of other normalized parameters.
    """

    __slots__ = ("D", "mean_", "std_", "list_is_log")
    LOG_10 = np.log(10.0)

    def __init__(
        self,
        mean_: np.ndarray,
        std_: np.ndarray,
        list_is_log: List[bool],
    ):
        assert mean_.shape == std_.shape
        assert mean_.size in [len(list_is_log), len(list_is_log) - 1]

        # if there is a kappa in the set of parameters
        # (if there is no kappa, then each parameter should have an associated
        # mean and std in the scaler)
        if mean_.size == len(list_is_log) - 1:
            # kappa: mean = 0 and std = 1 / np.log(10)
            mean_ = np.array([0.0] + list(mean_))
            std_ = np.array([1.0 / self.LOG_10] + list(std_))

        self.D = mean_.size
        r"""int: total number of physical parameters that require a standard scaler, including the scaling factor :math:`\kappa`"""

        self.mean_ = mean_
        r"""np.ndarray of shape (D,): mean of the D components :math:`\theta_d`, used in the data normalization"""

        self.std_ = std_
        r"""np.ndarray of shape (D,): standard deviation of the D components :math:`\theta_d`, used in the data normalization"""

        self.list_is_log = list_is_log
        r"""list of bool of length D: whether the normalization should be applied on the log10 scale or in the linear scale"""

    def from_scaled_to_lin(self, Theta_scaled: np.ndarray) -> np.ndarray:
        assert len(Theta_scaled.shape) == 2, Theta_scaled.shape
        assert Theta_scaled.shape[1] == self.D, Theta_scaled.shape

        Theta_linscale = _from_scaled_to_lin(
            Theta_scaled,
            self.mean_,
            self.std_,
            self.list_is_log,
            self.D,
            self.LOG_10,
        )
        return Theta_linscale

    def from_lin_to_scaled(self, Theta_linscale: np.ndarray) -> np.ndarray:
        assert len(Theta_linscale.shape) == 2, Theta_linscale.shape
        assert Theta_linscale.shape[1] == self.D, Theta_linscale.shape

        Theta_scaled = _from_lin_to_scaled(
            Theta_linscale,
            self.mean_,
            self.std_,
            self.list_is_log,
            self.D,
        )
        return Theta_scaled
