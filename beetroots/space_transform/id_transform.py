r"""Contains a mock class for the scaler object, that does not transform the arrays it receives
"""
import numpy as np

from beetroots.space_transform.abstract_transform import Scaler


class IdScaler(Scaler):

    __slots__ = ()

    def __init__(self):
        pass

    def from_scaled_to_lin(self, Theta_scaled: np.ndarray) -> np.ndarray:
        return Theta_scaled

    def from_lin_to_scaled(self, Theta_linscale: np.ndarray) -> np.ndarray:
        return Theta_linscale
