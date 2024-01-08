import abc

import numpy as np


class Scaler(abc.ABC):
    r"""class that defines a transition between the sampling space and a more user friendly / interpretable space"""

    @abc.abstractmethod
    def from_scaled_to_lin(self, Theta_scaled: np.ndarray) -> np.ndarray:
        r"""applies the transformation from sampling scale to user friendly scale

        Parameters
        ----------
        Theta_scaled : np.ndarray of shape (-1, D)
            array in sampling scale

        Returns
        -------
        np.ndarray of shape (-1, D)
            array in user friendly scale
        """
        pass

    @abc.abstractmethod
    def from_lin_to_scaled(self, Theta_linscale: np.ndarray) -> np.ndarray:
        r"""applies the transformation from user friendly scale to sampling scale

        Parameters
        ----------
        Theta_linscale : np.ndarray of shape (-1, D)
            array in user friendly scale

        Returns
        -------
        np.ndarray of shape (-1, D)
            array in sampling scale
        """
        pass
