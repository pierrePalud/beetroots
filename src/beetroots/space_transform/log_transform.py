import numba as nb

try:
    import cupy as xp  # type: ignore

    decorator_nb = nb.cuda.jit
except ImportError:
    import numpy as xp

    decorator_nb = nb.njit

from beetroots.space_transform.abstract_transform import Scaler


@decorator_nb()
def scaled_to_lin(Var_scaled: xp.ndarray) -> xp.ndarray:
    return xp.exp(Var_scaled)


@decorator_nb()
def lin_to_scaled(Var_linscale: xp.ndarray) -> xp.ndarray:
    return xp.log(Var_linscale)


class LogScaler(Scaler):
    r"""Simple log transformation class"""

    def from_scaled_to_lin(self, Var_scaled: xp.ndarray) -> xp.ndarray:
        r"""applies the transformation from sampling scale to user friendly scale

        Parameters
        ----------
        Var_scaled : xp.ndarray of shape (-1, ...)
            array in sampling scale

        Returns
        -------
        xp.ndarray of shape (-1, ...)
            array in user friendly scale
        """
        return scaled_to_lin(Var_scaled)

    def from_lin_to_scaled(self, Var_linscale: xp.ndarray) -> xp.ndarray:
        r"""applies the transformation from user friendly scale to sampling scale

        Parameters
        ----------
        Var_linscale : xp.ndarray of shape (-1, ...)
            array in user friendly scale

        Returns
        -------
        xp.ndarray of shape (-1, ...)
            array in sampling scale
        """
        return lin_to_scaled(Var_linscale)
