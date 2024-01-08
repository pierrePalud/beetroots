"""Utils functions for censored likelihoods
"""
from typing import Union, overload

import numpy as np
from scipy.special import log_ndtr


@overload
def logpdf_normal(x: np.ndarray) -> np.ndarray:
    ...


@overload
def logpdf_normal(x: Union[float, int]) -> float:
    ...


def logpdf_normal(Theta: Union[np.ndarray, float, int]) -> Union[np.ndarray, float]:
    """log pdf of the standard gaussian distribution

    Parameters
    ----------
    Theta : np.ndarray
        points at which the function is to be evaluated in a vectorized way

    Returns
    -------
    np.ndarray
        log pdf of the standard gaussian distribution
    """
    return -(Theta**2) / 2 - 0.5 * np.log(2 * np.pi)


@overload
def norm_pdf_cdf_ratio(Theta: np.ndarray) -> np.ndarray:
    ...


@overload
def norm_pdf_cdf_ratio(Theta: Union[float, int]) -> float:
    ...


def norm_pdf_cdf_ratio(
    Theta: Union[np.ndarray, float, int]
) -> Union[np.ndarray, float]:
    r"""computes the ratio of the pdf and cdf of the standard gaussian distribution at a given point

    Parameters
    ----------
    Theta : float or np.array
        current point

    Returns
    -------
    float or np.array
        ratio of the pdf and cdf of the standard gaussian distribution (has the same shape as Theta, if Theta is a np.array)
    """
    return np.exp(logpdf_normal(Theta) - log_ndtr(Theta))
