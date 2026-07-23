"""Utils functions for censored likelihoods"""

from typing import Union, overload

import numpy as np
from scipy.special import log_ndtr


@overload
def logpdf_normal(x: np.ndarray) -> np.ndarray: ...


@overload
def logpdf_normal(x: Union[float, int]) -> float: ...


def logpdf_normal(Var: Union[np.ndarray, float, int]) -> Union[np.ndarray, float]:
    """log pdf of the standard gaussian distribution

    Parameters
    ----------
    Var : np.ndarray
        points at which the function is to be evaluated in a vectorized way

    Returns
    -------
    np.ndarray
        log pdf of the standard gaussian distribution
    """
    return -(Var**2) / 2 - 0.5 * np.log(2 * np.pi)


@overload
def norm_pdf_cdf_ratio(Var: np.ndarray) -> np.ndarray: ...


@overload
def norm_pdf_cdf_ratio(Var: Union[float, int]) -> float: ...


def norm_pdf_cdf_ratio(Var: Union[np.ndarray, float, int]) -> Union[np.ndarray, float]:
    r"""computes the ratio of the pdf and cdf of the standard gaussian distribution at a given point

    Parameters
    ----------
    Var : float or np.array
        current point

    Returns
    -------
    float or np.array
        ratio of the pdf and cdf of the standard gaussian distribution (has the same shape as Var, if Var is a np.array)
    """
    return np.exp(logpdf_normal(Var) - log_ndtr(Var))
