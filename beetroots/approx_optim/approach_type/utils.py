import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import ks_1samp
from tqdm.auto import tqdm

P_lambda = np.poly1d(np.array([-6.0, 15.0, -10.0, 0.0, 0.0, 1.0]))
SQRT_2_PI = np.sqrt(2 * np.pi)
LOG_10 = np.log(10)


def neg_log_pdf_add(
    y: Union[float, np.ndarray],
    m_a: Union[float, np.ndarray],
    s_a: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """evaluates the negative log pdf of a Gaussian distribution

    Parameters
    ----------
    y : Union[float, np.ndarray]
        points at which the pdf is to be evaluated
    m_a : Union[float, np.ndarray]
        bias of the Gaussian distribution
    s_a : Union[float, np.ndarray]
        standard deviation of the Gaussian distribution

    Returns
    -------
    Union[float, np.ndarray]
        negative log pdf
    """
    return 0.5 * ((y - m_a) / s_a) ** 2 + np.log(SQRT_2_PI * s_a)


def neg_log_pdf_mult(y, m_m, s_m):
    nll_mu = np.where(
        y > 0,
        0.5 * ((np.log(y) - m_m) / s_m) ** 2 + np.log(y * SQRT_2_PI * s_m),
        np.infty,
    )
    return nll_mu


def compute_lambda(
    a0: float, a1: float, f_Theta_true: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""careful: a0 and a1 are in log, base 10, and computations require natural base

    Parameters
    ----------
    a0 : float
        center of mixing interval
    a1 : float
        radius of mixing interval
    f_Theta_true : Union[float, np.ndarray]
        value of true :math:`f(\theta)`

    Returns
    -------
    lambda: Union[float, np.ndarray]
        mixing parameter
    """
    log_fm1 = (a0 - a1) * LOG_10
    log_fp1 = (a0 + a1) * LOG_10

    lambda_ = np.where(
        np.log(f_Theta_true) <= log_fm1,
        1,
        np.where(
            np.log(f_Theta_true) >= log_fp1,
            0,
            P_lambda((np.log(f_Theta_true) - log_fm1) / (log_fp1 - log_fm1)),
        ),
    )
    return lambda_


def pdf_my_approx(y, lambda_, m_a, s_a, m_m, s_m, f_Theta_true):
    neg_log_pdf_add_arr = neg_log_pdf_add(y, +m_a + f_Theta_true, s_a)
    neg_log_pdf_mult_arr = np.nan_to_num(
        neg_log_pdf_mult(y, +m_m + np.log(f_Theta_true), s_m)
    )

    log_res = neg_log_pdf_add_arr * lambda_ + neg_log_pdf_mult_arr * (1 - lambda_)
    return np.exp(-log_res)


def evaluate_pdf(y, a0, a1, f_Theta_true, sigma_a, sigma_m):
    lambda_ = compute_lambda(a0, a1, f_Theta_true)

    m_a = (np.exp(sigma_m**2 / 2) - 1) * f_Theta_true
    s_a = np.sqrt(
        f_Theta_true**2 * np.exp(sigma_m**2) * (np.exp(sigma_m**2) - 1)
        + sigma_a**2
    )

    combination = (sigma_a / f_Theta_true) * np.exp(-(sigma_m**2))
    m_m = -0.5 * np.log(1 + combination**2)
    s_m = np.sqrt(sigma_m**2 - 2 * m_m)

    my_pdf = pdf_my_approx(y, lambda_, m_a, s_a, m_m, s_m, f_Theta_true)
    return my_pdf


def evaluate_cdf(y, a0, a1, f_Theta_true, sigma_a, sigma_m):
    idx = np.argsort(y)
    y = y[idx]
    dy = np.diff(y, prepend=0)
    dy[0] = 0

    my_pdf = evaluate_pdf(y, a0, a1, f_Theta_true, sigma_a, sigma_m)
    my_pdf = my_pdf[idx]

    my_cdf = (my_pdf * dy).cumsum()
    my_cdf /= my_cdf[-1]  # normalize
    return my_cdf


def _estimate_one_dks(dict_input: Dict):
    a0 = dict_input["a0"]
    a1 = dict_input["a1"]
    i = dict_input["i"]
    f_Theta_true = dict_input["f_Theta_true"]
    sigma_a = dict_input["sigma_a"]
    sigma_m = dict_input["sigma_m"]
    N_samples_y = dict_input["N_samples_y"]

    samples_a = np.random.normal(scale=sigma_a, size=N_samples_y)
    samples_m = np.random.lognormal(sigma=sigma_m, size=N_samples_y)

    samples_true = samples_m * f_Theta_true + samples_a
    samples_true.sort()

    dks, _ = ks_1samp(
        samples_true,
        evaluate_cdf,
        args=(a0, a1, f_Theta_true, sigma_a, sigma_m),
    )
    dict_output = {"i": i, "dks": dks}
    return dict_output


def estimate_avg_dks_full_bo(
    a0: float,
    a1: float,
    list_log10_f_grid: np.ndarray,
    pdf_kde_log10_f_Theta: np.ndarray,
    sigma_a: float,
    sigma_m: float,
    N_samples_y: int,
    max_workers: int,
) -> float:
    list_params = [
        {
            "a0": a0,
            "a1": a1,
            "i": i,
            "f_Theta_true": f_Theta_true,
            "sigma_a": sigma_a,
            "sigma_m": sigma_m,
            "N_samples_y": N_samples_y,
        }
        for i, f_Theta_true in enumerate(np.exp(list_log10_f_grid * np.log(10.0)))
    ]
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp.get_context("fork")
    ) as p:
        list_results = list(
            tqdm(p.map(_estimate_one_dks, list_params), total=len(list_params))
        )

    df_results = pd.DataFrame.from_records(list_results)
    df_results = df_results.sort_values("i")
    list_dks = df_results["dks"].values
    assert len(df_results) == len(list_params)
    assert list_dks.shape == pdf_kde_log10_f_Theta.shape

    # ! average of Dks weighted with pdf of log10_f_Theta
    # ! integrated between min and max of log10_f_Theta
    avg_dist = simpson(
        y=pdf_kde_log10_f_Theta * list_dks,
        x=list_log10_f_grid,
    )
    # print(avg_dist)

    return -np.log10(avg_dist)
