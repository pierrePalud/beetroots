import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, simpson  # cumulative_simpson
from scipy.stats import ks_1samp, lognorm, norm
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
        0.5 * ((np.log(np.maximum(y, 1e-100)) - m_m) / s_m) ** 2
        + np.log(np.maximum(y, 1e-100) * SQRT_2_PI * s_m),
        np.inf,
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


def neg_log_pdf_my_approx(y, lambda_, m_a, s_a, m_m, s_m, f_Theta_true):
    neg_log_pdf_add_arr = neg_log_pdf_add(y, +m_a + f_Theta_true, s_a)
    neg_log_pdf_mult_arr = np.nan_to_num(
        neg_log_pdf_mult(y, +m_m + np.log(f_Theta_true), s_m)
    )

    log_res = neg_log_pdf_add_arr * lambda_ + neg_log_pdf_mult_arr * (1 - lambda_)
    return log_res


def params_moment_matching(sigma_a, sigma_m, f_Theta_true):
    # Moment matching for additive likelihood
    m_a = 0.0
    s_a = np.sqrt((np.exp(sigma_m**2) - 1) * f_Theta_true**2 + sigma_a**2)

    # Moment matching for multiplicative likelihood
    log_combination = np.log(sigma_a) - np.log(f_Theta_true) - sigma_m**2 / 2
    exp_m2_log_combin = np.exp(-2 * log_combination)
    m_m = -0.5 * (sigma_m**2 + np.log(1 + 1 / exp_m2_log_combin))
    s_m = np.sqrt(-2 * m_m)

    return m_a, s_a, m_m, s_m


def evaluate_cdf(
    y_censored,
    y_uncensored,
    f_Theta_true,
    sigma_a,
    sigma_m,
    omega=None,
    likelihood="my_approx",
    a0=None,
    a1=None,
    N_samples_u=None,
) -> np.ndarray:
    idx = np.argsort(y_uncensored)
    y_uncensored = y_uncensored[idx]
    # dy = np.diff(y_uncensored, prepend=0)
    # dy[0] = 0

    if likelihood == "my_approx":
        assert a0 is not None and a1 is not None, (
            "a0 and a1 must be provided for 'my_approx' likelihood"
        )
        lambda_ = compute_lambda(a0, a1, f_Theta_true)
        m_a, s_a, m_m, s_m = params_moment_matching(sigma_a, sigma_m, f_Theta_true)
        my_pdf = neg_log_pdf_my_approx(
            y_uncensored, lambda_, m_a, s_a, m_m, s_m, f_Theta_true
        )
        my_pdf = np.exp(-my_pdf)
    elif likelihood == "additive":
        m_a, s_a, _, _ = params_moment_matching(sigma_a, sigma_m, f_Theta_true)
        my_pdf = neg_log_pdf_add(y_uncensored, m_a + f_Theta_true, s_a)
        my_pdf = np.exp(-my_pdf)
        my_cdf = norm.cdf((y_uncensored - f_Theta_true - m_a) / s_a)
        # if omega is not None:
        #     idx_last_censored = np.searchsorted(y_uncensored, omega, side="right")
        #     my_cdf[:idx_last_censored] = my_cdf[
        #         idx_last_censored - 1
        #     ]  # Won't change anything if no censored data
        # return my_cdf
    elif likelihood == "multiplicative":
        _, _, m_m, s_m = params_moment_matching(sigma_a, sigma_m, f_Theta_true)
        my_pdf = neg_log_pdf_mult(y_uncensored, m_m + np.log(f_Theta_true), s_m)
        my_pdf = np.exp(-my_pdf)
        # my_cdf = lognorm.cdf(
        #     y_uncensored,
        #     s=s_m,
        #     scale=np.exp(m_m + np.log(f_Theta_true)),
        # )
        # if omega is not None:
        #     idx_last_censored = np.searchsorted(y_uncensored, omega, side="right")
        #     my_cdf[:idx_last_censored] = my_cdf[
        #         idx_last_censored - 1
        #     ]  # Won't change anything if no censored data
        # return my_cdf
    elif likelihood == "hierarchical":
        assert N_samples_u is not None, (
            "N_samples_u must be provided for 'hierarchical' likelihood"
        )
        # samples_u_MC_integration = f_Theta_true * np.random.lognormal(
        #     mean=-(sigma_m**2) / 2, sigma=sigma_m, size=(y.shape[0], N_samples_u)
        # )
        if f_Theta_true > sigma_a / np.sqrt(np.exp(sigma_m**2) - 1):
            samples_u_MC_integration_IS = y_uncensored[
                :, np.newaxis
            ] - np.random.normal(
                scale=2 * sigma_a, size=(y_uncensored.shape[0], N_samples_u)
            )
            neg_log_weights = -neg_log_pdf_add(
                y_uncensored[:, np.newaxis], samples_u_MC_integration_IS, 2 * sigma_a
            )
        else:
            s_m = sigma_m + np.log(1.1)
            samples_u_MC_integration_IS = f_Theta_true * np.random.lognormal(
                mean=-(s_m**2) / 2, sigma=s_m, size=(y_uncensored.shape[0], N_samples_u)
            )
            neg_log_weights = -neg_log_pdf_mult(
                samples_u_MC_integration_IS,
                -(s_m**2) / 2 + np.log(f_Theta_true),
                s_m,
            )
        neg_log_weights += neg_log_pdf_add(
            y_uncensored[:, np.newaxis], samples_u_MC_integration_IS, sigma_a
        )
        neg_log_weights += neg_log_pdf_mult(
            samples_u_MC_integration_IS,
            -(sigma_m**2) / 2 + np.log(f_Theta_true),
            sigma_m,
        )
        my_pdf = np.exp(-neg_log_weights).mean(axis=-1)
        # my_pdf = neg_log_pdf_add(y[:, np.newaxis], samples_u_MC_integration, sigma_a)
        # my_pdf = np.exp(-my_pdf).mean(axis=-1)  # Marginalization over U | \Theta

        # my_pdf = neg_log_pdf_add(y, samples_u, sigma_a)
        # my_pdf = np.exp(-my_pdf)
        # print(my_pdf.max(), my_pdf.min(), my_pdf.mean(), np.median(my_pdf))

    else:
        raise ValueError(f"Unknown likelihood type: {likelihood}")

    # my_cdf = (my_pdf * dy).cumsum()
    # my_cdf /= my_cdf[-1]  # normalize

    my_cdf = cumulative_trapezoid(my_pdf, x=y_uncensored, initial=0)
    my_cdf /= my_cdf[-1]  # normalize

    if omega is not None:
        idx_last_censored = np.searchsorted(y_uncensored, omega, side="right")
        my_cdf[:idx_last_censored] = my_cdf[
            idx_last_censored - 1
        ]  # Won't change anything if no censored data

    return my_cdf


def _estimate_one_dks(dict_input: Dict) -> Dict:
    a0 = dict_input["a0"]
    a1 = dict_input["a1"]
    i = dict_input["i"]
    f_Theta_true = dict_input["f_Theta_true"]
    sigma_a = dict_input["sigma_a"]
    sigma_m = dict_input["sigma_m"]
    omega = dict_input["omega"]
    N_samples_y = dict_input["N_samples_y"]
    likelihood = dict_input["likelihood"]
    N_samples_u = dict_input["N_samples_u"]
    generative_noise_model = dict_input["generative_noise_model"]

    samples_a = np.random.normal(scale=sigma_a, size=N_samples_y)
    samples_m = np.random.lognormal(
        sigma=sigma_m, mean=-(sigma_m**2) / 2, size=N_samples_y
    )
    samples_uncensored = f_Theta_true * 1.0
    if generative_noise_model in ["multiplicative", "both"]:
        samples_uncensored *= samples_m
    if generative_noise_model in ["additive", "both"]:
        samples_uncensored += samples_a
    if generative_noise_model not in ["additive", "multiplicative", "both"]:
        raise ValueError(f"Unknown generative noise model: {generative_noise_model}")

    samples_uncensored.sort()
    if omega is not None:
        samples_censored = np.maximum(samples_uncensored, omega)
    else:
        samples_censored = samples_uncensored * 1.0

    if omega is None:
        dks, _ = ks_1samp(
            samples_censored,
            evaluate_cdf,
            args=(
                samples_uncensored,
                f_Theta_true,
                sigma_a,
                sigma_m,
                omega,
                likelihood,
                a0,
                a1,
                N_samples_u,
            ),
        )
    else:  # Custom computation of empirical CDF with censored data. We can have duplicate values so defining empirical CDF using np.arange is not possible
        # Based on scipy.stats.ks_1samp source code: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_1samp.html
        samples_censored = np.sort(samples_censored)
        ranks = np.searchsorted(samples_censored, samples_censored, side="right")
        empirical_cdf_plus = ranks / len(samples_censored)
        empirical_cdf_minus = (ranks - 1) / len(samples_censored)

        model_cdf = evaluate_cdf(
            samples_censored,
            samples_uncensored,
            f_Theta_true,
            sigma_a,
            sigma_m,
            omega,
            likelihood,
            a0,
            a1,
            N_samples_u,
        )
        dks_plus = (empirical_cdf_plus - model_cdf).max()
        dks_minus = (model_cdf - empirical_cdf_minus).max()
        dks = max(dks_plus, dks_minus)

    dict_output = {"i": i, "dks": dks}
    return dict_output


def estimate_avg_dks_full_bo(
    list_log10_f_grid: np.ndarray,
    pdf_kde_log10_f_Theta: np.ndarray,
    sigma_a: float,
    sigma_m: float,
    N_samples_y: int,
    max_workers: int,
    omega=None,
    likelihood: str = "my_approx",
    a0: float = None,
    a1: float = None,
    N_samples_u: int = None,  # Only if hierarchical likelihood is used
    generative_noise_model: str = "both",
    return_list_dks: bool = False,
) -> float:
    if generative_noise_model not in ["additive", "multiplicative", "both"]:
        warnings.warn(
            f"Unknown generative noise model: {generative_noise_model}. Setting to 'both'"
        )
        generative_noise_model = "both"

    list_params = [
        {
            "a0": a0,
            "a1": a1,
            "i": i,
            "f_Theta_true": f_Theta_true,
            "sigma_a": sigma_a,
            "sigma_m": sigma_m,
            "omega": omega,
            "N_samples_y": N_samples_y,
            "likelihood": likelihood,
            "N_samples_u": N_samples_u,
            "generative_noise_model": generative_noise_model,
        }
        for i, f_Theta_true in enumerate(np.exp(list_log10_f_grid * np.log(10.0)))
    ]

    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp.get_context("fork")
    ) as p:
        list_results = list(
            tqdm(
                p.map(_estimate_one_dks, list_params),
                total=len(list_params),
                leave=False,
            )
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
    if return_list_dks:
        return -np.log10(avg_dist), list_dks
    else:
        return -np.log10(avg_dist)
