"""Implementation of a full grid evaluation of the approximation parameters of the likelihood with both additive and multiplicative noises.
"""
import datetime
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import gaussian_kde, ks_1samp
from tqdm.auto import tqdm

# import beetroots.utils as utils
from beetroots.modelling.forward_maps.regression_poly import PolynomialApprox
from beetroots.space_transform.transform import MyScaler

SMALLER_SIZE = 12
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALLER_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


P_lambda = np.poly1d(np.array([-6.0, 15.0, -10.0, 0.0, 0.0, 1.0]))
SQRT_2_PI = np.sqrt(2 * np.pi)
LOG_10 = np.log(10)


@nb.jit(nopython=True)
def neg_log_pdf_add(y, m_a, s_a):
    nll_au = 0.5 * ((y - m_a) / s_a) ** 2 + np.log(SQRT_2_PI * s_a)
    return nll_au


@nb.jit(nopython=True)
def neg_log_pdf_mult(y, m_m, s_m):
    nll_mu = np.where(
        y > 0,
        0.5 * ((np.log(y) - m_m) / s_m) ** 2 + np.log(y * SQRT_2_PI * s_m),
        np.infty,
    )
    return nll_mu


def compute_lambda(a0, a1, f_Theta_true):
    """carefull: a0 and a1 are in log, base 10, and computations require natural base

    Parameters
    ----------
    a0 : float
        center of mixing interval
    a1 : float
        radius of mixing interval
    f_Theta_true : float
        value of true f(x)

    Returns
    -------
    lambda: float
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


def build_forward_model(
    D: int,
    D_no_kappa: int,
    deg: int,
    grid_name: str,
    ell: int,
):
    # load grid
    # grid_reg = utils.read_extraction_result_dat(
    #     f"{os.path.dirname(os.path.abspath(__file__))}/../../../data/grids/{grid_name}"
    # )
    # cols = list(grid_reg.columns)

    grid_reg["kappa"] = 1.0
    grid_reg = grid_reg[["kappa"] + cols]

    # define scaler and apply it
    scaler = MyScaler(X_grid_lin=grid_reg.iloc[:, :D].values, D_no_kappa=D_no_kappa)

    grid_reg.iloc[:, :D] = scaler.from_lin_to_scaled(grid_reg.iloc[:, :D].values)
    grid_reg = grid_reg.iloc[:, list(range(D)) + [D + ell]]

    L = len(grid_reg.columns) - D

    # fit forward model
    pdr_code = PolynomialApprox(grid_reg, D, D_no_kappa, L, deg)
    return pdr_code, grid_reg


def sample_theta(
    K: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    rng: np.random._generator.Generator,
) -> np.ndarray:
    r"""sample $\theta$ from Stratified MC in cube

    Parameters
    ----------
    K : int
        total number of samples per axis
    lower_bounds : np.ndarray of shape (D,)
        lower bounds of cube on $\theta$
    upper_bounds : np.ndarray of shape (D,)
        upper bounds of cube on $\theta$
    rng : np.random._generator.Generator
        random number generator

    Returns
    -------
    np.ndarray of shape (N_samples, D)
        $\theta$ samples
    """
    delta_upper_lower = np.array([upper_bounds[d] - lower_bounds[d] for d in range(D)])
    list_lower_bounds = [
        np.arange(K) / K * delta_upper_lower[d] + lower_bounds[d] for d in range(D)
    ]
    list_lower_bounds = np.meshgrid(*list_lower_bounds)
    list_lower_bounds = [
        lower_bounds_Theta_i.flatten() for lower_bounds_Theta_i in list_lower_bounds
    ]
    list_samples = []
    for d, lower_bounds_Theta_i in enumerate(list_lower_bounds):
        Vd = rng.uniform(
            low=lower_bounds_Theta_i,
            high=lower_bounds_Theta_i + delta_upper_lower[d] / K,
        )
        list_samples.append(Vd)
    x = np.vstack(list_samples).T
    return x


def plot_hist_log10_f_Theta(
    log10_f_Theta: np.ndarray, kde_log10_f_Theta, line: str, ell: int, output_path: str
) -> None:
    """plots histogram of log10(f(\theta))

    Parameters
    ----------
    log10_f_Theta : np.ndarray of shape (-1, 1)
        array of values of f(x) for considered line
    line : str
        name of the line
    ell : int
        index of the line
    output_path : str
        path of folder where the figure is to be saved
    """
    log10_f_Theta_low = log10_f_Theta.min()
    log10_f_Theta_high = log10_f_Theta.max()
    nbins = 200

    list_log10_f_Theta = np.linspace(log10_f_Theta_low, log10_f_Theta_high, nbins)

    pi_log10_f_Theta = kde_log10_f_Theta.pdf(list_log10_f_Theta)

    plt.figure(figsize=(8, 6))
    plt.title(f"line {ell} : {line}")
    plt.hist(
        log10_f_Theta,
        bins=nbins,
        range=(log10_f_Theta_low, log10_f_Theta_high),
        density=True,
        label="samples",
    )
    plt.plot(list_log10_f_Theta, pi_log10_f_Theta, "r-", label="KDE")
    plt.xlabel(r"$\log f_\ell (x)$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/hist_log_f_Theta_ell{ell}.PNG")
    plt.close()


def estimate_avg_dks(dict_input: dict) -> dict:
    a0 = dict_input["a0"]
    a1 = dict_input["a1"]
    # log10_f_Theta = dict_input["log10_f_Theta"]
    list_log10_f_grid = dict_input["list_log10_f_grid"]
    sigma_a = dict_input["sigma_a"]
    sigma_m = dict_input["sigma_m"]
    N_samples_y = dict_input["N_samples_y"]
    pdf_kde_log10_f_Theta = dict_input["pdf_kde_log10_f_Theta"]

    list_f_grid = 10**list_log10_f_grid

    list_dks = np.zeros_like(list_log10_f_grid)
    for i, f_Theta_true in enumerate(list_f_grid):
        samples_a = np.random.normal(scale=sigma_a, size=N_samples_y)
        samples_m = np.random.lognormal(sigma=sigma_m, size=N_samples_y)

        samples_true = samples_m * f_Theta_true + samples_a
        samples_true.sort()

        statistic, _ = ks_1samp(
            samples_true,
            evaluate_cdf,
            args=(a0, a1, f_Theta_true, sigma_a, sigma_m),
        )
        list_dks[i] += statistic

    # Dks = interp1d(list_log10_f_grid, list_statistics, bounds_error=True)
    # list_dks = Dks(log10_f_Theta)

    #! average of Dks weighted with pdf of log10_f_Theta
    #! integrated between min and max of log10_f_Theta
    avg_dist = simpson(
        y=pdf_kde_log10_f_Theta * list_dks,
        x=list_log10_f_grid,
    )

    # list_dks = np.zeros_like(log10_f_Theta)
    # for ell in range(L):
    #     list_dks[:, ell] = Dks(log10_f_Theta[:, ell])

    dict_output = {
        "a0": a0,
        "a1": a1,
        "list_dists": list_dks,
        "avg_dist": avg_dist,
    }

    return dict_output


if __name__ == "__main__":
    tps0 = time.time()

    # * Problem settings
    # noise variances and considered line
    sigma_a = 1.38715e-10
    sigma_m = np.log(1.1)
    ell = 9

    # * forward model definition
    grid_name = "PDR17G1E20_P_cte_grid.dat"
    D = 4
    D_no_kappa = 3
    deg = 6

    forward_model, grid_reg = build_forward_model(D, D_no_kappa, deg, grid_name, ell)
    list_lines = list(grid_reg.columns[D:])
    L = len(list_lines) * 1
    assert L == 1
    line = list_lines[0]

    # * define bounds on \theta
    lower_bound_kappa = np.log(0.1)
    upper_bound_kappa = np.log(10)

    lower_bounds = np.array(
        [lower_bound_kappa] + list(grid_reg.iloc[:, 1:D].min().values)
    )
    upper_bounds = np.array(
        [upper_bound_kappa] + list(grid_reg.iloc[:, 1:D].max().values)
    )

    # * number of samples
    K = 30
    N_samples_theta = K**D
    r"""to build pdf of $P(\theta_\ell)$"""

    log10_f_grid_size = 100  # 250
    r"""number of points in grid on $P(\theta_\ell)$"""

    N_samples_y = 250_000  # 250_000
    r"""number of samples on $y_\ell$"""

    # * definition of grid of parameters to optimize
    # parameter to optimize : a = (center of interval, radius of interval)
    # (radius of interval = interval size /2)
    N_grid_a0 = 100  # 100
    N_grid_a1 = 50  # 50

    #! log10_f0 = log10(ratio std of noises), ie value of f at which noise
    #! variances are equal
    var_eps_m = np.exp(sigma_m**2) * (np.exp(sigma_m**2) - 1)
    log10_f0 = 0.5 * (2 * np.log(sigma_a) - np.log(var_eps_m)) / LOG_10

    bounds_a0_low = log10_f0 - 2
    bounds_a0_high = log10_f0 + 8

    bounds_a1_low = 0.01
    bounds_a1_high = 2

    list_a0 = np.linspace(bounds_a0_low, bounds_a0_high, N_grid_a0)
    list_a1 = np.linspace(bounds_a1_low, bounds_a1_high, N_grid_a1)

    # * create output folder
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H")

    output_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    output_path = f"{output_path}/outputs/approx_optim_grid_{dt_string}"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # * save params of optimization
    optim_params = {
        "sigma_a": sigma_a,
        "sigma_m": sigma_m,
        "K": K,
        "N_samples_theta": N_samples_theta,
        "N_samples_y": N_samples_y,
        "log10_f_grid_size": log10_f_grid_size,
        "grid_a": {
            "a0": {"low": bounds_a0_low, "high": bounds_a0_high, "size": N_grid_a0},
            "a1": {"low": bounds_a1_low, "high": bounds_a1_high, "size": N_grid_a1},
        },
        "grid_name": grid_name,
        "ell": ell,
    }
    with open(f"{output_path}/optim_params.json", "w", encoding="utf-8") as f:
        json.dump(
            optim_params,
            f,
            ensure_ascii=False,
            indent=4,
        )
    print("params saved")

    # * sample X using Stratified MC and build distribution f(X)
    rng = np.random.default_rng(42)
    x = sample_theta(K, lower_bounds, upper_bounds, rng)

    # the division is to get log in base 10
    log10_f_Theta = forward_model.evaluate_log(x) / LOG_10
    assert log10_f_Theta.shape == (N_samples_theta, 1)
    log10_f_Theta = log10_f_Theta.flatten()

    log10_f_Theta_low = log10_f_Theta.min()
    log10_f_Theta_high = log10_f_Theta.max()

    list_log10_f_grid = np.linspace(
        log10_f_Theta_low, log10_f_Theta_high, log10_f_grid_size
    )

    kde_log10_f_Theta = gaussian_kde(log10_f_Theta)
    pdf_kde_log10_f_Theta = kde_log10_f_Theta.pdf(list_log10_f_grid)
    print("samples of x ready")

    plot_hist_log10_f_Theta(log10_f_Theta, kde_log10_f_Theta, line, ell, output_path)
    print("histograms of f(x) done")

    # * perform main computation: evaluate Kolmogorov distance on grid of f_Theta
    print("starting Kolmogorov Smirnov distances computations")
    list_params = [
        {
            "a0": a0,
            "a1": a1,
            "log10_f_Theta": log10_f_Theta,
            "list_log10_f_grid": list_log10_f_grid,
            "sigma_a": sigma_a,
            "sigma_m": sigma_m,
            "N_samples_y": N_samples_y,
            "pdf_kde_log10_f_Theta": pdf_kde_log10_f_Theta,
        }
        for a0 in list_a0
        for a1 in list_a1
    ]

    with ProcessPoolExecutor(max_workers=30, mp_context=mp.get_context("fork")) as p:
        list_results = list(
            tqdm(p.map(estimate_avg_dks, list_params), total=len(list_params))
        )
    print("Kolmogorov Smirnov distances computations done")

    # * save list_dists (in list_results) as table in csv
    list_dicts = []
    for dict_output in list_results:
        a0 = dict_output["a0"]
        a1 = dict_output["a1"]
        list_statistics = list(dict_output["list_dists"])
        dict_ = {"a0": a0, "a1": a1}
        for log10_f_Theta, statistic in zip(list_log10_f_grid, list_statistics):
            dict_[log10_f_Theta] = statistic

        list_dicts.append(dict_)

    df_statistics = pd.DataFrame(list_dicts)
    df_statistics.to_csv(f"{output_path}/results_statistics.csv")

    # * save values of avg dist
    list_dicts = []
    for dict_output in list_results:
        a0 = dict_output["a0"]
        a1 = dict_output["a1"]
        avg_dist = dict_output["avg_dist"]
        dict_ = {"a0": a0, "a1": a1, "avg_dist": avg_dist}
        list_dicts.append(dict_)

    df_avg_dist = pd.DataFrame(list_dicts)
    df_avg_dist.to_csv(f"{output_path}/results_avg_dist.csv")

    # * plot contourf of avg_dist in function of (a, alpha_f)
    AA0, AA1 = np.meshgrid(list_a0, list_a1)
    arr_avg_dist = np.zeros_like(AA0)

    for dict_output in list_results:
        a0 = dict_output["a0"]
        a1 = dict_output["a1"]
        avg_dist = dict_output["avg_dist"]

        idx_a0 = np.where(list_a0 == a0)[0][0]
        idx_a1 = np.where(list_a1 == a1)[0][0]
        # print(idx_a, N_a, idx_alpha_f, N_alpha_f)
        # print(arr_avg_dist.shape)
        arr_avg_dist[idx_a1, idx_a0] = avg_dist

    plt.figure(figsize=(8, 6))
    plt.title(f"neg log average KS distance on line {ell}")
    plt.contourf(AA0, AA1, -np.log10(arr_avg_dist), levels=200)
    plt.colorbar(format=lambda x, _: f"{x:.3f}")
    plt.xlabel(r"mixing interval center $(a_{\ell,1} + a_{\ell,0})/2$")
    plt.ylabel(r"mixing interval radius $(a_{\ell,1} - a_{\ell,0})/2$")
    plt.axvline(log10_f0, c="r", ls="--", label="equal variances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/final_log.PNG")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.title("average KS distance on marginals")
    plt.contourf(AA0, AA1, arr_avg_dist, levels=200)
    plt.colorbar(format=lambda x, _: f"{x:1.2e}")
    plt.xlabel(r"mixing interval center $(a_{\ell,1} + a_{\ell,0})/2$")
    plt.ylabel(r"mixing interval radius $(a_{\ell,1} - a_{\ell,0})/2$")
    plt.axvline(log10_f0, c="r", ls="--", label="equal variances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/final_lin.PNG")
    plt.close()

    duration = time.time() - tps0
    print(f"optimization of parameters finished. Total duration: {duration:.2f} s")
