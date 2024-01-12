from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy import interpolate


def plot_1D_hist(
    list_Theta_lin_seed: np.ndarray,
    n: Union[int, None],
    d: int,
    folder_path: str,
    title: str,
    lower_bounds_lin: Optional[np.ndarray],
    upper_bounds_lin: Optional[np.ndarray],
    seed: Optional[int] = None,
    estimator: Optional[float] = None,
    IC_low: Optional[float] = None,
    IC_high: Optional[float] = None,
    true_val: Optional[float] = None,
):
    if len(list_Theta_lin_seed.shape) == 3:
        list_Theta_lin_nd = list_Theta_lin_seed[:, n, d] * 1
    else:
        list_Theta_lin_nd = list_Theta_lin_seed * 1

    if lower_bounds_lin is None:
        lower_bounds_lin = np.min(list_Theta_lin_seed) * np.ones((10,))
    if upper_bounds_lin is None:
        upper_bounds_lin = np.max(list_Theta_lin_seed) * np.ones((10,))

    plt.figure(figsize=(8, 6))
    plt.title(title)

    assert lower_bounds_lin is not None
    if list_Theta_lin_seed.min() > 0 and lower_bounds_lin.min() > 0:
        plt.hist(np.log10(list_Theta_lin_nd), bins=100, label="samples")
    else:
        plt.hist(list_Theta_lin_nd, bins=100, label="samples")

    if estimator is not None:
        if list_Theta_lin_seed.min() > 0 and lower_bounds_lin.min() > 0:
            plt.axvline(np.log10(estimator), c="orange", ls="-", label="mean")
        else:
            plt.axvline(estimator, c="orange", ls="-", label="mean")

    if np.all([IC_low is not None, IC_high is not None]):
        assert IC_low is not None and IC_high is not None
        if list_Theta_lin_seed.min() > 0 and lower_bounds_lin.min() > 0:
            plt.axvline(np.log10(IC_low), c="k", ls="--", label="CI")
            plt.axvline(np.log10(IC_high), c="k", ls="--")
        else:
            plt.axvline(IC_low, c="k", ls="--", label="CI")
            plt.axvline(IC_high, c="k", ls="--")

    if true_val is not None:
        if list_Theta_lin_seed.min() > 0 and lower_bounds_lin.min() > 0:
            plt.axvline(np.log10(true_val), c="red", ls="--", label="true")
        else:
            plt.axvline(true_val, c="red", ls="--", label="true")

    plt.grid()
    plt.legend()
    # plt.tight_layout()
    if seed is not None:
        if n is not None:
            plt.savefig(f"{folder_path}/hist_n{n}_d{d}_seed{seed}.PNG")
        else:
            plt.savefig(f"{folder_path}/hist_d{d}_seed{seed}.PNG")
    else:
        if n is not None:
            plt.savefig(f"{folder_path}/hist_n{n}_d{d}_overall.PNG")
        else:
            plt.savefig(f"{folder_path}/hist_d{d}_overall.PNG")

    plt.close()
    return


def plot_1D_chain(
    list_Theta_lin_nd: np.ndarray,
    n: Optional[int],
    d: int,
    folder_path: str,
    title: str,
    lower_bounds_lin: Optional[np.ndarray],
    upper_bounds_lin: Optional[np.ndarray],
    N_MCMC: int,
    T_MC: int,
    T_BI: int,
    true_val: Optional[float] = None,
    is_u: bool = False,
) -> None:
    assert len(list_Theta_lin_nd.shape) == 1  # (N_MCMC * (T_MC - T_BI),)

    if lower_bounds_lin is None:
        lower_bounds_lin = np.min(list_Theta_lin_nd) * np.ones((10,))
    if upper_bounds_lin is None:
        upper_bounds_lin = np.max(list_Theta_lin_nd) * np.ones((10,))

    assert lower_bounds_lin is not None

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.plot(list_Theta_lin_nd, label="MC chain")

    for seed in range(1, N_MCMC):
        if seed == 1:
            plt.axvline(
                seed * (T_MC - T_BI),
                c="k",
                ls="-",
                label="new MC",
            )
        else:
            plt.axvline(seed * (T_MC - T_BI) - seed, c="k", ls="-")

    if true_val is not None:
        plt.axhline(true_val, c="r", ls="--", label="true value")

    if list_Theta_lin_nd.min() > 0 and lower_bounds_lin.min() > 0:
        plt.yscale("log")

    plt.grid()
    plt.legend()
    # plt.tight_layout()

    if is_u:
        filename = f"{folder_path}/mc_1D_n{n}_ell{d}.PNG"
    else:
        filename = f"{folder_path}/mc_1D_n{n}_d{d}.PNG"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_2D_chain(
    list_Theta_lin_seed: np.ndarray,
    n: int,
    d1: int,
    d2: int,
    model_name: str,
    seed: int,
    folder_path: str,
    list_names: list,
    Theta_estimator: np.ndarray,
    Theta_true: Optional[np.ndarray] = None,
):
    # assert list_Theta_lin_seed.shape in [
    #     (self.T_MC - self.T_BI, self.N, self.D),
    #     (self.N_MCMC * (self.T_MC - self.T_BI), self.N, self.D),
    # ]
    freq = 5
    plt.figure(figsize=(8, 6))

    title = f"joint MC of ({list_names[d1]}, {list_names[d2]}) for pixel n={n}"
    plt.title(title)
    if list_Theta_lin_seed.min() > 0:
        plt.scatter(
            np.log10(list_Theta_lin_seed[::freq, n, d1]).flatten(),
            np.log10(list_Theta_lin_seed[::freq, n, d2]).flatten(),
            c=np.arange(list_Theta_lin_seed[::freq].shape[0]),
            s=10,
            label="samples",
        )
    else:
        plt.scatter(
            list_Theta_lin_seed[::freq, n, d1].flatten(),
            list_Theta_lin_seed[::freq, n, d2].flatten(),
            c=np.arange(list_Theta_lin_seed[::freq].shape[0]),
            s=10,
            label="samples",
        )
    if Theta_true is not None:
        plt.plot(
            np.log10([Theta_true[n, d1]]),
            np.log10([Theta_true[n, d2]]),
            "r+",
            ms=20,
            label="truth",
        )

    plt.colorbar()
    plt.legend()
    plt.grid()
    if list_Theta_lin_seed.min() > 0:
        plt.xlabel(r"$\log$ " + list_names[d1])
        plt.ylabel(r"$\log$ " + list_names[d2])
    else:
        plt.xlabel(list_names[d1])
        plt.ylabel(list_names[d2])
    # plt.tight_layout()
    plt.savefig(f"{folder_path}/mc_2D_n{n}_d1{d1}_d2{d2}_seed{seed}_chain.PNG")
    plt.close()


def plot_2D_hist(
    list_Theta_lin_seed: np.ndarray,
    n: int,
    d1: int,
    d2: int,
    model_name: str,
    folder_path: str,
    list_names: list,
    lower_bounds_lin: np.ndarray,
    upper_bounds_lin: np.ndarray,
    Theta_MMSE: np.ndarray,
    true_val: Optional[np.ndarray] = None,
    seed: Union[int, str] = "overall",
    point_challenger: Dict = {},
):
    # assert list_Theta_lin_seed.shape in [
    #     (self.T_MC - self.T_BI, self.N, self.D),
    #     (self.N_MCMC * (self.T_MC - self.T_BI), self.N, self.D),
    # ]
    def _set_edges():
        list_Theta_lin_seed_used = list_Theta_lin_seed * 1

        # x-axis
        if list_Theta_lin_seed[:, 0].min() > 0 and lower_bounds_lin[d1] > 0:
            bounds_theta = (
                np.log10(lower_bounds_lin[d1]) - 0.1,
                np.log10(upper_bounds_lin[d1]) + 0.1,
            )
            list_Theta_lin_seed_used[:, 0] = np.log10(list_Theta_lin_seed_used[:, 0])
            is_theta_log = True
        else:
            bounds_theta = (
                lower_bounds_lin[d1] - 0.1,
                upper_bounds_lin[d1] + 0.1,
            )
            is_theta_log = False

        # y-axis
        if list_Theta_lin_seed[:, 1].min() > 0 and lower_bounds_lin[d2] > 0:
            bounds_y = (
                np.log10(lower_bounds_lin[d2]) - 0.1,
                np.log10(upper_bounds_lin[d2]) + 0.1,
            )
            list_Theta_lin_seed_used[:, 1] = np.log10(list_Theta_lin_seed_used[:, 1])
            is_y_log = True
        else:
            bounds_y = (
                lower_bounds_lin[d2] - 0.1,
                upper_bounds_lin[d2] + 0.1,
            )
            is_y_log = False

        return list_Theta_lin_seed_used, bounds_theta, bounds_y, is_theta_log, is_y_log

    (
        list_Theta_lin_seed_used,
        bounds_theta,
        bounds_y,
        is_theta_log,
        is_y_log,
    ) = _set_edges()

    plt.figure(figsize=(8, 6))

    title = f"joint MC of ({list_names[d1]}, {list_names[d2]}) for pixel n={n}"
    plt.title(title)

    plt.hist2d(
        list_Theta_lin_seed_used[:, 0].flatten(),
        list_Theta_lin_seed_used[:, 1].flatten(),
        (100, 100),
        range=[[bounds_theta[0], bounds_theta[1]], [bounds_y[0], bounds_y[1]]],
        norm=colors.LogNorm(),
    )

    plt.plot(
        np.log10([Theta_MMSE[0]]) if is_theta_log else [Theta_MMSE[0]],
        np.log10([Theta_MMSE[1]]) if is_y_log else [Theta_MMSE[1]],
        "rx",
        ms=20,
        label="MMSE",
        markeredgewidth=3,
    )

    if len(point_challenger) > 0:
        x_chal = point_challenger["value"] * 1
        plt.plot(
            np.log10([x_chal[n, d1]]) if is_theta_log else [x_chal[n, d1]],
            np.log10([x_chal[n, d2]]) if is_y_log else [x_chal[n, d2]],
            "k+",
            ms=20,
            label=point_challenger["name"],
            markeredgewidth=3,
        )

    if true_val is not None:
        plt.plot(
            np.log10([true_val[0]]) if is_theta_log else [true_val[0]],
            np.log10([true_val[1]]) if is_y_log else [true_val[1]],
            "r+",
            ms=20,
            label="truth",
            markeredgewidth=3,
        )

    plt.legend(loc="best")

    # in case no point in acceptable set
    plt.colorbar()

    plt.grid()
    plt.xlabel(r"$\log$ " + list_names[d1] if is_theta_log else list_names[d1])
    plt.ylabel(r"$\log$ " + list_names[d2] if is_y_log else list_names[d2])
    # plt.tight_layout()

    if seed == "overall":
        filename = f"{folder_path}/hist2D_n{n}_d1{d1}_d2{d2}_overall_chain.PNG"
    else:
        filename = f"{folder_path}/hist2D_n{n}_d1{d1}_d2{d2}_seed{seed}"
        filename += "_chain.PNG"
    plt.savefig(filename)
    plt.close()


def plot_2D_proba_contours(
    list_Theta_lin_seed: np.ndarray,
    n: int,
    d1: int,
    d2: int,
    model_name: str,
    folder_path: str,
    list_names: list,
    lower_bounds_lin: np.ndarray,
    upper_bounds_lin: np.ndarray,
    Theta_MMSE: np.ndarray,
    true_val: Optional[np.ndarray] = None,
    seed: Union[int, str] = "overall",
    point_challenger: Dict = {},
):
    """plots the 2D contours of the 68% and 95% high probability regions

    inspired from https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution
    """
    # assert list_Theta_lin_seed.shape in [
    #     (self.T_MC - self.T_BI, self.N, self.D),
    #     (self.N_MCMC * (self.T_MC - self.T_BI), self.N, self.D),
    # ]
    def _set_edges_and_hist(n_per_axis: int):
        list_Theta_lin_seed_used = list_Theta_lin_seed * 1

        # x-axis
        if list_Theta_lin_seed[:, 0].min() > 0 and lower_bounds_lin[d1] > 0:
            x_edges = np.linspace(
                np.log10(lower_bounds_lin[d1]) - 0.1,
                np.log10(upper_bounds_lin[d1]) + 0.1,
                n_per_axis,
            )
            list_Theta_lin_seed_used[:, 0] = np.log10(list_Theta_lin_seed_used[:, 0])
            is_theta_log = True
        else:
            x_edges = np.linspace(
                lower_bounds_lin[d1] - 0.1,
                upper_bounds_lin[d1] + 0.1,
                n_per_axis,
            )
            is_theta_log = False

        # y-axis
        if list_Theta_lin_seed[:, 1].min() > 0 and lower_bounds_lin[d2] > 0:
            y_edges = np.linspace(
                np.log10(lower_bounds_lin[d2]) - 0.1,
                np.log10(upper_bounds_lin[d2]) + 0.1,
                n_per_axis,
            )
            list_Theta_lin_seed_used[:, 1] = np.log10(list_Theta_lin_seed_used[:, 1])
            is_y_log = True
        else:
            y_edges = np.linspace(
                lower_bounds_lin[d2] - 0.1,
                upper_bounds_lin[d2] + 0.1,
                n_per_axis,
            )
            is_y_log = False

        H, x_edges, y_edges = np.histogram2d(
            list_Theta_lin_seed_used[:, 0].flatten(),
            list_Theta_lin_seed_used[:, 1].flatten(),
            bins=(x_edges, y_edges),
        )
        return H, x_edges, y_edges, is_theta_log, is_y_log

    n_per_axis = 100

    #! Contour levels must be increasing, ie percentiles must be decreasing
    percentiles_arr = np.array([0.95, 0.68])

    H, x_edges, y_edges, is_theta_log, is_y_log = _set_edges_and_hist(n_per_axis)
    H /= H.sum()
    t = np.linspace(0, H.max(), 1_000)
    integral = ((H >= t[:, None, None]) * H).sum(axis=(1, 2))

    # an error comes if the histogram grid is not fine enough to compute
    # the contours
    # Hence we progressively increase the grid size per axis until the grid
    # gets fine enough.
    # this error occurs frequently when the Markov chain size is small and
    # when the high proba region is very small compared to the acceptable
    # solutions set.
    while (integral[-1] > percentiles_arr[-1]) and (n_per_axis < 5e2):
        n_per_axis *= 2  # refine histogram grid
        H, x_edges, y_edges, _, _ = _set_edges_and_hist(n_per_axis)
        H /= H.sum()
        t = np.linspace(0, H.max(), 1_000)
        integral = ((H >= t[:, None, None]) * H).sum(axis=(1, 2))

    f = interpolate.interp1d(integral, t)

    if integral[-1] > percentiles_arr[-1]:
        percentiles_arr = percentiles_arr[integral[-1] < percentiles_arr]

        msg = "Issue with proba contour plot for (n, d1, d2) = "
        msg += f"({n}, {d1}, {d2})"
        print(msg)

        if percentiles_arr.size == 0:
            return

    # define contours
    t_contours = f(percentiles_arr)

    plt.figure(figsize=(8, 6))

    title = f"High Probability Region of ({list_names[d1]}, {list_names[d2]})"
    title += f" for pixel n={n}"
    plt.title(title)

    for percentile, t_contour in zip(percentiles_arr, t_contours):
        plt.contour(
            H.T,
            t_contours,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            label=f"{100 * percentile} %",
        )

    if list_Theta_lin_seed.min() > 0 and lower_bounds_lin.min() > 0:
        plt.plot(
            np.log10([Theta_MMSE[0]]),
            np.log10([Theta_MMSE[1]]),
            "rx",
            ms=20,
            label="MMSE",
            markeredgewidth=3,
        )
    else:
        plt.plot(
            [Theta_MMSE[0]],
            [Theta_MMSE[1]],
            "rx",
            ms=20,
            label="MMSE",
            markeredgewidth=3,
        )

    if len(point_challenger) > 0:
        x_challenger = point_challenger["value"]
        if list_Theta_lin_seed.min() > 0 and lower_bounds_lin.min() > 0:
            plt.plot(
                np.log10([x_challenger[n, d1]]),
                np.log10([x_challenger[n, d2]]),
                "k+",
                ms=20,
                label=point_challenger["name"],
                markeredgewidth=3,
            )
        else:
            plt.plot(
                [x_challenger[n, d1]],
                [x_challenger[n, d2]],
                "k+",
                ms=20,
                label=point_challenger["name"],
                markeredgewidth=3,
            )

    if true_val is not None:
        if list_Theta_lin_seed.min() > 0 and lower_bounds_lin.min() > 0:
            plt.plot(
                np.log10([true_val[0]]),
                np.log10([true_val[1]]),
                "r+",
                ms=20,
                label="truth",
                markeredgewidth=3,
            )
        else:
            plt.plot(
                [true_val[0]],
                [true_val[1]],
                "r+",
                ms=20,
                label="truth",
                markeredgewidth=3,
            )

    plt.legend(loc="best")

    # in case no point in acceptable set
    # plt.colorbar()

    plt.grid()
    plt.xlabel(r"$\log$ " + list_names[d1] if is_theta_log else list_names[d1])
    plt.ylabel(r"$\log$ " + list_names[d2] if is_y_log else list_names[d2])
    # plt.tight_layout()

    if seed == "overall":
        filename = f"{folder_path}/HPR_2D_n{n}_d1{d1}_d2{d2}_overall_chain.PNG"
    else:
        filename = f"{folder_path}/HPR_2D_n{n}_d1{d1}_d2{d2}_seed{seed}"
        filename += "_chain.PNG"
    plt.savefig(filename)
    plt.close()
    return
