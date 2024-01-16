import json
from functools import partial
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from matplotlib.ticker import FuncFormatter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from tqdm.auto import tqdm

import beetroots.approx_optim.approach_type.utils as utils
from beetroots.approx_optim.approach_type.abstract_approach_type import ApproachType


class BayesianOptimizationApproach(ApproachType):
    r"""implements a Bayesian optimization approach to adjust the likelihood parameter"""

    def optimization(
        self,
        first_points: List,
        init_points: int,
        n_iter: int,
        list_log10_f_grid: np.ndarray,
        pdf_kde_log10_f_Theta: np.ndarray,
        pbounds: Dict,
        sigma_a_val: float,
        sigma_m_val: float,
        n: int,
        ell: int,
    ) -> None:
        """performs Bayesian optimization

        Parameters
        ----------
        first_points : List
            list of the points to evaluate before running the optimization
        init_points : int
            number of init points
        n_iter : int
            number of optimization steps to perform
        list_log10_f_grid : np.ndarray
            _description_
        pdf_kde_log10_f_Theta : np.ndarray
            _description_
        pbounds : Dict
            _description_
        sigma_a_val : float
            _description_
        sigma_m_val : float
            _description_
        n : int
            pixel index for which the optimization is performed
        ell : int
            line index for which the optimization is performed
        """

        # fix secondary parameters
        estimate_avg_dks = partial(
            utils.estimate_avg_dks_full_bo,
            list_log10_f_grid=list_log10_f_grid,
            pdf_kde_log10_f_Theta=pdf_kde_log10_f_Theta,
            sigma_a=sigma_a_val,
            sigma_m=sigma_m_val,
            N_samples_y=self.N_samples_y,
            max_workers=self.max_workers,
        )

        # setup optimization
        optimizer = BayesianOptimization(
            f=estimate_avg_dks,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )

        logger = JSONLogger(
            path=f"{self.path_logs}/logs_n{n}_{self.list_lines[ell]}.json"
        )
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        # preliminary points
        for point in first_points:
            optimizer.probe(
                params={"a0": point[0], "a1": point[1]},
                lazy=True,
            )

        # perform optimization
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        return

    def plots_postprocessing(
        self,
        bounds_a0_low: np.ndarray,
        bounds_a0_high: np.ndarray,
        bounds_a1_low: float,
        bounds_a1_high: float,
        n_iter: int,
    ) -> None:
        """makes some plots after the optimization process to facilitate the execution understanding

        Parameters
        ----------
        bounds_a0_low : np.ndarray
            array of lower bounds on the transition location
        bounds_a0_high : np.ndarray
            array of upper bounds on the transition location
        bounds_a1_low : float
            lower bound on the transition slope
        bounds_a1_high : float
            upper bound on the transition slope
        n_iter : int
            number of performed optimization iterations
        """
        self.plot_evaluation_points(
            bounds_a0_low,
            bounds_a0_high,
            bounds_a1_low,
            bounds_a1_high,
        )
        self.plot_GP(
            bounds_a0_low,
            bounds_a0_high,
            bounds_a1_low,
            bounds_a1_high,
            n_iter,
        )
        return

    def plot_evaluation_points(
        self,
        bounds_a0_low: np.ndarray,
        bounds_a0_high: np.ndarray,
        bounds_a1_low: float,
        bounds_a1_high: float,
    ) -> None:
        """plots the sequence of sampled points with the order as colorbar

        Parameters
        ----------
        bounds_a0_low : np.ndarray
            array of lower bounds on the transition location
        bounds_a0_high : np.ndarray
            array of upper bounds on the transition location
        bounds_a1_low : float
            lower bound on the transition slope
        bounds_a1_high : float
            upper bound on the transition slope
        """
        for n in range(self.N):
            for ell in tqdm(range(self.L)):
                with open(
                    f"{self.path_logs}/logs_correct_format_n{n}_{self.list_lines[ell]}.json",
                    "r",
                ) as f:
                    data = json.load(f)

                # Flatten data
                df = pd.json_normalize(data, max_level=1)
                df = df.rename(columns={"params.a0": "a0", "params.a1": "a1"})

                plt.figure(figsize=(8, 6))
                plt.title("order of evaluation")
                plt.scatter(df["a0"], df["a1"], c=df.index.values)
                plt.xlim([bounds_a0_low[n, ell] - 0.5, bounds_a0_high[n, ell] + 0.5])
                plt.ylim([bounds_a1_low - 0.1, bounds_a1_high + 0.1])
                plt.colorbar()
                plt.grid()
                plt.savefig(
                    f"{self.path_img_final}/order_points_n{n}_{self.list_lines[ell]}.PNG"
                )
                plt.close()

                plt.figure(figsize=(8, 6))
                plt.title("target evaluated values")
                plt.scatter(df["a0"], df["a1"], c=df["target"])
                plt.xlim([bounds_a0_low[n, ell] - 0.5, bounds_a0_high[n, ell] + 0.5])
                plt.ylim([bounds_a1_low - 0.1, bounds_a1_high + 0.1])
                plt.colorbar()
                plt.grid()
                plt.savefig(
                    f"{self.path_img_final}/target_points_n{n}_{self.list_lines[ell]}.PNG"
                )
                plt.close()

        return

    def plot_GP(
        self,
        bounds_a0_low: np.ndarray,
        bounds_a0_high: np.ndarray,
        bounds_a1_low: float,
        bounds_a1_high: float,
        n_iter: int,
    ) -> None:
        """plot the state (both mean and standard deviations) of the Gaussian process at multiple steps of the optimization process.

        Parameters
        ----------
        bounds_a0_low : np.ndarray
            array of lower bounds on the transition location
        bounds_a0_high : np.ndarray
            array of upper bounds on the transition location
        bounds_a1_low : float
            lower bound on the transition slope
        bounds_a1_high : float
            upper bound on the transition slope
        n_iter : int
            number of performed optimization iterations
        """
        # show_theta_scale = True
        # show_y_scale = True

        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-2,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=1,
        )
        n_a0 = 100
        n_a1 = 100

        fmt_mu = lambda x, pos: "{:.2f}".format(x)
        fmt_sigma = lambda x, pos: "{:.3f}".format(x)

        for n in range(self.N):
            for ell in tqdm(range(self.L)):
                with open(
                    f"{self.path_logs}/logs_correct_format_n{n}_{self.list_lines[ell]}.json",
                    "r",
                ) as f:
                    data = json.load(f)

                # Flatten data
                df = pd.json_normalize(data, max_level=1)
                df = df.rename(columns={"params.a0": "a0", "params.a1": "a1"})

                list_a0 = np.linspace(
                    bounds_a0_low[n, ell], bounds_a0_high[n, ell], n_a0
                )
                list_a1 = np.linspace(bounds_a1_low, bounds_a1_high, n_a1)
                AA0, AA1 = np.meshgrid(list_a0, list_a1)

                assert AA0.shape == (n_a1, n_a0)

                grid = np.vstack((AA0.flatten(), AA1.flatten())).T  # (n_a0 * n_a1, 2)

                sigma_a = self.sigma_a[n, ell]
                sigma_m = self.sigma_m[n, ell]

                var_eps_m = np.exp(sigma_m**2) * (np.exp(sigma_m**2) - 1)
                log10_f0 = 0.5 * (2 * np.log(sigma_a) - np.log(var_eps_m)) / np.log(10)

                # with full dataset
                a0_obs_full = df.loc[:, "a0"].values
                a1_obs_full = df.loc[:, "a1"].values
                x_obs_full = np.vstack((a0_obs_full, a1_obs_full)).T
                y_obs_full = df.loc[:, "target"].values

                df_sorted = df.sort_values("target", ascending=False)
                df_sorted = df_sorted.reset_index()
                a0_best, a1_best = df_sorted.loc[0, ["a0", "a1"]]

                gp.fit(x_obs_full, y_obs_full)
                mu_full, sigma_full = gp.predict(grid, return_std=True)

                mu_full = mu_full.reshape((n_a1, n_a0))
                sigma_full = sigma_full.reshape((n_a1, n_a0))

                mu_min = mu_full.min()
                mu_max = mu_full.max()
                sigma_min = sigma_full.min()
                sigma_max = sigma_full.max()

                for i_iter, show_estimate in zip([5, 20, n_iter], [False, False, True]):
                    a0_obs = df.loc[:i_iter, "a0"].values
                    a1_obs = df.loc[:i_iter, "a1"].values
                    x_obs = np.vstack((a0_obs, a1_obs)).T
                    y_obs = df.loc[:i_iter, "target"].values

                    gp.fit(x_obs, y_obs)
                    mu, sigma = gp.predict(grid, return_std=True)
                    mu = mu.reshape((n_a1, n_a0))
                    sigma = sigma.reshape((n_a1, n_a0))

                    plt.figure(figsize=(8, 6))
                    plt.title(f"GP mean after {i_iter} iterations", fontsize=24)
                    plt.contourf(
                        AA0,
                        AA1,
                        mu,
                        levels=100,
                        vmax=mu_max,
                        vmin=mu_min,
                    )
                    plt.axvline(log10_f0, c="r", ls="--")  # , label="equal variances")
                    plt.colorbar(format=FuncFormatter(fmt_mu))
                    plt.scatter(a0_obs, a1_obs, c="k")

                    if show_estimate:
                        plt.scatter(
                            [a0_best],
                            [a1_best],
                            c="r",
                            marker="+",
                            label=r"$\hat{a}$",
                            linewidths=3.5,
                            s=225,
                        )
                        plt.legend()

                    # if show_theta_scale:
                    plt.xlabel(r"interval center $(a_1 + a_0) \; / \; 2$")
                    # plt.xticks([-10, -8, -6, -4, -2], [-10, -8, -6, -4, -2])
                    # else:
                    # plt.xticks([-10, -8, -6, -4, -2], [])

                    # if show_y_scale:
                    plt.ylabel(r"interval radius $(a_1 - a_0) \; / \; 2$")
                    # plt.yticks([0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 1.5, 2.0])
                    # else:
                    # plt.yticks([0.5, 1.0, 1.5, 2.0], [])

                    plt.tight_layout()
                    plt.savefig(
                        f"{self.path_img_final}/gp_n{n}_{self.list_lines[ell]}_mean_iter{i_iter}.PNG",
                        bbox_inches="tight",
                        transparent=True,
                    )
                    plt.close()

                    plt.figure(figsize=(8, 6))
                    plt.title(f"GP std after {i_iter} iterations", fontsize=24)
                    # plt.title("std of GP")
                    plt.contourf(
                        AA0,
                        AA1,
                        sigma,
                        levels=100,
                        vmax=sigma_max,
                        vmin=sigma_min,
                    )
                    plt.axvline(log10_f0, c="r", ls="--")  # , label="equal variances")
                    plt.colorbar(format=FuncFormatter(fmt_sigma))
                    plt.scatter(a0_obs, a1_obs, c="k")

                    if show_estimate:
                        plt.scatter(
                            [a0_best],
                            [a1_best],
                            c="r",
                            marker="+",
                            label=r"$\hat{a}$",
                            linewidths=3.5,
                            s=225,
                        )
                        plt.legend()

                    # if show_theta_scale:
                    plt.xlabel(r"interval center $(a_1 + a_0) \; / \; 2$")
                    #     plt.xticks([-10, -8, -6, -4, -2], [-10, -8, -6, -4, -2])
                    # else:
                    #     plt.xticks([-10, -8, -6, -4, -2], [])

                    # if show_y_scale:
                    plt.ylabel(r"interval radius $(a_1 - a_0) \; / \; 2$")
                    #     plt.yticks([0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 1.5, 2.0])
                    # else:
                    #     plt.yticks([0.5, 1.0, 1.5, 2.0], [])

                    # plt.ylabel(r"mixing interval radius $(a_1 - a_0)/2$")
                    plt.tight_layout()

                    plt.savefig(
                        f"{self.path_img_final}/gp_n{n}_{self.list_lines[ell]}_std_iter{i_iter}.PNG",
                        bbox_inches="tight",
                        transparent=True,
                    )
                    plt.close()

        return

    def plot_cost_map(self):
        pass
