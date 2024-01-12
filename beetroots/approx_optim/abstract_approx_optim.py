import abc
import datetime
import json
import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from beetroots.approx_optim.approach_type import utils


class ApproxParamsOptim(abc.ABC):
    r"""Optimization of an approximation parameter for the likelihood function ``beetroots.modelling.likelihoods.approx_censored_add_mult.MixingModelsLikelihood``.
    The optimization framework used to adjust this parameter is introduced in Appendix A of :cite:t:`paludEfficientSamplingNon2023`.
    """

    def __init__(
        self,
        list_lines,
        name: str,
        D: int,
        D_no_kappa: int,
        K: int,
        log10_f_grid_size: int,
        N_samples_y: int,
        max_workers: int,
        sigma_a: Union[np.ndarray, float],
        sigma_m: Union[np.ndarray, float],
        small_size: int = 16,
        medium_size: int = 20,
        bigger_size: int = 24,
    ):
        self.max_workers = max_workers

        self.list_lines = list_lines
        self.L = len(list_lines)

        if isinstance(sigma_a, np.ndarray):
            assert len(sigma_a.shape) == 2, f"{sigma_a.shape}"
            assert sigma_a.shape[1] == self.L
            self.N = sigma_a.shape[0]
        else:
            self.N = 1
            sigma_a = sigma_a * np.ones((self.N, self.L))

        if isinstance(sigma_m, np.ndarray):
            assert sigma_m.shape == (self.N, self.L), f"{sigma_m.shape}"
        else:
            self.N = 1
            sigma_m = sigma_m * np.ones((self.N, self.L))

        assert isinstance(sigma_a, np.ndarray)
        assert isinstance(sigma_m, np.ndarray)
        self.sigma_a = sigma_a
        self.sigma_m = sigma_m

        self.D = D
        self.D_no_kappa = D_no_kappa

        self.K = K
        self.N_samples_y = N_samples_y  # number of samples on $y_\ell$
        self.N_samples_Theta = K**D  # to build pdf of $P(theta)$
        self.log10_f_grid_size = log10_f_grid_size  # nb points P(theta) grid

        self.setup_plot_text_sizes(small_size, medium_size, bigger_size)
        self.create_empty_output_folders(name)

    def setup_plot_text_sizes(
        self,
        small_size: int = 16,
        medium_size: int = 20,
        bigger_size: int = 24,
    ) -> None:
        """Defines text sizes on matplotlib plots"""
        plt.rc("font", size=small_size)  # controls default text sizes
        plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
        plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
        plt.rc("legend", fontsize=small_size)  # legend fontsize
        plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title
        return

    def create_empty_output_folders(self, name: str) -> None:
        r"""creates the output directories

        Parameters
        ----------
        name : str
            name of the simulation to be run
        """
        now = datetime.datetime.now()
        dt_str = now.strftime("%Y-%m-%d_%H")

        # path to the outputs dir
        path_ouput_general = f"{os.path.abspath(__file__)}/../../../../outputs"
        path_ouput_general = os.path.abspath(path_ouput_general)

        path_output_sim = f"{path_ouput_general}/approx_optim_{name}_{dt_str}"
        self.path_output_sim = os.path.abspath(path_output_sim)

        self.path_img = path_output_sim + "/img"
        self.path_img_hist = self.path_img + "/hist"
        self.path_img_hist_final = self.path_img + "/hist_final"
        self.path_img_final = self.path_img + "/final"
        self.path_logs = path_output_sim + "/logs"
        self.path_params = path_output_sim + "/optim_params"

        for folder_path in [
            path_ouput_general,
            self.path_output_sim,
            self.path_img,
            self.path_img_hist,
            self.path_img_hist_final,
            self.path_img_final,
            self.path_logs,
            self.path_params,
        ]:
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)

        return

    def sample_Theta(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
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

        Returns
        -------
        np.ndarray of shape (N_samples, D)
            $\theta$ samples
        """
        rng = np.random.default_rng(42)

        delta_upper_lower = np.array(
            [upper_bounds[d] - lower_bounds[d] for d in range(self.D)]
        )
        list_lower_bounds = [
            np.arange(self.K) / self.K * delta_upper_lower[d] + lower_bounds[d]
            for d in range(self.D)
        ]
        list_lower_bounds = np.meshgrid(*list_lower_bounds)
        list_lower_bounds = [
            lower_bounds_Theta_i.flatten() for lower_bounds_Theta_i in list_lower_bounds
        ]
        list_samples = []
        for d, lower_bounds_Theta_i in enumerate(list_lower_bounds):
            Vd = rng.uniform(
                low=lower_bounds_Theta_i,
                high=lower_bounds_Theta_i + delta_upper_lower[d] / self.K,
            )
            list_samples.append(Vd)
        x = np.vstack(list_samples).T
        return x

    def plot_hist_log10_f_Theta(
        self,
        log10_f_Theta: np.ndarray,
        log10_f_Theta_low: float,
        log10_f_Theta_high: float,
        list_log10_f_grid: np.ndarray,
        pdf_kde_log10_f_Theta: np.ndarray,
        ell: int,
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
        img_path : str
            path of folder where the figure is to be saved
        """
        plt.figure(figsize=(8, 6))
        plt.title(f"line {ell} : {self.list_lines[ell]}")
        plt.hist(
            log10_f_Theta,
            bins=self.log10_f_grid_size,
            range=(log10_f_Theta_low, log10_f_Theta_high),
            density=True,
            label="samples",
        )
        plt.plot(list_log10_f_grid, pdf_kde_log10_f_Theta, "k--", label="KDE")
        plt.xlabel(r"$\log f_\ell (\theta)$")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.path_img_hist}/hist_log_f_Theta_{self.list_lines[ell]}.PNG")
        plt.close()

        return

    def plot_hist_log10_f_Theta_with_optim_results(
        self,
        log10_f_Theta: np.ndarray,
        log10_f_Theta_low: float,
        log10_f_Theta_high: float,
        list_log10_f_grid: np.ndarray,
        pdf_kde_log10_f_Theta: np.ndarray,
        n: int,
        ell: int,
        best_point: np.ndarray,
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
        img_path : str
            path of folder where the figure is to be saved
        """
        lambda_ = 1 - utils.compute_lambda(
            a0=best_point[0],
            a1=best_point[1],
            f_Theta_true=np.exp(list_log10_f_grid * np.log(10.0)),
        )

        var_eps_m = np.exp(self.sigma_m**2) * (np.exp(self.sigma_m**2) - 1)
        log10_f0_n_ell = (
            0.5
            * (2 * np.log(self.sigma_a[n, ell]) - np.log(var_eps_m[n, ell]))
            / np.log(10)
        )

        plt.figure(figsize=(8, 6))
        plt.title(f"line {ell} : {self.list_lines[ell]}")
        vals, _, _ = plt.hist(
            log10_f_Theta,
            bins=self.log10_f_grid_size,
            range=(log10_f_Theta_low, log10_f_Theta_high),
            density=True,
            label="samples",
        )
        plt.plot(list_log10_f_grid, pdf_kde_log10_f_Theta, "k--", label="KDE")
        plt.plot(
            list_log10_f_grid,
            lambda_ * vals.max(),
            "r-",
            label=r"multi. weight $\lambda$",
        )
        plt.axvline(
            np.log10(self.sigma_a[n, ell]), c="orange", ls="-", label=r"$\sigma_a$"
        )
        plt.axvline(
            log10_f0_n_ell,
            c="orange",
            ls="--",
            label=r"$\sigma_a^2 = f_\ell(\theta)^2$ Var$( \epsilon^{(m)}_{\ell} )$",
        )

        plt.xlabel(r"$\log_{10} f_\ell (\theta)$")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"{self.path_img_hist_final}/hist_final_log_f_Theta_n{n}_{self.list_lines[ell]}.PNG"
        )
        plt.close()

        return

    def save_setup_to_json(self, n: int, ell: int, pbounds: Dict) -> None:
        """save optimization context and parameters

        Parameters
        ----------
        n : _type_
            _description_
        ell : _type_
            _description_
        pbounds : _type_
            _description_
        """
        optim_params = {
            "n": n,
            "ell": ell,
            "line": self.list_lines[ell],
            "sigma_a": self.sigma_a[n, ell],
            "sigma_m": self.sigma_m[n, ell],
            "K": self.K,
            "N_samples_Theta": self.N_samples_Theta,
            "N_samples_y": self.N_samples_y,
            "log10_f_grid_size": self.log10_f_grid_size,
            "a0": {"low": pbounds["a0"][0], "high": pbounds["a0"][1]},
            "a1": {"low": pbounds["a1"][0], "high": pbounds["a1"][1]},
        }
        with open(
            f"{self.path_params}/optim_params_n{n}_{self.list_lines[ell]}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                optim_params,
                f,
                ensure_ascii=False,
                indent=4,
            )
        return

    def rewrite_logs_correct_json_format(self) -> None:
        """rewrites the log files with correct json format"""
        for ell in range(self.L):
            line = self.list_lines[ell] * 1

            for n in range(self.N):
                with open(f"{self.path_logs}/logs_n{n}_{line}.json", "r") as f:
                    text = f.read()

                text = "[" + text.replace("}}", "}},")[:-2] + "]"
                with open(
                    f"{self.path_logs}/logs_correct_format_n{n}_{line}.json",
                    "w",
                ) as f:
                    f.write(text)

        return

    def extract_optimal_params(self) -> pd.DataFrame:
        self.rewrite_logs_correct_json_format()

        list_best = []
        for n in range(self.N):
            for ell in range(self.L):
                line = self.list_lines[ell] * 1
                # import data
                with open(
                    f"{self.path_logs}/logs_correct_format_n{n}_{line}.json",
                    "r",
                ) as f:
                    data = json.load(f)

                # Flatten data
                df = pd.json_normalize(data, max_level=1)
                df = df.rename(columns={"params.a0": "a0", "params.a1": "a1"})

                # get best
                df_sorted = df.sort_values("target", ascending=False)
                idx_best = df_sorted.index[0]
                dict_best = {
                    "n": n,
                    "ell": ell,
                    "line": self.list_lines[ell],
                    "a0_best": df.at[idx_best, "a0"],
                    "a1_best": df.at[idx_best, "a1"],
                    "target_best": df.at[idx_best, "target"],
                }
                list_best.append(dict_best)

        # * save best points
        df_best = pd.DataFrame(list_best)
        df_best.to_csv(f"{self.path_output_sim}/best_params.csv", index=False)
        return df_best

    def setup_params_bounds(self):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        sigma_a : Union[np.ndarray, float]
            _description_
        sigma_m : Union[np.ndarray, float]
            _description_

        Returns
        -------
        Tuple[List[Dict[str, float]], np.ndarray]
            _description_
        """
        # log10_f0 = log10(ratio std of noises)
        # ie value of f at which noise variances are equal
        var_eps_m: np.ndarray = np.exp(self.sigma_m**2) * (
            np.exp(self.sigma_m**2) - 1
        )
        log10_f0: np.ndarray = (
            0.5 * (2 * np.log(self.sigma_a) - np.log(var_eps_m)) / np.log(10)
        )

        bounds_a0_low: np.ndarray = log10_f0 - 2  # (N, L)
        bounds_a0_high: np.ndarray = log10_f0 + 8  # (N, L)

        bounds_a1_low = 0.01
        bounds_a1_high = 2.0

        return (
            log10_f0,
            bounds_a0_low,
            bounds_a0_high,
            bounds_a1_low,
            bounds_a1_high,
        )
