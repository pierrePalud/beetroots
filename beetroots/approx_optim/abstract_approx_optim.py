import abc
import datetime
import json
import os
import sys
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
        list_lines: List[str],
        simu_name: str,
        D: int,
        D_no_kappa: int,
        K: int,
        log10_f_grid_size: int,
        N_samples_y: int,
        max_workers: int,
        sigma_a: Union[np.ndarray, float],
        sigma_m: Union[np.ndarray, float],
        path_outputs: str,
        path_models: str,
        small_size: int = 16,
        medium_size: int = 20,
        bigger_size: int = 24,
    ):
        r"""

        Parameters
        ----------
        list_lines : List[str]
            names of the observables for which the likelihood parameter needs to be adjusted
        simu_name : str
            simu_name of the process, to be used in the output folder simu_name
        D : int
            total number of physical parameters involved in the forward map
        D_no_kappa : int
            total number of physical parameters involved in the forward map except for the scaling parameter :math:`\kappa` (if it is part of the considered physical parameters)
        K : int
            the number of sampled theta values is :math:`K^D`
        log10_f_grid_size : int
            number of points in the grid on :math:`\log f_\ell(\theta)`
        N_samples_y : int
            number of samples for :math:`y_\ell`
        max_workers : int
            maximum number of workers that can be used for optimization or results extraction
        sigma_a : Union[np.ndarray, float]
            standard deviation of the additive Gaussian noise
        sigma_m : Union[np.ndarray, float]
            standard deviation parameter of the multiplicative lognormal noise
        path_data : str
            path to the folder containing the input yaml file
        path_outputs : str
            path to the output folder (to be created), where the run results are to be saved
        path_models : str
            path to the folder containing the forward models
        small_size : int, optional
            size for basic text, axes titles, xticks and yticks, by default 16
        medium_size : int, optional
            size of the axis labels, by default 20
        bigger_size : int, optional
            size of the figure title, by default 24
        """
        self.max_workers = max_workers
        r"""int: maximum number of workers that can be used for optimization or results extraction"""

        self.list_lines = list_lines
        r"""List[str]: names of the observables for which the likelihood parameter needs to be adjusted"""
        self.L = len(list_lines)

        if isinstance(sigma_a, np.ndarray):
            assert len(sigma_a.shape) == 2, f"{sigma_a.shape}"
            assert sigma_a.shape[1] == self.L
            N = sigma_a.shape[0]
        else:
            N = 1
            sigma_a = sigma_a * np.ones((N, self.L))

        if isinstance(sigma_m, np.ndarray):
            assert sigma_m.shape == (N, self.L), f"{sigma_m.shape}"
        else:
            N = 1
            sigma_m = sigma_m * np.ones((N, self.L))

        self.N = N
        r"""int: number of pixels / components for which the optimization needs to be performed"""

        assert isinstance(sigma_a, np.ndarray)
        assert isinstance(sigma_m, np.ndarray)
        self.sigma_a = sigma_a
        r"""np.ndarray: standard deviations of the additive Gaussian noise"""
        self.sigma_m = sigma_m
        r"""np.ndarray: standard deviation parameter of the multiplicative lognormal noise"""

        self.D = D
        r"""int: total number of physical parameters involved in the forward map"""

        self.D_no_kappa = D_no_kappa
        r"""int: total number of physical parameters involved in the forward map except for the scaling parameter :math:`\kappa` (if it is part of the considered physical parameters)"""

        self.K = K
        r"""int: the number of sampled theta values is ``K^D_sampling``"""
        self.N_samples_y = N_samples_y
        r"""int: number of samples for :math:`y_\ell`"""
        self.log10_f_grid_size = log10_f_grid_size  # nb points P(theta) grid
        r"""int: number of points in the grid on :math:`\log_{10} f_\ell(\theta)`"""

        self.N_samples_theta = 0
        r"""int: number of samples for :math:`\theta` used to build the histogram of :math:`\log_{10} f_\ell(\theta)`. To be defined in the daughter classes."""

        self.setup_plot_text_sizes(small_size, medium_size, bigger_size)
        self.create_empty_output_folders(simu_name, path_outputs)

        self.MODELS_PATH = path_models
        r"""str: path to the folder containing all the already defined and saved models (i.e., polynomials or neural networks)"""

    @classmethod
    def parse_args(cls) -> Tuple[str, str, str, str]:
        """parses the inputs of the command-line, that should contain

        - the name of the input YAML file
        - path to the data folder
        - path to the models folder
        - path to the outputs folder to be created (by default '.')

        Returns
        -------
        str
            name of the input YAML file
        str
            path to the data folder
        str
            path to the models folder
        str
            path to the outputs folder to be created (by default '.')
        """
        if len(sys.argv) < 4:
            raise ValueError(
                "Please provide the following  arguments in your command: \n 1) the name of the input YAML file, \n 2) the path to the data folder, \n 3) the path to the models folder, \n 4) the path to the outputs folder to be created (by default '.')"
            )

        yaml_file = sys.argv[1]
        path_data = os.path.abspath(sys.argv[2])
        path_models = os.path.abspath(sys.argv[3])

        path_outputs = (
            os.path.abspath(sys.argv[4]) if len(sys.argv) == 5 else os.path.abspath(".")
        )
        path_outputs += "/outputs"

        print(f"input file name: {yaml_file}")
        print(f"path to data folder: {path_data}")
        print(f"path to models folder: {path_models}")
        print(f"path to outputs folder: {path_outputs}")

        return yaml_file, path_data, path_models, path_outputs

    def setup_plot_text_sizes(
        self,
        small_size: int = 16,
        medium_size: int = 20,
        bigger_size: int = 24,
    ) -> None:
        r"""defines text sizes on matplotlib plots

        Parameters
        ----------
        small_size : int, optional
            size for basic text, axes titles, xticks and yticks, by default 16
        medium_size : int, optional
            size of the axis labels, by default 20
        bigger_size : int, optional
            size of the figure title, by default 24
        """
        plt.rc("font", size=small_size)  # controls default text sizes
        plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
        plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
        plt.rc("legend", fontsize=small_size)  # legend fontsize
        plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title
        return

    def create_empty_output_folders(self, simu_name: str, path_outputs: str) -> None:
        r"""creates the directories that receive the results of the likelihood parameter optimization

        Parameters
        ----------
        simu_name : str
            name of the simulation to be run
        path_yaml_file : str
            path of the folder containing the data and yaml files
        path_outputs: str
            folder where to write outputs
        """
        now = datetime.datetime.now()
        dt_str = now.strftime("%Y-%m-%d_%H")

        self.path_output_sim = os.path.abspath(
            f"{path_outputs}/approx_optim_{simu_name}_{dt_str}"
        )
        r"""str: path to the output root folder, e.g., ``./outputs/simu1``"""

        self.path_img = self.path_output_sim + "/img"
        r"""str: path to the output image folder, e.g., ``./outputs/simu1/img``"""
        self.path_img_hist = self.path_img + "/hist"
        r"""str: path to the histogram image folder, e.g., ``./outputs/simu1/img/hist``"""
        self.path_img_hist_final = self.path_img + "/hist_final"
        r"""str: path to the final histogram image folder, e.g., ``./outputs/simu1/img/hist_final``"""
        self.path_img_final = self.path_img + "/final"
        r"""str: path to the final image folder, e.g., ``./outputs/simu1/img/final``"""
        self.path_logs = self.path_output_sim + "/logs"
        r"""str: path to the procedure logs, e.g., ``./outputs/logs``"""
        self.path_params = self.path_output_sim + "/optim_params"
        r"""str: path to the adjusted likelihood parameters, e.g., ``./outputs/optim_params``"""

        for folder_path in [
            path_outputs,
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

    def sample_theta(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> np.ndarray:
        r"""sample :math:`\theta` from Stratified MC in cube

        Parameters
        ----------
        K : int
            total number of samples per axis
        lower_bounds : np.ndarray of shape (D,)
            lower bounds of cube on :math:`\theta`
        upper_bounds : np.ndarray of shape (D,)
            upper bounds of cube on :math:`\theta`

        Returns
        -------
        np.ndarray of shape (N_samples, D)
            :math:`\theta` samples
        """
        print("starting generation of theta values with Stratified MC")
        rng = np.random.default_rng(42)
        D_sampling: int = upper_bounds.size

        delta_upper_lower = np.array(
            [upper_bounds[d] - lower_bounds[d] for d in range(D_sampling)]
        )
        list_lower_bounds = [
            np.arange(self.K) / self.K * delta_upper_lower[d] + lower_bounds[d]
            for d in range(D_sampling)
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
        print("generation of theta values with Stratified MC done")
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
        r"""plots histogram of :math:`log_{10}(f_\ell(\theta))`

        Parameters
        ----------
        log10_f_Theta : np.ndarray of shape (-1, 1)
            array of values of :math:`\log_{10} f_\ell (\theta)` for considered line
        log10_f_Theta_low: float
            lower bound for :math:`\log_{10} f_\ell (\theta)` for the considered line
        log10_f_Theta_high: float
            upper bound for :math:`\log_{10} f_\ell (\theta)` for the considered line
        list_log10_f_grid: np.ndarray
            grid values of :math:`\log_{10} f_\ell (\theta)` for the considered line
        pdf_kde_log10_f_Theta: np.ndarray
            pdf of :math:`\log_{10} f_\ell (\theta)` evaluated with a kernel density estimator
        ell : int
            index of the line
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
        r"""plots histogram of :math:`\log_{10} f_\ell (\theta)`

        Parameters
        ----------
        log10_f_Theta : np.ndarray of shape (-1, 1)
            array of values of :math:`\log_{10} f_\ell (\theta)` for considered line
        log10_f_Theta_low: float
            lower bound for :math:`\log_{10} f_\ell (\theta)` for the considered line
        log10_f_Theta_high: float
            upper bound for :math:`\log_{10} f_\ell (\theta)` for the considered line
        list_log10_f_grid: np.ndarray
            grid values of :math:`\log_{10} f_\ell (\theta)` for the considered line
        pdf_kde_log10_f_Theta: np.ndarray
            pdf of :math:`\log_{10} f_\ell (\theta)` evaluated with a kernel density estimator
        n : int
            pixel / component index
        ell : int
            index of the line
        best_point : np.ndarray
            position for the best point, to be displayed
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
            lambda_ * np.max(vals),
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

    def save_setup_to_json(
        self, n: int, ell: int, pbounds: Dict[str, np.ndarray]
    ) -> None:
        """save optimization context and parameters

        Parameters
        ----------
        n : int
            pixel / component index
        ell : int
            observable index
        pbounds : dict[str, np.ndarray]
            contains the bounds on the parameters to be adjusted
        """
        optim_params = {
            "n": n,
            "ell": ell,
            "line": self.list_lines[ell],
            "sigma_a": self.sigma_a[n, ell],
            "sigma_m": self.sigma_m[n, ell],
            "K": self.K,
            "N_samples_theta": self.N_samples_theta,
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
        r"""rewrites the log files with correct json format"""
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
        """extracts the adjusted likelihood parameters from the log files and gather them in a DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with the set of evaluated optimal parameters for each component n and line ell
        """
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

    def setup_params_bounds(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        r"""sets the bounds on the parameters to be adjusted, defined here as transition interval center for ``a0`` position and width for ``a1``

        Returns
        -------
        log10_f0 : np.ndarray of shape (N, L)
            values of :math:`f_\ell(\theta)` at which additive and multiplicative noise variances are equal
        bounds_a0_low : np.ndarray of shape (N, L)
            lower bounds on the center of the transition interval (defined as deltas around the ``log10_f0``)
        bounds_a0_high : np.ndarray of shape (N, L)
            upper bounds on the center of the transition interval (defined as deltas around the ``log10_f0``)
        bounds_a1_low : float
            lower value for the transition interval size
        bounds_a1_high : float
            upper value for the transition interval size
        """
        print("starting definition of parameters bounds")
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
        print("definition of parameters bounds done")

        return (
            log10_f0,
            bounds_a0_low,
            bounds_a0_high,
            bounds_a1_low,
            bounds_a1_high,
        )
