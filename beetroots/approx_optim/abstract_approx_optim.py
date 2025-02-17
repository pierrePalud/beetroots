import abc
import datetime
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
        sigma_a_raw: Union[np.ndarray, float],
        sigma_m: Union[np.ndarray, float],
        path_outputs: str,
        path_models: str,
        N_clusters_a_priori: Optional[int],
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
        sigma_a_raw : Union[np.ndarray, float]
            standard deviation of the additive Gaussian noise
        sigma_m : Union[np.ndarray, float]
            standard deviation parameter of the multiplicative lognormal noise
        path_outputs : str
            path to the output folder (to be created), where the run results are to be saved
        path_models : str
            path to the folder containing the forward models
        N_clusters_a_priori: Optional[int]
            The number of different values of sigma_a_raw to consider for each line (and thus of optimization problem to solve per line), computed with a clustering algorithm on the `N` of values sigma_a_raw. Raises an error if `N_clusters_a_priori > N`.
        small_size : int, optional
            size for basic text, axes titles, xticks and yticks, by default 16
        medium_size : int, optional
            size of the axis labels, by default 20
        bigger_size : int, optional
            size of the figure title, by default 24
        """
        self.setup_plot_text_sizes(small_size, medium_size, bigger_size)
        self.create_empty_output_folders(simu_name, path_outputs)

        self.MODELS_PATH = path_models
        r"""str: path to the folder containing all the already defined and saved models (i.e., polynomials or neural networks)"""

        self.path_intermediate_result = (
            f"{self.path_output_sim}/best_params_intermediate.csv"
        )
        r"""str: path to the intermediate results (per cluster, saved in `output` folder)"""

        self.path_centroids = f"{self.path_output_sim}/sigma_a_centroids.csv"
        r"""str: path to the csv file containing the sigma_a centroid values for each cluster (saved in `output` folder)"""

        self.max_workers = max_workers
        r"""int: maximum number of workers that can be used for optimization or results extraction"""

        self.list_lines = list_lines
        r"""List[str]: names of the observables for which the likelihood parameter needs to be adjusted"""

        self.L = len(list_lines)
        r"""int: number of observables for which the likelihood parameter needs to be adjusted"""

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

        # check size of sigma_a_raw
        if isinstance(sigma_a_raw, np.ndarray):
            assert len(sigma_a_raw.shape) == 2, f"{sigma_a_raw.shape}"
            assert sigma_a_raw.shape[1] == self.L
            N = sigma_a_raw.shape[0]
        else:
            N = 1
            sigma_a_raw = sigma_a_raw * np.ones((N, self.L))

        self.N = N
        r"""int: number of pixels / components for which the optimization needs to be performed"""

        self.N_clusters = self.check_num_uniques(sigma_a_raw, N_clusters_a_priori)
        r"""Optional[int]: The number of different values of sigma_a to consider for each line (and thus of optimization problem to solve per line), computed with a clustering algorithm on the `N` of values sigma_a."""

        self.N_optim_per_line = self.N if self.N_clusters is None else self.N_clusters
        r"""int: number of optimization procedures to run per line"""

        print(
            f"run properties: N={self.N}, L={self.L}, N_clusters_a_priori={N_clusters_a_priori}, N_clusters={self.N_clusters}, N_optim={self.N_optim_per_line}\n"
        )

        # set sigma_a to sigma_a_raw if no clustering, else to centroids
        if self.N_optim_per_line == self.N:
            sigma_a = sigma_a_raw * 1
        else:
            # run the clustering algorithms on the set of standard deviations
            # for each line
            assert self.N_optim_per_line == self.N_clusters
            sigma_a = self.cluster_sigma_a_raw(sigma_a_raw)

        # check size of sigma_m
        assert isinstance(
            sigma_m, float
        ), "The current implementation only works for constant sigma_m over the map"
        if isinstance(sigma_m, np.ndarray):
            assert sigma_m.shape == (self.N_optim_per_line, self.L), f"{sigma_m.shape}"
        else:
            sigma_m = sigma_m * np.ones((self.N_optim_per_line, self.L))

        assert isinstance(sigma_a, np.ndarray)
        assert isinstance(sigma_m, np.ndarray)
        self.sigma_a = sigma_a
        r"""np.ndarray: standard deviations of the additive Gaussian noise"""

        self.sigma_m = sigma_m
        r"""np.ndarray: standard deviation parameter of the multiplicative lognormal noise"""

        return

    def check_num_uniques(
        self,
        sigma_a_raw: np.ndarray,
        N_clusters_a_priori: Optional[int],
    ) -> Optional[int]:
        r"""Sets the number of clusters to consider `N_clusters`, that is, the number of optimization procedure to run per line. There are 4 possible cases:

        * case 1: set N_clusters with the number of distinct values of sigma_a
        * case 2: the number of distinct values of sigma_a is lower than the number of clusters provided by the user. In this case, use the value that minimizes the number of optimization procedures to run.
        * case 3: use the number of clusters indicated by the user.
        * case 4: last case : run one optim per pixel, ie run self.N optimizations.

        Parameters
        ----------
        sigma_a_raw : np.ndarray of shape (N, L)
            set of standard deviations in the
        N_clusters_a_priori : Optional[int]
            number of clusters to consider indicated by the user

        Returns
        -------
        Optional[int]
            definitive number of clusters to consider
        """
        assert (N_clusters_a_priori is None) or (
            N_clusters_a_priori <= self.N
        ), "The number of clusters `N_clusters` should be either None or inferior to the total number of pixels."

        assert sigma_a_raw.shape == (
            self.N,
            self.L,
        ), f"sigma_a_raw has shape {sigma_a_raw.shape} but should have ({self.N}, {self.L})"

        # compute the number of distinct values for sigma_a in a single line
        max_distincts = 1
        for ell in range(self.L):
            n_uniques = np.unique(sigma_a_raw[:, ell]).size
            if max_distincts < n_uniques:
                max_distincts = n_uniques * 1

        # case 1 : set N_clusters with the number of distinct values of sigma_a
        if max_distincts < self.N and N_clusters_a_priori is None:
            N_clusters = max_distincts * 1
            return N_clusters

        # case 2 : the number of distinct values of sigma_a is lower than
        # the number of clusters provided by the user.
        # In this case, use the value that minimizes the number of
        # optimization procedures to run.
        if N_clusters_a_priori is not None and max_distincts < N_clusters_a_priori:
            N_clusters = max_distincts * 1
            return N_clusters

        # case 3 : use the number of clusters indicated by the user
        if N_clusters_a_priori is not None:
            return N_clusters_a_priori

        # last case : run one optim per pixel, ie run self.N optimizations
        return None

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

        self.path_img_hist_sigma_a = self.path_img + "/hist_sigma_a"
        r"""str: path to the final histogram image folder, e.g., ``./outputs/simu1/img/hist_sigma_a``"""

        self.path_img_relation_sigma_a_approx = (
            self.path_img + "/relation_sigma_a_approx"
        )
        r"""str: path to the final histogram image folder, e.g., ``./outputs/simu1/img/relation_sigma_a_approx``"""

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
            self.path_img_hist_sigma_a,
            self.path_img_relation_sigma_a_approx,
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
        plt.xlabel(r"$\log_{10} f_\ell (\theta)$")
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

            for n in range(self.N_optim_per_line):
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
        for n in range(self.N_optim_per_line):
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
                    "sigma_a": self.sigma_a[n, ell],
                    "sigma_m": self.sigma_m[n, ell],
                    "a0_best": df.at[idx_best, "a0"],
                    "a1_best": df.at[idx_best, "a1"],
                    "target_best": df.at[idx_best, "target"],
                }
                list_best.append(dict_best)

        # * save best points
        df_best = pd.DataFrame(list_best)
        df_best.to_csv(self.path_intermediate_result, index=False)
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

    def cluster_sigma_a_raw(self, sigma_a_raw: np.ndarray) -> np.ndarray:
        r"""runs `self.L` k-means clustering algorithms (one per line) on the  sets of standard deviation of the additive noise `sigma_a`. The number of clusters is defined with `self.N_clusters`. The obtained dataframe is saved as a `csv` file.

        Parameters
        ----------
        sigma_a_raw : np.ndarray of shape (N, L)
            array of all the sigma_a values associated with the observations.

        Returns
        -------
        np.ndarray of shape (N_clusters, L)
            reduced sigma_a array, with `N_clusters` lines instead of `N` (with `N_clusters` potentially much smaller than `N`)
        """
        assert self.N_clusters is not None

        df_clusters = pd.DataFrame()
        df_clusters["Y"] = np.arange(self.N_clusters)
        df_clusters["X"] = 0

        for ell, line in enumerate(self.list_lines):
            log10_sigma_a_ell_nonnan = np.log10(sigma_a_raw[:, ell])
            log10_sigma_a_ell_nonnan = log10_sigma_a_ell_nonnan[
                ~np.isnan(log10_sigma_a_ell_nonnan)
            ]
            log10_sigma_a_ell_nonnan = log10_sigma_a_ell_nonnan[
                log10_sigma_a_ell_nonnan < 3.0
            ]

            max_min_diff = (
                log10_sigma_a_ell_nonnan.max() - log10_sigma_a_ell_nonnan.min()
            )

            km = KMeans(self.N_clusters).fit(log10_sigma_a_ell_nonnan.reshape((-1, 1)))
            log10_centroids = km.cluster_centers_.flatten()
            log10_centroids.sort()
            df_clusters[line] = np.exp(log10_centroids * np.log(10))

            self.plot_clusters_sigma_a(line, log10_sigma_a_ell_nonnan, log10_centroids)

            assert (
                max_min_diff < 15.0
            ), f"The difference between the min and max is over 15 orders of magnitude for line {line}. This should not happen."

        df_clusters = df_clusters.set_index(["X", "Y"])
        df_clusters.to_csv(self.path_centroids)

        return df_clusters.values

    def plot_clusters_sigma_a(
        self,
        line: str,
        log10_sigma_a_ell_nonnan: np.ndarray,
        log10_centroids: np.ndarray,
    ) -> None:
        r"""plots one one-dimension histogram on the log10 of the standard deviation on the additive noise in the observation, and the computed centroids.

        Parameters
        ----------
        line : str
            name of the line
        log10_sigma_a_ell_nonnan : np.ndarray
            non-nan values of sigma_a for the considered line
        log10_centroids : np.ndarray of shape (N_clusters,)
            values of the sigma_a centroids for the considered line
        """
        assert self.N_clusters is not None
        assert (
            isinstance(log10_sigma_a_ell_nonnan, np.ndarray)
            and len(log10_sigma_a_ell_nonnan.shape) == 1
        )
        assert (
            isinstance(log10_centroids, np.ndarray)
            and log10_centroids.size == self.N_clusters
        )

        plt.title(f"line {line}: sigma_a and centroids")
        plt.hist(log10_sigma_a_ell_nonnan, bins=50)

        for j in range(self.N_clusters):
            if j == 0:
                plt.axvline(log10_centroids[j], ls="--", c="k", label="centroids")
            else:
                plt.axvline(log10_centroids[j], ls="--", c="k")

        plt.ylabel("counts in sigma_a")
        plt.xlabel(
            r"$\log_{10} \sigma_{a,n\ell}$ for all pixels $n$ and one line $\ell$"
        )

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.path_img_hist_sigma_a}/sigma_a_and_centroids_{line}.PNG")
        plt.close()

    def save_results_in_data_folder(self, path_data: str, filename_err: str) -> None:
        r"""

        Parameters
        ----------
        path_data : str
            path to the data folder
        filename_err : str
            name of the file containing the sigma_a values
        """
        filename_final_result = f"{path_data}/best_params_approx.csv"

        if self.N_optim_per_line == self.N_clusters:
            # step 1 : aller chercher le best_params.csv dans l'output folder
            df_results = pd.read_csv(
                self.path_intermediate_result, index_col=["n", "line"]
            )

            # step 2 : create the large best_params.csv file
            # in case of nan, set a1_best = 3.0, a0_best=0.01
            df_err = pd.read_pickle(f"{path_data}/{filename_err}")
            df_err["n"] = np.arange(len(df_err))
            df_err = df_err.reset_index().set_index("n")

            df_centroids = pd.read_csv(self.path_centroids)

            cols_results = ["a0_best", "a1_best", "target_best"]

            df_best_params_full = pd.DataFrame(
                columns=cols_results,
                index=pd.MultiIndex.from_product(
                    [np.arange(self.N), self.list_lines], names=["n", "line"]
                ),
            )

            for line in self.list_lines:
                centroids = np.log10(df_centroids.loc[:, line].values)

                for n in range(self.N):
                    # get sigma_{a,n\ell}
                    log10_sigma_a = np.log10(df_err.at[n, line])

                    if np.isnan(log10_sigma_a):
                        df_best_params_full.loc[(n, line), cols_results] = [
                            3.0,
                            0.01,
                            np.nan,
                        ]
                    else:
                        idx_min = np.argmin(np.abs(log10_sigma_a - centroids))

                        df_best_params_full.loc[(n, line), cols_results] = (
                            df_results.loc[(idx_min, line), cols_results] * 1
                        )

            # step 3 : save it in the same folder as the input_params.yaml file
            df_best_params_full.to_csv(filename_final_result)

        else:
            # copy and paste the re
            assert self.N_optim_per_line == self.N
            shutil.copyfile(self.path_intermediate_result, filename_final_result)
        return

    def plot_params_with_sigma_a(self, df_best: pd.DataFrame) -> None:
        for ell, line in enumerate(self.list_lines):
            a0 = df_best.loc[df_best["line"] == line, "a0_best"]
            a1 = df_best.loc[df_best["line"] == line, "a1_best"]

            fig, ax = plt.subplots(
                1, 2, sharex=True, constrained_layout=True, figsize=(6, 3)
            )

            fig.suptitle(f"line {line}")
            ax[0].plot(np.log10(self.sigma_a[:, ell]), a0, "+")
            ax[0].set_ylabel("a0 best")
            ax[0].set_xlabel(r"$\log_{10} \sigma_a$")
            ax[0].grid()

            ax[1].plot(np.log10(self.sigma_a[:, ell]), a1, "+")
            ax[1].set_ylabel("a1 best")
            ax[1].set_xlabel(r"$\log_{10} \sigma_a$")
            ax[1].grid()

            plt.savefig(f"{self.path_img_relation_sigma_a_approx}/{line}.PNG")
            plt.close()
        return
