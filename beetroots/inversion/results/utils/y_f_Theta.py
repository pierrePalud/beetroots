import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from beetroots.inversion.plots import readable_line_names
from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.modelling.forward_maps.abstract_base import ForwardMap
from beetroots.space_transform.abstract_transform import Scaler


class ResultsDistributionComparisonYandFTheta(ResultsUtil):

    __slots__ = (
        "model_name",
        "path_img",
        "path_data_csv_out_mcmc",
        "N_MCMC",
        "T_MC",
        "T_BI",
        "freq_save",
        "effective_len_mc",
        "N",
        "D",
    )

    def __init__(
        self,
        model_name: str,
        path_img: str,
        path_data_csv_out_mcmc: str,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        freq_save: int,
        N: int,
        D: int,
        list_idx_sampling: List[int],
        max_workers: int,
    ):
        self.model_name = model_name
        self.path_img = path_img
        self.path_data_csv_out_mcmc = path_data_csv_out_mcmc

        self.N_MCMC = N_MCMC
        self.T_MC = T_MC
        self.T_BI = T_BI
        self.freq_save = freq_save
        self.effective_len_mc = (T_MC - T_BI) // freq_save

        self.N = N
        self.D = D

        self.list_idx_sampling = list_idx_sampling
        self.D_sampling = len(list_idx_sampling)

        self.max_workers = max_workers

    def read_data(self) -> np.ndarray:
        path_file = f"{self.path_data_csv_out_mcmc}/"
        path_file += f"estimation_Theta_{self.model_name}_MAP_mcmc.csv"

        df_Theta_MAP_lin = pd.read_csv(path_file, index_col=["n", "d"])
        df_Theta_MAP_lin = df_Theta_MAP_lin.sort_index()
        Theta_MAP_lin_full = df_Theta_MAP_lin["Theta_MAP_mcmc"].values.reshape(
            (self.N, self.D)
        )

        return Theta_MAP_lin_full

    def create_folders(self) -> str:
        folder_path_inter = f"{self.path_img}/distri_comp_yspace"
        folder_path = f"{folder_path_inter}/{self.model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return folder_path

    def main(
        self,
        list_mcmc_folders: List[str],
        scaler: Scaler,
        forward_map: ForwardMap,
        y: np.ndarray,
        omega: np.ndarray,
        sigma_a: np.ndarray,
        sigma_m: np.ndarray,
        list_lines: List[str],
        name_list_lines: str,
        point_challenger: Dict = {},
    ) -> None:
        global _plot_one_distribution_comparison

        assert name_list_lines in ["fit", "valid"]

        N_samples_per_chain = self.effective_len_mc // 20
        N_samples = N_samples_per_chain * self.N_MCMC
        rng = np.random.default_rng(42)

        L = y.shape[1]

        folder_path = self.create_folders()

        forward_map.restrict_to_output_subset(list_lines)

        print("starting plot comparison of distributions of y and f(theta)")

        Theta_MAP_lin_full = self.read_data()
        Theta_MAP_scaled_full = scaler.from_lin_to_scaled(Theta_MAP_lin_full)

        Theta_MAP_scaled = Theta_MAP_scaled_full[:, self.list_idx_sampling]

        f_Theta_MAP = forward_map.evaluate(Theta_MAP_scaled)

        if len(point_challenger) > 0:
            x_challenger_full = scaler.from_lin_to_scaled(point_challenger["value"])
            x_challenger = x_challenger_full[:, self.list_idx_sampling]
            f_Theta_challenger = forward_map.evaluate(x_challenger)

        # try:
        #     Theta_MLE_lin, _ = self.read_MLE_from_csv_file(model_name)
        #     Theta_MLE_scaled = self.scaler.from_lin_to_scaled(Theta_MLE_lin)
        #     f_Theta_MLE = forward_map.evaluate(Theta_MLE_scaled)
        # except:
        #     f_Theta_MLE = None

        def add_label(violin, label, list_labels):
            color = violin["bodies"][0].get_facecolor().flatten()
            list_labels.append((mpatches.Patch(color=color), label))
            return list_labels

        def _plot_one_distribution_comparison(n):
            list_theta_n_lin = np.zeros(
                (self.N_MCMC, N_samples_per_chain, self.D_sampling)
            )
            for seed, mc_path in enumerate(list_mcmc_folders):
                list_t = list(
                    rng.choice(
                        a=np.arange(
                            self.T_BI // self.freq_save,
                            self.T_MC // self.freq_save,
                        ),
                        size=N_samples_per_chain,
                        replace=False,
                    )
                )  # list
                list_t.sort()  # list

                with h5py.File(mc_path, "r") as f:
                    list_theta_n_lin[seed] = np.array(f["list_Theta"][list_t, n, :])

            list_theta_n_lin = list_theta_n_lin.transpose((2, 0, 1))
            list_theta_n_lin = list_theta_n_lin.reshape((self.D_sampling, N_samples)).T

            list_theta_n_lin_full = np.ones((N_samples, self.D))
            for idx_d, d in enumerate(self.list_idx_sampling):
                list_theta_n_lin_full[:, d] = list_theta_n_lin[:, idx_d]

            list_theta_n_scaled_full = scaler.from_lin_to_scaled(list_theta_n_lin_full)

            list_theta_n_scaled = list_theta_n_scaled_full[:, self.list_idx_sampling]

            list_f_Theta_n_lin = forward_map.evaluate(list_theta_n_scaled)  # (N_s, L)

            list_labels = []

            plt.figure(figsize=(12, 8))
            if self.N > 1:
                title = r"comparison of $f(\theta)$ and $y$ distributions for pixel"
                title += f" {n}"
                plt.title(title)
            else:
                plt.title(r"comparison of $f(\theta)$ and $y$ distributions")

            plt.xlabel("lines")
            plt.ylabel(r"$\log y$")

            n_std = 1

            l_y_add = None
            for ell in range(L):
                if y[n, ell] > omega[n, ell]:
                    list_theta = np.array([ell, ell, ell]) + 1
                    list_y = [
                        y[n, ell] - n_std * sigma_a[n, ell],
                        y[n, ell],
                        y[n, ell] + n_std * sigma_a[n, ell],
                    ]

                    (l_y_add,) = plt.plot(list_theta, list_y, "C2_-")

            l_y_multi = None
            for ell in range(L):
                if y[n, ell] > omega[n, ell]:
                    list_theta = np.array([ell, ell, ell]) + 1.15
                    list_y = [
                        np.exp(np.log(y[n, ell]) - (n_std * sigma_m[n, ell])),
                        y[n, ell],
                        np.exp(np.log(y[n, ell]) + (n_std * sigma_m[n, ell])),
                    ]

                    (l_y_multi,) = plt.plot(list_theta, list_y, "C1_-")

            list_labels = []
            if l_y_add is not None:
                list_labels.append((l_y_add, r"$Y \pm \sigma_a$"))
            if l_y_multi is not None:
                list_labels.append((l_y_multi, r"$Y \pm \sigma_m Y$"))

            list_labels = add_label(
                plt.violinplot(
                    list_f_Theta_n_lin,
                    positions=np.arange(1, L + 1) + 0.45,
                    widths=0.45,
                    showmeans=True,
                    showextrema=True,
                ),
                label=r"$f(Theta)$",
                list_labels=list_labels,
            )

            l_map = plt.scatter(
                np.arange(1, L + 1) + 0.45,
                f_Theta_MAP[n, :],
                marker="*",
                c="r",
                s=50,
            )
            list_labels += [(l_map, r"$f(\hat{x}_{MAP})$")]

            if len(point_challenger) > 0:
                l_challenger = plt.scatter(
                    np.arange(1, L + 1) + 0.45,
                    f_Theta_challenger[n, :],
                    marker="+",
                    c="k",
                    s=50,
                )
                list_labels += [(l_challenger, point_challenger["name"])]

            # if f_Theta_MLE is not None:
            #     l_mle = plt.scatter(
            #         np.arange(1, L + 1) + 0.45,
            #         f_Theta_MLE[n, :],
            #         marker="*",
            #         c="g",
            #         s=50,
            #     )
            #     list_labels += [(l_mle, r"$f(\hat{x}_{MLE})$")]

            l_omega = None
            for ell in range(L):
                if y[n, ell] <= omega[n, ell]:
                    (l_omega,) = plt.plot(
                        [1 + ell, 1 + ell + 0.8],
                        [omega[n, ell], omega[n, ell]],
                        "k--",
                    )

            if l_omega is not None:
                list_labels.append((l_omega, r"censor limit $\omega$"))

            list_lines_readable = [
                readable_line_names.lines_to_latex(line) for line in list_lines
            ]
            plt.xticks(np.arange(1, L + 1), list_lines_readable, rotation=90)
            plt.yscale("log")
            plt.ylim([min(0.1, list_f_Theta_n_lin.min() / 2), None])
            plt.grid()
            plt.legend(*zip(*list_labels))
            # plt.tight_layout()

            filename = f"{folder_path}/distribution_comparison_"
            filename += f"pix_{n}_{name_list_lines}.PNG"

            plt.savefig(filename, bbox_inches="tight")
            plt.close()
            return

        list_params = range(self.N)

        # set forward map device to cpu to make parallel computations possible
        device = forward_map.network.device * 1
        if "cuda" in forward_map.network.device:
            forward_map.network.set_device("cpu")

        # ? The parallel execution may fail on mac, even with the mp_context
        # ? argument. As I can't correct the error, in case of fail, I perform
        # ? the extraction in series, which is much slower.
        try:
            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=mp.get_context("fork")
            ) as p:
                _ = list(
                    tqdm(
                        p.map(_plot_one_distribution_comparison, list_params),
                        total=self.N,
                    )
                )

            # set back forward map device on its original value
            if "cuda" in device:
                forward_map.network.set_device(device)

        except:
            warnings.warn(
                "The parallel pixel-wise result extraction failed. Extracting in series instead."
            )

            # set back forward map device on its original value
            if "cuda" in device:
                forward_map.network.set_device(device)

            # if "cuda" in forward_map.network.device:
            for params in tqdm(list_params):
                _plot_one_distribution_comparison(params)

        print("plot comparison of distributions of y and f(theta) done")
        return
