import os
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.inversion.results.utils.mc import histograms


class ResultsRegularizationWeights(ResultsUtil):

    __slots__ = (
        "model_name",
        "list_names",
        "path_img",
        "path_data_csv_out_mcmc",
        "N_MCMC",
        "T_MC",
        "T_BI",
        "freq_save",
        "effective_T_BI",
        "effective_T_MC",
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
        D: int,
        list_names: List[str],
    ):
        self.model_name = model_name
        self.list_names = list_names
        self.path_img = path_img
        self.path_data_csv_out_mcmc = path_data_csv_out_mcmc

        self.N_MCMC = N_MCMC
        self.T_MC = T_MC
        self.T_BI = T_BI
        self.freq_save = freq_save
        self.effective_T_BI = T_BI // freq_save
        self.effective_T_MC = T_MC // freq_save
        self.D = D

    def read_data(self, list_mcmc_folders: List[str]):
        list_tau = np.zeros((self.N_MCMC, self.effective_T_MC, self.D))
        for seed, mc_path in enumerate(list_mcmc_folders):
            with h5py.File(mc_path, "r") as f:
                # try:
                list_tau[seed] = np.array(f["list_tau"])
                # except:
                #     return np.empty((self.D,)), False

        return list_tau

    def create_folders(self):
        folder_path_inter = f"{self.path_img}/regularization_weights"
        folder_path = f"{folder_path_inter}/{self.model_name}"

        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return folder_path

    def estimate_regu_weight(self, list_mcmc_folders: List[str]) -> np.ndarray:
        for i, mc_path in enumerate(list_mcmc_folders):
            if i == 0:
                with h5py.File(mc_path, "r") as f:
                    list_tau = np.array(f["list_tau"][self.effective_T_BI :])

            else:
                with h5py.File(mc_path, "r") as f:
                    list_tau = np.concatenate(
                        [
                            list_tau,
                            np.array(
                                f["list_tau"][self.effective_T_BI :],
                            ),
                        ]
                    )
        estimated_regu_weights = list_tau.mean(0)
        return estimated_regu_weights

    def main(
        self,
        list_mcmc_folders: List[str],
    ) -> np.ndarray:
        folder_path = self.create_folders()
        list_tau = self.read_data(list_mcmc_folders)

        print("starting plots of regularization weights")

        for seed in range(self.N_MCMC):
            for d in range(self.D):
                list_tau_sd = list_tau[seed, :, d] * 1
                assert list_tau_sd.shape == (self.effective_T_MC,)

                list_tau_sd_no_BI = list_tau_sd[self.effective_T_BI :] * 1
                tau_MMSE = list_tau_sd_no_BI.mean(0)
                IC_2p5 = np.percentile(list_tau_sd_no_BI, q=2.5, axis=0)
                IC_97p5 = np.percentile(list_tau_sd_no_BI, q=97.5, axis=0)
                assert isinstance(tau_MMSE, float), tau_MMSE
                assert isinstance(IC_2p5, float), IC_2p5
                assert isinstance(IC_97p5, float), IC_97p5

                title = f"MC {self.list_names[d]} spatial regularization"
                title += " weight"
                histograms.plot_1D_chain(
                    list_tau_sd,
                    None,
                    d,
                    folder_path,
                    title,
                    lower_bounds_lin=1e-8 * np.ones((1,)),
                    upper_bounds_lin=1e8 * np.ones((1,)),
                    N_MCMC=self.N_MCMC,
                    T_MC=self.T_MC,
                    T_BI=self.T_BI,
                )

                title = "posterior distribution of spatial regularization"
                title += f" weight of {self.list_names[d]}"
                histograms.plot_1D_hist(
                    list_tau_sd_no_BI,
                    None,
                    d,
                    folder_path,
                    title=title,
                    lower_bounds_lin=1e-8 * np.ones((1,)),
                    upper_bounds_lin=1e8 * np.ones((1,)),
                    seed=seed,
                    estimator=tau_MMSE,
                    IC_low=IC_2p5,
                    IC_high=IC_97p5,
                )

        # altogether
        list_tau_flatter = list_tau.reshape(
            (self.N_MCMC * self.effective_T_MC, self.D),
        )
        list_tau_flatter_no_BI = list_tau[:, self.effective_T_BI :].reshape(
            (self.N_MCMC * (self.T_MC - self.T_BI) // self.freq_save, self.D)
        )

        tau_MMSE, _ = list_tau_sd_no_BI.mean(0)  # (D,)
        IC_2p5 = np.percentile(list_tau_sd_no_BI, q=2.5, axis=0)  # (D,)
        IC_97p5 = np.percentile(list_tau_sd_no_BI, q=97.5, axis=0)  # (D,)
        assert tau_MMSE.shape == (self.D,)
        assert IC_2p5.shape == (self.D,)
        assert IC_97p5.shape == (self.D,)

        plt.figure(figsize=(8, 6))
        plt.title("regularization weights sampling")
        for d, name in enumerate(self.list_names):
            plt.semilogy(list_tau_flatter[:, d], label=name)

        for seed in range(self.N_MCMC):
            if seed == 0:
                plt.axvline(
                    seed * self.effective_T_MC + self.effective_T_BI,
                    c="k",
                    ls="--",
                    label="T_BI",
                )
            elif seed == 1:
                plt.axvline(
                    seed * self.effective_T_MC,
                    c="k",
                    ls="-",
                    label="new MC",
                )
                plt.axvline(
                    seed * self.effective_T_MC + self.effective_T_BI,
                    c="k",
                    ls="--",
                )

            else:
                plt.axvline(seed * self.effective_T_MC, c="k", ls="-")
                plt.axvline(
                    seed * self.effective_T_MC + self.effective_T_BI,
                    c="k",
                    ls="--",
                )

        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{folder_path}/mc_regu_weights.PNG",
            bbox_inches="tight",
        )
        plt.close()

        for d in range(self.D):
            title = "posterior distribution of spatial regularization weight"
            title = f" of {self.list_names[d]}"
            histograms.plot_1D_hist(
                list_tau_flatter_no_BI[:, d],
                None,
                d,
                folder_path,
                title,
                self.lower_bounds_lin,
                self.upper_bounds_lin,
                None,
                tau_MMSE[d],
                IC_2p5[d],
                IC_97p5[d],
            )

        list_estimation_tau = []
        for d in range(self.D):
            dict_ = {
                "model_name": self.model_name,
                "d": d,
                "MMSE": tau_MMSE[d],
            }
            for q in [0.5, 2.5, 5, 95, 97.5, 99]:
                dict_[f"per_{q}"] = np.percentile(
                    list_tau_flatter_no_BI[:, d],
                    q=q,
                )
            list_estimation_tau.append(dict_)

        df_estimation_tau = pd.DataFrame(list_estimation_tau)
        path_file = f"{self.path_data_csv_out_mcmc}/estimation_tau.csv"
        df_estimation_tau.to_csv(
            path_file,
            mode="a",
            header=not (os.path.exists(path_file)),
        )
        print("plots of regularization weights done")
        return
