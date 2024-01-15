import os
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

from beetroots.inversion.results.utils.abstract_util import ResultsUtil


class ResultsObjective(ResultsUtil):

    __slots__ = (
        "model_name",
        "chain_type",
        "path_img",
        "N_MCMC",
        "effective_T_MC",
        "effective_T_BI",
    )

    def __init__(
        self,
        model_name: str,
        chain_type: str,
        path_img: str,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        freq_save: int,
        N: int,
        D: int,
        L: int,
    ):
        assert chain_type in ["mcmc", "optim_map", "optim_mle"]

        self.model_name = model_name
        self.chain_type = chain_type
        self.path_img = path_img

        self.N_MCMC = N_MCMC
        self.effective_T_MC = T_MC // freq_save
        self.effective_T_BI = T_BI // freq_save

        self.N = N
        self.D = D
        self.L = L

    def read_data(
        self, list_chains_folders: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        list_objective = np.zeros((self.N_MCMC, self.effective_T_MC))
        list_nll = np.zeros((self.N_MCMC, self.effective_T_MC))
        # list_nl_prior_indicator = np.zeros((self.N_MCMC, self.effective_T_MC))
        # list_nl_prior_spatial = np.zeros((self.N_MCMC, self.effective_T_MC))

        for seed, mc_path in enumerate(list_chains_folders):
            with h5py.File(mc_path, "r") as f:
                list_objective[seed] = np.array(f["list_objective"])
                list_nll[seed] = np.array(f["list_nll"])
                # if "list_nl_prior_indicator" in f:
                #     list_nl_prior_indicator[seed] = np.array(
                #         f["list_nl_prior_indicator"]
                #     )
                # if "list_nl_prior_spatial" in f:
                #     list_nl_prior_spatial[seed] = np.array(f["list_nl_prior_spatial"])

        return list_objective, list_nll

    def create_folders(self) -> str:
        folder_path_inter = f"{self.path_img}/objective"
        folder_path = f"{folder_path_inter}/{self.model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)
        return folder_path

    def plot_objective(
        self,
        folder_path: str,
        list_objective: np.ndarray,
        list_nll: np.ndarray,
        # list_nl_prior_indicator: np.ndarray,
        # list_nl_prior_spatial: np.ndarray,
        objective_true: Optional[float],
    ):
        filename_prefix = f"{folder_path}/{self.chain_type}_objective"

        list_objective_flat = list_objective.flatten()
        list_objective_no_BI_flat = list_objective[:, self.effective_T_BI :]
        list_objective_no_BI_flat = list_objective_no_BI_flat.flatten()

        list_nll_flat = list_nll.flatten()
        list_nll_no_BI_flat = list_nll[:, self.effective_T_BI :]
        list_nll_no_BI_flat = list_nll_no_BI_flat.flatten()

        # list_list_nl_prior_indicator_flat = []
        # list_list_nl_prior_indicator_no_BI_flat = []
        # for d in range(self.D):
        #     list_nl_prior_indicator_flat = list_nl_prior_indicator[:, :, d].flatten()

        #     list_nl_prior_indicator_no_BI_flat = list_nl_prior_indicator[
        #         :, self.effective_T_BI :, d
        #     ]
        #     list_nl_prior_indicator_no_BI_flat = (
        #         list_nl_prior_indicator_no_BI_flat.flatten()
        #     )

        # list_nl_prior_spatial_flat = list_nl_prior_spatial.flatten()
        # list_nl_prior_spatial_no_BI_flat = list_nl_prior_spatial[
        #     :, self.effective_T_BI :
        # ]
        # list_nl_prior_spatial_no_BI_flat = list_nl_prior_spatial_no_BI_flat.flatten()

        # * With Burn in
        plt.figure(figsize=(8, 6))
        plt.title("Objective evolution")
        plt.plot(list_objective_flat, label="objective")
        plt.plot(list_nll_flat, label="nll")

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
                    label="new run",
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

        if list_objective.max() <= 0:
            plt.yscale("linear")
        elif list_objective.min() < 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")
        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{filename_prefix}_with_bi.PNG",
            bbox_inches="tight",
        )

        if objective_true is not None:
            plt.axhline(objective_true, c="r", ls="--", label="obj Theta_true")
            plt.legend()
            plt.savefig(
                f"{filename_prefix}_with_bi_and_true.PNG",
                bbox_inches="tight",
            )
        # if self.Theta_true_scaled is not None:
        #     forward_map_evals = self.dict_posteriors[
        #         model_name
        #     ].likelihood.evaluate_all_forward_map(self.Theta_true_scaled, True)
        #     nll_utils = self.dict_posteriors[
        #         model_name
        #     ].likelihood.evaluate_all_nll_utils(forward_map_evals)
        #     objective_true = self.dict_posteriors[model_name].neglog_pdf(
        #         self.Theta_true_scaled, forward_map_evals, nll_utils
        #     )
        #     plt.axhline(objective_true, c="r", ls="--", label="obj Theta_true")
        #     plt.legend()
        #     plt.savefig(
        #         f"{filename_prefix}_with_bi_and_true.PNG",
        #         bbox_inches="tight",
        #     )
        plt.close()

        # * Without Burn in
        plt.figure(figsize=(8, 6))
        plt.title("Objective evolution (no Burn-In)")
        plt.plot(list_objective_no_BI_flat, label="objective")
        plt.plot(list_nll_no_BI_flat, label="nll")

        for seed in range(1, self.N_MCMC):
            if seed == 1:
                plt.axvline(
                    seed * (self.effective_T_MC - self.effective_T_BI),
                    c="k",
                    ls="-",
                    label="new run",
                )
            else:
                plt.axvline(
                    seed * (self.effective_T_MC - self.effective_T_BI),
                    c="k",
                    ls="-",
                )

        if list_objective.max() < 0:
            plt.yscale("linear")
        elif list_objective.min() < 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")

        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{filename_prefix}_no_bi.PNG",
            bbox_inches="tight",
        )

        if objective_true is not None:
            plt.axhline(objective_true, c="r", ls="--", label="obj Theta_true")
            plt.legend()
            plt.savefig(
                f"{filename_prefix}_no_bi_with_true.PNG",
                bbox_inches="tight",
            )
        plt.close()
        return

    def plot_normalized_nll(
        self,
        folder_path: str,
        list_nll: np.ndarray,
    ) -> None:
        filename_prefix = f"{folder_path}/{self.chain_type}_nll"

        list_normalized_nll_flat = list_nll.flatten() / (self.N * self.L)
        list_normalized_nll_no_BI_flat = list_nll[:, self.effective_T_BI :] / (
            self.N * self.L
        )
        list_normalized_nll_no_BI_flat = list_normalized_nll_no_BI_flat.flatten()

        # * With Burn in
        plt.figure(figsize=(8, 6))
        plt.title("Normalized neglog likelihood evolution")
        plt.plot(list_normalized_nll_flat, label="nll")

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
                    label="new run",
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

        if list_nll.max() <= 0:
            plt.yscale("linear")
        elif list_nll.min() < 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")
        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{filename_prefix}_with_bi.PNG",
            bbox_inches="tight",
        )
        plt.close()

        # * Without Burn in
        plt.figure(figsize=(8, 6))
        plt.title("Normalized neglog likelihood evolution (no Burn-In)")
        plt.plot(list_normalized_nll_no_BI_flat, label="nll")

        for seed in range(1, self.N_MCMC):
            if seed == 1:
                plt.axvline(
                    seed * (self.effective_T_MC - self.effective_T_BI),
                    c="k",
                    ls="-",
                    label="new run",
                )
            else:
                plt.axvline(
                    seed * (self.effective_T_MC - self.effective_T_BI),
                    c="k",
                    ls="-",
                )

        if list_nll.max() < 0:
            plt.yscale("linear")
        elif list_nll.min() < 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")

        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{filename_prefix}_no_bi.PNG",
            bbox_inches="tight",
        )
        plt.close()
        return

    def find_lowest_objective(
        self,
        list_objective: np.ndarray,
    ) -> Tuple[int, float]:
        # compute index of sampling MAP
        list_objective_no_BI = list_objective.flatten()
        idx_lowest_obj = int(np.argmin(list_objective_no_BI))
        lowest_obj = np.min(list_objective_no_BI)
        return idx_lowest_obj, lowest_obj

    def main(
        self,
        list_chains_folders: List[str],
        objective_true: Optional[float],
    ) -> Tuple[int, float]:
        list_objective, list_nll = self.read_data(list_chains_folders)
        folder_objective = self.create_folders()

        print("starting plot of objective function")
        self.plot_objective(
            folder_objective,
            list_objective,
            list_nll,
            objective_true,
        )
        self.plot_normalized_nll(folder_objective, list_nll)
        print("plot of objective function done")

        idx_lowest_obj, lowest_obj = self.find_lowest_objective(list_objective)
        return idx_lowest_obj, lowest_obj
