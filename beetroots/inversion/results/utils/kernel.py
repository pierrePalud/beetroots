import os
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

from beetroots.inversion.results.utils.abstract_util import ResultsUtil

# TODO: for now, assumes that the Sampler has a MTM kernel and a PMALA kernel
# -> generalize ?


class ResultsKernels(ResultsUtil):

    __slots__ = (
        "model_name",
        "chain_type",
        "path_img",
        "N_run",
        "effective_T",
    )

    def __init__(
        self,
        model_name: str,
        chain_type: str,
        path_img: str,
        N_run: int,
        T: int,
        freq_save: int,
    ):
        assert chain_type in ["mcmc", "optim_map", "optim_mle"]

        self.model_name = model_name
        self.chain_type = chain_type
        self.path_img = path_img

        self.N_run = N_run
        self.effective_T = T // freq_save

    def read_data(
        self,
        list_chains_folders: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        list_type = np.zeros((self.N_run, self.effective_T))
        list_accepted = np.zeros((self.N_run, self.effective_T))
        list_log_proba = np.zeros((self.N_run, self.effective_T))

        for seed, mc_path in enumerate(list_chains_folders):
            with h5py.File(mc_path, "r") as f:
                list_type[seed] = np.array(f["list_type_t"])
                list_accepted[seed] = np.array(f["list_accepted_t"])
                list_log_proba[seed] = np.array(f["list_log_proba_accept_t"])

        return list_type, list_accepted, list_log_proba

    def create_folders(self) -> Tuple[str, str]:
        folder_path_inter = f"{self.path_img}/accepted_freq"
        folder_path_inter2 = f"{folder_path_inter}/{self.chain_type}"
        folder_path_accept_freq = f"{folder_path_inter2}/{self.model_name}"

        for path_ in [
            folder_path_inter,
            folder_path_inter2,
            folder_path_accept_freq,
        ]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        folder_path_inter = f"{self.path_img}/log_proba_accept"
        folder_path_inter2 = f"{folder_path_inter}/{self.chain_type}"
        folder_path_log_p_accept = f"{folder_path_inter2}/{self.model_name}"
        for path_ in [
            folder_path_inter,
            folder_path_inter2,
            folder_path_log_p_accept,
        ]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return (folder_path_accept_freq, folder_path_log_p_accept)

    def plot_accept_freq(
        self,
        folder_path: str,
        list_type: np.ndarray,
        list_accepted: np.ndarray,
    ) -> None:
        # mobile mean size
        k_mm_mtm = 20  # MTM
        k_mm_mala = 20  # P-MALA

        print("starting plot of accepted frequencies")

        # * MTM kernel
        for seed in range(self.N_run):
            idx_mtm = list_type[seed] == 0
            accepted_mtm = list_accepted[seed, idx_mtm]

            if accepted_mtm.size > k_mm_mtm:
                accepted_mtm_smooth = np.convolve(
                    accepted_mtm,
                    np.ones(k_mm_mtm) / k_mm_mtm,
                    mode="valid",
                )

                plt.figure(figsize=(8, 6))
                nan_mean = 100 * np.nanmean(accepted_mtm)
                plt.title(f"MTM : {nan_mean:.2f} % accepted")
                plt.plot(accepted_mtm_smooth, label="mobile mean")
                plt.grid()
                plt.legend()
                plt.xticks(rotation=45)
                # plt.tight_layout()

                filename = f"{folder_path}/freq_accept_seed{seed}_MTM.PNG"
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

        # * PMALA
        for seed in range(self.N_run):
            idx_pmala = list_type[seed] == 1
            accepted_pmala = list_accepted[seed, idx_pmala]

            if accepted_pmala.size > k_mm_mala:
                accepted_pmala_smooth = np.convolve(
                    accepted_pmala,
                    np.ones(k_mm_mala) / k_mm_mala,
                    mode="valid",
                )

                plt.figure(figsize=(8, 6))
                nan_mean = 100 * np.nanmean(accepted_pmala)
                plt.title(f"PMALA : {nan_mean:.2f} % accepted")
                plt.plot(accepted_pmala_smooth, label="mobile mean")
                plt.grid()
                plt.legend()
                plt.xticks(rotation=45)
                # plt.tight_layout()

                filename = f"{folder_path}/freq_accept_seed{seed}_PMALA.PNG"
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

        print("plots of accepted frequencies done")
        return

    def plot_log_proba_accept(
        self,
        folder_path: str,
        list_type: np.ndarray,
        list_log_proba: np.ndarray,
    ) -> None:
        """plots log proba accept per kernel"""

        # mobile mean size
        k_mm_mtm = 20  # MTM
        k_mm_mala = 20  # P-MALA

        print("starting plot of log proba accept")

        # * MTM
        for seed in range(self.N_run):
            idx_mtm = list_type[seed] == 0
            list_log_proba_mtm = list_log_proba[seed, idx_mtm]

            if list_log_proba_mtm.size > k_mm_mtm:
                list_log_proba_mtm_smooth = np.convolve(
                    list_log_proba_mtm,
                    np.ones(k_mm_mtm) / k_mm_mtm,
                    mode="valid",
                )

                plt.figure(figsize=(8, 6))

                nan_mean = np.nanmean(list_log_proba_mtm)
                nan_median = np.nanmedian(list_log_proba_mtm)
                title = f"MTM: log proba accept avg: {nan_mean:.3e},"
                title += f"median: {nan_median:.3e}"
                plt.title(title)
                plt.plot(list_log_proba_mtm_smooth, label="mobile mean")
                # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
                plt.grid()
                plt.legend()
                plt.yscale("symlog")
                plt.xticks(rotation=45)
                # plt.tight_layout()

                filename = f"{folder_path}/log_proba_accept_seed{seed}_MTM.PNG"
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

        # * PMALA
        for seed in range(self.N_run):
            idx_pmala = list_type[seed] == 1
            list_log_proba_pmala = list_log_proba[seed, idx_pmala]

            if list_log_proba_pmala.size > k_mm_mala:
                list_log_proba_pmala_smooth = np.convolve(
                    list_log_proba_pmala,
                    np.ones(k_mm_mala) / k_mm_mala,
                    mode="valid",
                )

                plt.figure(figsize=(8, 6))

                nan_mean = np.nanmean(list_log_proba_mtm)
                nan_median = np.nanmedian(list_log_proba_mtm)
                title = f"PMALA: log proba accept avg: {nan_mean:.3e},"
                title += f"median: {nan_median:.3e}"

                plt.title(title)
                plt.plot(list_log_proba_pmala_smooth, label="mobile mean")
                # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
                plt.grid()
                plt.legend()
                plt.yscale("symlog")
                plt.xticks(rotation=45)
                # plt.tight_layout()

                filename = f"{folder_path}/log_proba_accept_"
                filename += f"seed{seed}_PMALA.PNG"
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

        print("plots of log proba accept done")
        return

    def main(self, list_chains_folders: List[str]) -> None:
        list_type, list_accepted, list_log_proba = self.read_data(
            list_chains_folders,
        )
        (folder_accept_freq, folder_log_proba_accept) = self.create_folders()

        self.plot_accept_freq(folder_accept_freq, list_type, list_accepted)
        self.plot_log_proba_accept(
            folder_log_proba_accept,
            list_type,
            list_log_proba,
        )
