import os
from typing import List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.inversion.results.utils.abstract_util import ResultsUtil


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
        T_BI: int,
        freq_save: int,
        sub_var: Union[str, None] = "",
    ):
        assert chain_type in ["mcmc", "optim_map", "optim_mle"]

        self.model_name = model_name
        self.chain_type = chain_type
        self.path_img = path_img

        self.sub_var = sub_var

        self.N_run = N_run
        self.effective_T = T // freq_save
        self.T_BI = T_BI

    def read_data(
        self,
        list_chains_folders: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # list_type = np.zeros((self.N_run, self.effective_T), dtype=np.dtypes.StrDType)
        # list_accepted = np.zeros((self.N_run, self.effective_T))
        # list_log_proba = np.zeros((self.N_run, self.effective_T))
        list_type = []
        list_accepted = []
        list_log_proba = []

        for seed, mc_path in enumerate(list_chains_folders):
            with h5py.File(mc_path, "r") as f:
                if self.sub_var:
                    f = f[
                        self.sub_var
                    ]  # If more than one variable sampled, select the right one
                # list_type[seed] = np.array(f["list_type_t"])
                # list_accepted[seed] = np.array(f["list_accepted_t"])
                # list_log_proba[seed] = np.array(f["list_log_proba_accept_t"])
                list_type.append(np.array(f["list_type_t"]))
                list_accepted.append(np.array(f["list_accepted_t"]))
                list_log_proba.append(np.array(f["list_log_proba_accept_t"]))

        return (
            np.array(list_type).astype(str),
            np.array(list_accepted),
            np.array(list_log_proba),
        )

    def create_folders(self) -> Tuple[str, str]:
        folder_path_inter = os.path.join(self.path_img, "accepted_freq")
        folder_path_inter2 = os.path.join(folder_path_inter, self.chain_type)
        folder_path_accept_freq = os.path.join(
            folder_path_inter2, self.model_name, self.sub_var
        )

        if not os.path.isdir(folder_path_accept_freq):
            os.makedirs(folder_path_accept_freq)

        folder_path_inter = os.path.join(self.path_img, "log_proba_accept")
        folder_path_inter2 = os.path.join(folder_path_inter, self.chain_type)
        folder_path_log_p_accept = os.path.join(
            folder_path_inter2, self.model_name, self.sub_var
        )

        if not os.path.isdir(folder_path_log_p_accept):
            os.makedirs(folder_path_log_p_accept)

        return (folder_path_accept_freq, folder_path_log_p_accept)

    def plot_accept_freq(
        self,
        folder_path: str,
        list_type: np.ndarray,
        list_accepted: np.ndarray,
        map_shaper: MapShaper,
    ) -> None:
        # mobile mean size
        k_mm = 20

        print("starting plot of accepted frequencies")

        for seed in range(self.N_run):
            type_kernels = np.unique(list_type[seed])
            for kernel_name in type_kernels:
                idx_kernel = list_type[seed] == kernel_name
                idx_kernel_BI = (list_type[seed] == kernel_name) & (
                    self.T_BI > np.arange(list_accepted.shape[1])
                )
                accepted_kernel = list_accepted[seed, idx_kernel]

                if accepted_kernel.ndim > 1:
                    # Plot 2D pixelwise acceptance
                    accepted_kernel_pixelwise_with_BI = np.mean(accepted_kernel, axis=0)
                    accepted_kernel_pixelwise_no_BI = np.mean(
                        list_accepted[seed, idx_kernel_BI], axis=0
                    )
                    # With BI
                    plt.figure(figsize=(8, 6))
                    plt.title(f"Pixelwise acceptance (with BI):\n{kernel_name}")
                    plt.imshow(
                        map_shaper.from_vector_to_map(
                            accepted_kernel_pixelwise_with_BI
                        ),
                        vmin=0,
                        vmax=1,
                        cmap="gray",
                        origin="lower",
                    )
                    plt.colorbar()
                    filename = os.path.join(
                        folder_path,
                        f"freq_accept_pixelwise_with-BI_seed{seed}_{kernel_name}.PNG",
                    )
                    plt.savefig(filename, bbox_inches="tight")
                    plt.close()
                    # No BI
                    plt.figure(figsize=(8, 6))
                    plt.title(f"Pixelwise acceptance (no BI):\n{kernel_name}")
                    plt.imshow(
                        map_shaper.from_vector_to_map(accepted_kernel_pixelwise_no_BI),
                        vmin=0,
                        vmax=1,
                        cmap="gray",
                        origin="lower",
                    )
                    plt.colorbar()
                    filename = os.path.join(
                        folder_path,
                        f"freq_accept_pixelwise_no-BI_seed{seed}_{kernel_name}.PNG",
                    )
                    plt.savefig(filename, bbox_inches="tight")
                    plt.close()

                    # Prepare 1D acceptance plot
                    accepted_kernel = np.mean(
                        accepted_kernel, axis=tuple(range(1, accepted_kernel.ndim))
                    )

                if accepted_kernel.size > k_mm:
                    accepted_kernel_smooth = np.convolve(
                        accepted_kernel,
                        np.ones(k_mm) / k_mm,
                        mode="valid",
                    )

                    plt.figure(figsize=(8, 6))
                    nan_mean = 100 * np.nanmean(accepted_kernel)
                    plt.title(f"{kernel_name} : {nan_mean:.2f} % accepted")
                    plt.plot(accepted_kernel_smooth, label="mobile mean")
                    plt.grid()
                    plt.legend()
                    plt.xticks(rotation=45)
                    # plt.tight_layout()

                    filename = os.path.join(
                        folder_path, f"freq_accept_seed{seed}_{kernel_name}.PNG"
                    )
                    plt.savefig(filename, bbox_inches="tight")
                    plt.close()

            if list_accepted[seed].ndim > 1:
                # With BI
                plt.figure(figsize=(8, 6))
                plt.title("Pixelwise acceptance (with BI):\nall kernels")
                plt.imshow(
                    map_shaper.from_vector_to_map(list_accepted[seed].mean(axis=0)),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                    origin="lower",
                )
                plt.colorbar()
                filename = os.path.join(
                    folder_path,
                    f"freq_accept_pixelwise_with-BI_seed{seed}_all-kernels.PNG",
                )
                plt.savefig(filename, bbox_inches="tight")
                plt.close()
                # No BI
                plt.figure(figsize=(8, 6))
                plt.title("Pixelwise acceptance (no BI):\nall kernels")
                plt.imshow(
                    map_shaper.from_vector_to_map(
                        list_accepted[seed, self.T_BI :].mean(axis=0)
                    ),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                    origin="lower",
                )
                plt.colorbar()
                filename = os.path.join(
                    folder_path,
                    f"freq_accept_pixelwise_no-BI_seed{seed}_all-kernels.PNG",
                )
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
        k_mm = 20

        print("starting plot of log proba accept")

        for seed in range(self.N_run):
            type_kernels = np.unique(list_type[seed])
            for kernel_name in type_kernels:
                idx_kernel = list_type[seed] == kernel_name
                list_log_proba_kernel = list_log_proba[seed, idx_kernel]
                if list_log_proba_kernel.ndim > 1:
                    list_log_proba_kernel = np.mean(
                        list_log_proba_kernel,
                        axis=tuple(range(1, list_log_proba_kernel.ndim)),
                    )

                if list_log_proba_kernel.size > k_mm:
                    list_log_proba_kernel_smooth = np.convolve(
                        list_log_proba_kernel,
                        np.ones(k_mm) / k_mm,
                        mode="valid",
                    )

                    plt.figure(figsize=(8, 6))

                    nan_mean = np.nanmean(list_log_proba_kernel)
                    nan_median = np.nanmedian(list_log_proba_kernel)
                    title = f"{kernel_name}: log proba accept avg: {nan_mean:.3e},"
                    title += f"median: {nan_median:.3e}"
                    plt.title(title)
                    plt.plot(list_log_proba_kernel_smooth, label="mobile mean")
                    # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
                    plt.grid()
                    plt.legend()
                    plt.yscale("symlog")
                    plt.xticks(rotation=45)
                    # plt.tight_layout()

                    filename = os.path.join(
                        folder_path, f"log_proba_accept_seed{seed}_{kernel_name}.PNG"
                    )
                    plt.savefig(filename, bbox_inches="tight")
                    plt.close()

        print("plots of log proba accept done")
        return

    def main(self, list_chains_folders: List[str], map_shaper: MapShaper) -> None:
        list_type, list_accepted, list_log_proba = self.read_data(
            list_chains_folders,
        )
        folder_accept_freq, folder_log_proba_accept = self.create_folders()

        self.plot_accept_freq(folder_accept_freq, list_type, list_accepted, map_shaper)
        self.plot_log_proba_accept(
            folder_log_proba_accept,
            list_type,
            list_log_proba,
        )
