import os
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd

from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.inversion.results.utils.perf_saver import EstimatorPerfSaver
from beetroots.space_transform.abstract_transform import Scaler


class ResultsLowestObjectiveHierarchical(ResultsUtil):

    __slots__ = (
        "model_name",
        "chain_type",
        "path_img",
        "path_data_csv_out",
        "N_run",
        "effective_T",
    )

    def __init__(
        self,
        model_name: str,
        chain_type: str,
        path_img: str,
        path_data_csv_out: str,
        N_run: int,
        T: int,
        freq_save: int,
    ):
        assert chain_type in ["mcmc", "optim_map", "optim_mle"]

        self.model_name = model_name
        self.chain_type = chain_type
        self.path_img = path_img
        self.path_data_csv_out = path_data_csv_out

        self.N_run = N_run
        self.effective_T = T // freq_save

    def read_data(
        self,
        list_chains_folders: List[str],
        idx_lowest_obj: int,
    ) -> np.ndarray:
        idx_chain = idx_lowest_obj // self.effective_T
        idx_inchain = idx_lowest_obj % self.effective_T

        assert 0 <= idx_chain < self.N_run
        assert 0 <= idx_inchain < self.effective_T

        for seed, mc_path in enumerate(list_chains_folders):
            if seed == idx_chain:
                with h5py.File(mc_path, "r") as f:
                    Theta_lowest_obj_lin = np.array(f["list_Theta"][idx_inchain])
                    u_lowest_obj_lin = np.array(f["list_U"][idx_inchain])

        return Theta_lowest_obj_lin, u_lowest_obj_lin

    def create_folders(self) -> str:
        folder_path_init = f"{self.path_img}/estimators"
        folder_path_inter = f"{folder_path_init}/{self.model_name}"
        folder_path = f"{folder_path_inter}/MAP_{self.chain_type}"
        for path_ in [folder_path_init, folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return folder_path

    def save_estimator_to_csv(
        self, Theta_lowest_obj_lin: np.ndarray, estimator_name: str
    ) -> None:
        # save estimator
        path_overall_results = f"{self.path_data_csv_out}/"
        path_overall_results += f"estimation_Theta_{self.model_name}_"
        path_overall_results += f"{estimator_name}.csv"

        N, D = Theta_lowest_obj_lin.shape

        df_MAP = pd.DataFrame()
        nn, dd = np.meshgrid(np.arange(N), np.arange(D))
        df_MAP["n"] = nn.astype(int).flatten()
        df_MAP["d"] = dd.astype(int).flatten()
        df_MAP = df_MAP.sort_values(by=["n", "d"])
        df_MAP[f"Theta_MAP_{self.chain_type}"] = Theta_lowest_obj_lin.flatten()
        df_MAP.to_csv(path_overall_results)
        return

    def main(
        self,
        list_chains_folders: List[str],
        lowest_obj: float,
        idx_lowest_obj: int,
        scaler: Scaler,
        Theta_true_scaled: Optional[np.ndarray],
        estimator_plot: Optional[PlotsEstimator],
        list_lines_fit: List[str],
    ) -> None:
        estimator_name = f"MAP_{self.chain_type}"

        Theta_lowest_obj_lin, u_lowest_obj_lin = self.read_data(
            list_chains_folders,
            idx_lowest_obj,
        )
        Theta_lowest_obj_scaled = scaler.from_lin_to_scaled(Theta_lowest_obj_lin)

        # save the estimator itself
        self.save_estimator_to_csv(Theta_lowest_obj_lin, estimator_name)

        # save its performance
        perf_saver = EstimatorPerfSaver()
        if Theta_true_scaled is not None:
            mse = perf_saver.compute_MSE(Theta_lowest_obj_scaled, Theta_true_scaled)
            snr = perf_saver.compute_SNR(Theta_lowest_obj_scaled, Theta_true_scaled)
        else:
            mse = None
            snr = None

        perf_saver.save_estimator_performance(
            self.path_data_csv_out,
            estimator_name,
            self.model_name,
            mse,
            snr,
            lowest_obj,
        )

        # plot it
        N, _ = Theta_lowest_obj_lin.shape
        if N > 1 and estimator_plot is not None:
            folder_lowest_obj = self.create_folders()
            estimator_plot.plot_estimator(
                Theta_lowest_obj_lin,
                estimator_name,
                folder_lowest_obj,
                self.model_name,
            )
            estimator_plot.plot_estimator_u(
                u_lowest_obj_lin,
                f"{estimator_name}_u",
                folder_lowest_obj,
                self.model_name,
                list_lines_fit,
            )
            # self.plot_map_nll_of_estimator(
            #     Theta_MAP_scaled, estimator_name, model_name
            # )
        return
