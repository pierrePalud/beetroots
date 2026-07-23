import os
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd

from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.inversion.results.utils.perf_saver import EstimatorPerfSaver
from beetroots.space_transform.abstract_transform import Scaler


class ResultsLowestObjective(ResultsUtil):
    """Plot the lowest objective estimator (e.g. MAP or MLE).
    Can be used for Posterior models.
    """

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
        sub_var: Optional[str] = "",
    ):
        assert chain_type in ["mcmc", "optim_map", "optim_mle"]

        self.model_name = model_name
        self.chain_type = chain_type
        self.path_img = path_img
        self.path_data_csv_out = path_data_csv_out

        self.sub_var = sub_var

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
                    Var_lowest_obj_lin = np.array(f["list_var"][idx_inchain])

        return Var_lowest_obj_lin

    def create_folders(self) -> str:
        folder_path = os.path.join(
            self.path_img,
            "estimators",
            self.model_name,
            f"MAP_{self.chain_type}",
            self.sub_var,
        )
        os.makedirs(folder_path, exist_ok=True)

        return folder_path

    def save_estimator_to_csv(
        self, Var_lowest_obj_lin_full: np.ndarray, estimator_name: str
    ) -> None:
        # save estimator
        path_overall_results = os.path.join(
            self.path_data_csv_out,
            self.sub_var,
            f"estimation_var_{self.model_name}_{estimator_name}.csv",
        )

        N, D = Var_lowest_obj_lin_full.shape

        df_MAP = pd.DataFrame()
        nn, dd = np.meshgrid(np.arange(N), np.arange(D))
        df_MAP["n"] = nn.astype(int).flatten()
        df_MAP["d"] = dd.astype(int).flatten()
        df_MAP = df_MAP.sort_values(by=["n", "d"])
        df_MAP[f"Var_MAP_objective_{self.chain_type}"] = (
            Var_lowest_obj_lin_full.flatten()
        )
        df_MAP.to_csv(path_overall_results)
        return

    def main(
        self,
        list_chains_folders: List[str],
        lowest_obj: float,
        idx_lowest_obj: int,
        scaler: Scaler,
        _full: Optional[np.ndarray],
        list_idx_sampling: List[int],
        list_fixed_values: np.ndarray,
        estimator_plot: Optional[PlotsEstimator],
    ) -> None:
        print("starting plot of lowest obj")

        estimator_name = f"MAP_{self.chain_type}"

        Var_lowest_obj_lin = self.read_data(
            list_chains_folders,
            idx_lowest_obj,
        )

        D = len(list_fixed_values)
        N, D_sampling = Var_lowest_obj_lin.shape

        Var_lowest_obj_lin_full = np.zeros((N, D))
        Var_lowest_obj_lin_full[:, list_idx_sampling] += Var_lowest_obj_lin

        for d in range(D):
            if list_fixed_values[d] is not None:
                Var_lowest_obj_lin_full[:, d] += list_fixed_values[d]

        Var_lowest_obj_scaled_full = scaler.from_lin_to_scaled(
            Var_lowest_obj_lin_full,
        )

        # save the estimator itself
        self.save_estimator_to_csv(Var_lowest_obj_lin_full, estimator_name)

        # save its performance
        perf_saver = EstimatorPerfSaver()
        if _full is not None:
            mse = perf_saver.compute_MSE(
                Var_lowest_obj_scaled_full,
                _full,
            )
            snr = perf_saver.compute_SNR(
                Var_lowest_obj_scaled_full,
                _full,
            )
            assert isinstance(mse, float)
            assert isinstance(snr, float)

            mse_component_wise = perf_saver.compute_MSE(
                Var_lowest_obj_scaled_full,
                _full,
                component_wise=True,
            )
            snr_component_wise = perf_saver.compute_SNR(
                Var_lowest_obj_scaled_full,
                _full,
                component_wise=True,
            )
            assert isinstance(mse_component_wise, np.ndarray)
            assert isinstance(snr_component_wise, np.ndarray)
        else:
            mse = None
            snr = None
            mse_component_wise = None
            snr_component_wise = None

        perf_saver.save_estimator_performance(
            self.path_data_csv_out,
            estimator_name,
            self.model_name,
            mse,
            snr,
            mse_component_wise,
            snr_component_wise,
            lowest_obj,
        )

        # plot it
        N, _ = Var_lowest_obj_lin.shape
        if N > 1 and estimator_plot is not None:
            folder_lowest_obj = self.create_folders()
            estimator_plot.plot_estimator(
                Var_lowest_obj_lin_full,
                estimator_name,
                folder_lowest_obj,
                self.model_name,
            )
            # self.plot_map_nll_of_estimator(
            #     Var_MAP_scaled, estimator_name, model_name
            # )

        print("plot of lowest obj done")

        return


class ResultsLowestObjectiveGlobal(ResultsUtil):
    """Plot the lowest objective estimator (e.g. MAP or MLE).
    Used for hierarchical models.
    """

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
        var_names: List[str],
    ):
        assert chain_type in ["mcmc", "optim_map", "optim_mle"]

        self.model_name = model_name
        self.chain_type = chain_type
        self.path_img = path_img
        self.path_data_csv_out = path_data_csv_out

        self.var_names = var_names

        self.N_run = N_run
        self.effective_T = T // freq_save

    def read_data(
        self,
        list_chains_folders: List[str],
        idx_lowest_obj: int,
    ) -> dict:
        idx_chain = idx_lowest_obj // self.effective_T
        idx_inchain = idx_lowest_obj % self.effective_T

        assert 0 <= idx_chain < self.N_run
        assert 0 <= idx_inchain < self.effective_T

        Vars_lowest_obj_lin = dict.fromkeys(self.var_names)
        for seed, mc_path in enumerate(list_chains_folders):
            if seed == idx_chain:
                with h5py.File(mc_path, "r") as f:
                    for key in self.var_names:
                        Vars_lowest_obj_lin[key] = np.array(
                            f[key]["list_var"][idx_inchain]
                        )

        return Vars_lowest_obj_lin

    def create_folders(self) -> str:
        folder_path_init = f"{self.path_img}/estimators"
        folder_path_inter = f"{folder_path_init}/{self.model_name}"
        folder_path = os.path.join(folder_path_inter, f"MAP_global_{self.chain_type}")
        for key in self.var_names:
            folder_path_temp = os.path.join(folder_path, key)
            os.makedirs(folder_path_temp, exist_ok=True)

            os.makedirs(
                os.path.join(self.path_data_csv_out, key), exist_ok=True
            )  # create the folder for the csv files

        return folder_path

    def save_estimator_to_csv(
        self,
        Var_lowest_obj_lin_full: np.ndarray,
        estimator_name: str,
        key: str,
    ) -> None:
        path_overall_results = os.path.join(
            self.path_data_csv_out,
            key,
            f"estimation_var_{self.model_name}_{estimator_name}.csv",
        )

        N, D = Var_lowest_obj_lin_full.shape

        df_MAP = pd.DataFrame()
        nn, dd = np.meshgrid(np.arange(N), np.arange(D))
        df_MAP["n"] = nn.astype(int).flatten()
        df_MAP["d"] = dd.astype(int).flatten()
        df_MAP = df_MAP.sort_values(by=["n", "d"])
        df_MAP[f"Var_MAP_{self.chain_type}"] = Var_lowest_obj_lin_full.flatten()
        df_MAP.to_csv(path_overall_results)
        return

    def main(
        self,
        list_chains_folders: List[str],
        lowest_obj: float,
        idx_lowest_obj: int,
        scalers: dict[str, Scaler],
        Vars_true_scaled_full: Optional[np.ndarray],
        list_vars_idx_sampling: dict[str, List[int]],
        list_vars_fixed_values: dict[str, np.ndarray],
        estimator_plots: dict[str, PlotsEstimator],
    ) -> None:
        print("starting plot of global lowest obj")

        estimator_name = f"MAP_global_{self.chain_type}"

        Vars_lowest_obj_lin = self.read_data(
            list_chains_folders,
            idx_lowest_obj,
        )
        Vars_lowest_obj_scaled_full = dict.fromkeys(self.var_names)
        folder_lowest_obj = self.create_folders()
        for key in self.var_names:
            D = len(list_vars_fixed_values[key])
            N, D_sampling = Vars_lowest_obj_lin[key].shape

            Var_lowest_obj_lin_full = np.zeros((N, D))
            Var_lowest_obj_lin_full[
                :, list_vars_idx_sampling[key]
            ] += Vars_lowest_obj_lin[key]

            for d in range(D):
                if list_vars_fixed_values[key][d] is not None:
                    Var_lowest_obj_lin_full[:, d] += list_vars_fixed_values[key][d]

            Vars_lowest_obj_scaled_full[key] = scalers[key].from_lin_to_scaled(
                Var_lowest_obj_lin_full,
            )

            # save the estimators themselves
            self.save_estimator_to_csv(Var_lowest_obj_lin_full, estimator_name, key)

            # save its performance
            perf_saver = EstimatorPerfSaver()
            if Vars_true_scaled_full[key] is not None:
                mse = perf_saver.compute_MSE(
                    Vars_lowest_obj_scaled_full[key],
                    Vars_true_scaled_full[key],
                )
                snr = perf_saver.compute_SNR(
                    Vars_lowest_obj_scaled_full[key],
                    Vars_true_scaled_full[key],
                )
                mse_component_wise = perf_saver.compute_MSE(
                    Vars_lowest_obj_scaled_full[key],
                    Vars_true_scaled_full[key],
                    component_wise=True,
                )
                snr_component_wise = perf_saver.compute_SNR(
                    Vars_lowest_obj_scaled_full[key],
                    Vars_true_scaled_full[key],
                    component_wise=True,
                )
            else:
                mse = None
                snr = None
                mse_component_wise = None
                snr_component_wise = None

            perf_saver.save_estimator_performance(
                self.path_data_csv_out,
                estimator_name,
                self.model_name,
                mse,
                snr,
                mse_component_wise,
                snr_component_wise,
                lowest_obj,
                sub_var=key,
            )

            # plot it
            N, _ = Vars_lowest_obj_lin[key].shape
            if N > 1 and estimator_plots[key] is not None:
                folder_lowest_obj_temp = os.path.join(folder_lowest_obj, key)
                estimator_plots[key].plot_estimator(
                    Var_lowest_obj_lin_full,
                    estimator_name,
                    folder_lowest_obj_temp,
                    self.model_name,
                )
                # self.plot_map_nll_of_estimator(
                #     Var_MAP_scaled, estimator_name, model_name
                # )

        print("plot of global lowest obj done")

        return
