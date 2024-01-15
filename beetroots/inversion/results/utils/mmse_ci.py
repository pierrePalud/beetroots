import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.inversion.results.utils.perf_saver import EstimatorPerfSaver
from beetroots.modelling.posterior import Posterior
from beetroots.space_transform.abstract_transform import Scaler


class ResultsMMSEandCI(ResultsUtil):

    __slots__ = (
        "model_name",
        "path_img",
        "path_data_csv_out_mcmc",
        "N",
        "D",
    )

    def __init__(
        self,
        model_name: str,
        path_img: str,
        path_data_csv_out_mcmc: str,
        N: int,
        D: int,
    ):
        self.model_name = model_name
        self.path_img = path_img
        self.path_data_csv_out_mcmc = path_data_csv_out_mcmc

        self.N = N
        self.D = D

    def read_data(self) -> pd.DataFrame:
        path_overall_results = f"{self.path_data_csv_out_mcmc}/"
        path_overall_results += f"estimation_Theta_{self.model_name}.csv"

        df_estim = pd.read_csv(path_overall_results, index_col=["n", "d"])
        df_estim = df_estim.sort_index().reset_index(drop=False)

        assert len(df_estim) == self.N * self.D
        return df_estim

    def create_folders(self, list_CI: List[int]) -> Tuple[str, dict[int, str]]:
        folder_path = f"{self.path_img}/estimators"
        folder_path_inter = f"{folder_path}/{self.model_name}"
        folder_path_MMSE = f"{folder_path_inter}/MMSE"

        dict_folder_path_CI = {ci: f"{folder_path_inter}/CI{ci}" for ci in list_CI}

        for path_ in [
            folder_path,
            folder_path_inter,
            folder_path_MMSE,
        ] + list(dict_folder_path_CI.values()):
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return folder_path_MMSE, dict_folder_path_CI

    def main(
        self,
        posterior: Posterior,
        scaler: Scaler,
        estimator_plot: Optional[PlotsEstimator],
        Theta_true_scaled_full: Optional[np.ndarray],
        list_idx_sampling: List[int],
        list_fixed_values: np.ndarray,
        list_CI: List[int],
    ) -> pd.DataFrame:
        estimator_name = "MMSE"

        df_estim = self.read_data()
        Theta_mmse_lin = df_estim["Theta_MMSE"].values.reshape((self.N, self.D))
        Theta_mmse_scaled_full = scaler.from_lin_to_scaled(Theta_mmse_lin)

        # save its performance
        perf_saver = EstimatorPerfSaver()
        if Theta_true_scaled_full is not None:
            mse = perf_saver.compute_MSE(
                Theta_mmse_scaled_full,
                Theta_true_scaled_full,
            )
            snr = perf_saver.compute_SNR(
                Theta_mmse_scaled_full,
                Theta_true_scaled_full,
            )
        else:
            mse = None
            snr = None

        # evaluate objective
        Theta_mmse_scaled = Theta_mmse_scaled_full[:, list_idx_sampling]

        forward_map_evals = posterior.likelihood.evaluate_all_forward_map(
            Theta_mmse_scaled,
            compute_derivatives=False,
            compute_derivatives_2nd_order=False,
        )
        nll_utils = posterior.likelihood.evaluate_all_nll_utils(
            forward_map_evals,
            compute_derivatives=False,
            compute_derivatives_2nd_order=False,
        )
        objective_mmse = posterior.neglog_pdf(
            Theta_mmse_scaled, forward_map_evals, nll_utils
        )

        perf_saver.save_estimator_performance(
            self.path_data_csv_out_mcmc,
            estimator_name,
            self.model_name,
            mse,
            snr,
            objective_mmse,
        )

        # plot it
        N, _ = Theta_mmse_lin.shape
        if N > 1 and estimator_plot is not None:
            (
                folder_path_MMSE,
                dict_folder_path_CI,
            ) = self.create_folders(list_CI)

            estimator_plot.plot_estimator(
                Theta_mmse_lin,
                estimator_name,
                folder_path_MMSE,
                self.model_name,
            )

            dict_ci_per = {
                ci: {"lower": (100 - ci) / 2, "upper": 100 - (100 - ci) / 2}
                for ci in list_CI
            }
            dict_per_ci = {}
            for ci in list_CI:
                per_lower = dict_ci_per[ci]["lower"] * 1
                per_upper = dict_ci_per[ci]["upper"] * 1
                dict_per_ci[per_lower] = ci * 1
                dict_per_ci[per_upper] = ci * 1

            list_per = list(dict_per_ci.keys())
            list_per.sort()

            for ci in list_CI:
                for percentile in dict_ci_per[ci].values():
                    estimator_name = f"percentile {percentile:.1f}%"
                    per_name = f"per_{percentile:.1f}".replace(".", "p")

                    Theta_percentile_lin = df_estim[per_name].values
                    Theta_percentile_lin = Theta_percentile_lin.reshape(
                        (self.N, self.D),
                    )

                    estimator_plot.plot_estimator(
                        Theta_percentile_lin,
                        estimator_name,
                        dict_folder_path_CI[ci],
                        self.model_name,
                    )

                per_low = dict_ci_per[ci]["lower"] * 1
                per_low_name = f"per_{per_low:.1f}".replace(".", "p")

                per_up = dict_ci_per[ci]["upper"] * 1
                per_up_name = f"per_{per_up:.1f}".replace(".", "p")

                df_ratio = df_estim[per_up_name] / df_estim[per_low_name]
                ratio = df_ratio.values.reshape((self.N, self.D))
                estimator_plot.plot_CI_size(
                    ratio,
                    CI_name=f"{ci}% CI size",
                    folder_path=dict_folder_path_CI[ci],
                )

                estimator_plot.plot_CI_size(
                    np.sqrt(ratio),
                    CI_name=f"{ci}% CI uncertainty factor",
                    folder_path=dict_folder_path_CI[ci],
                )

            # self.plot_map_nll_of_estimator(
            #     Theta_MAP_scaled, estimator_name, model_name
            # )
        return df_estim
