import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.inversion.results.utils.perf_saver import EstimatorPerfSaver
from beetroots.modelling.posterior import Posterior
from beetroots.space_transform.abstract_transform import Scaler


class ResultsMMSEandCIHierarchical(ResultsUtil):

    __slots__ = (
        "model_name",
        "path_img",
        "path_data_csv_out_mcmc",
        "N",
        "D",
        "L",
    )

    def __init__(
        self,
        model_name: str,
        path_img: str,
        path_data_csv_out_mcmc: str,
        N: int,
        D: int,
        L: int,
    ):
        self.model_name = model_name
        self.path_img = path_img
        self.path_data_csv_out_mcmc = path_data_csv_out_mcmc

        self.N = N
        self.D = D
        self.L = L

    def read_data(self) -> pd.DataFrame:
        path_overall_results = f"{self.path_data_csv_out_mcmc}/"
        path_overall_results += f"estimation_Theta_{self.model_name}.csv"

        df_estim = pd.read_csv(path_overall_results, index_col=["n", "d"])
        df_estim = df_estim.sort_index().reset_index(drop=False)

        path_overall_results_u = f"{self.path_data_csv_out_mcmc}/"
        path_overall_results_u += f"estimation_u_{self.model_name}.csv"

        df_estim_u = pd.read_csv(path_overall_results_u, index_col=["n", "ell"])
        df_estim_u = df_estim_u.sort_index().reset_index(drop=False)

        assert len(df_estim) == self.N * self.D
        assert (
            len(df_estim_u) == self.N * self.L
        ), f"is {len(df_estim_u)}, should be {self.N * self.L}"
        return df_estim, df_estim_u

    def create_folders(self) -> Tuple[str, str, str, str]:
        folder_path = f"{self.path_img}/estimators"
        folder_path_inter = f"{folder_path}/{self.model_name}"
        folder_path_MMSE = f"{folder_path_inter}/MMSE"
        folder_path_CI90 = f"{folder_path_inter}/CI90"
        folder_path_CI95 = f"{folder_path_inter}/CI95"
        folder_path_CI99 = f"{folder_path_inter}/CI99"

        for path_ in [
            folder_path,
            folder_path_inter,
            folder_path_MMSE,
            folder_path_CI90,
            folder_path_CI95,
            folder_path_CI99,
        ]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return (
            folder_path_MMSE,
            folder_path_CI90,
            folder_path_CI95,
            folder_path_CI99,
        )

    def main(
        self,
        conditional_theta: Posterior,
        scaler: Scaler,
        estimator_plot: Optional[PlotsEstimator],
        Theta_true_scaled: Optional[np.ndarray],
        list_lines: List[str],
    ) -> None:
        estimator_name = "MMSE"

        df_estim, df_estim_u = self.read_data()
        Theta_mmse_lin = df_estim["Theta_MMSE"].values.reshape((self.N, self.D))
        Theta_mmse_scaled = scaler.from_lin_to_scaled(Theta_mmse_lin)

        u_mmse_lin = df_estim_u["u_MMSE"].values.reshape((self.N, self.L))

        # save its performance
        perf_saver = EstimatorPerfSaver()
        if Theta_true_scaled is not None:
            mse = perf_saver.compute_MSE(Theta_mmse_scaled, Theta_true_scaled)
            snr = perf_saver.compute_SNR(Theta_mmse_scaled, Theta_true_scaled)
        else:
            mse = None
            snr = None

        # # evaluate objective
        # forward_map_evals = posterior.likelihood.evaluate_all_forward_map(
        #     Theta_mmse_scaled,
        #     True,
        # )
        # nll_utils = posterior.likelihood.evaluate_all_nll_utils(
        #     forward_map_evals,
        # )
        # objective_mmse = posterior.neglog_pdf(
        #     Theta_mmse_scaled, forward_map_evals, nll_utils
        # )
        objective_mmse = np.nan

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
                folder_path_CI90,
                folder_path_CI95,
                folder_path_CI99,
            ) = self.create_folders()

            estimator_plot.plot_estimator(
                Theta_mmse_lin,
                estimator_name,
                folder_path_MMSE,
                self.model_name,
            )
            estimator_plot.plot_estimator_u(
                u_mmse_lin,
                f"{estimator_name}_u",
                folder_path_MMSE,
                self.model_name,
                list_lines,
            )

            for percentile, estimator_name, folder_path in zip(
                [
                    "per_0p5",
                    "per_2p5",
                    "per_5",
                    "per_95",
                    "per_97p5",
                    "per_99p5",
                ],
                [
                    "percentile_0p5%",
                    "percentile_2p5%",
                    "percentile_5%",
                    "percentile_95%",
                    "percentile_97p5%",
                    "percentile_99p5%",
                ],
                [
                    folder_path_CI99,
                    folder_path_CI95,
                    folder_path_CI90,
                    folder_path_CI90,
                    folder_path_CI95,
                    folder_path_CI99,
                ],
            ):
                Theta_percentile_lin = df_estim[percentile].values
                Theta_percentile_lin = Theta_percentile_lin.reshape((self.N, self.D))

                u_percentile_lin = df_estim_u[percentile].values
                u_percentile_lin = u_percentile_lin.reshape((self.N, self.L))

                estimator_plot.plot_estimator(
                    Theta_percentile_lin,
                    estimator_name,
                    folder_path,
                    self.model_name,
                )
                estimator_plot.plot_estimator_u(
                    u_percentile_lin,
                    f"{estimator_name}_u",
                    folder_path,
                    self.model_name,
                    list_lines,
                )

            for per_low, per_high, CI_name, folder_path in zip(
                ["per_0p5", "per_2p5", "per_5"],
                ["per_99p5", "per_97p5", "per_95"],
                ["99%_CI_size", "95%_CI_size", "90%_CI_size"],
                [folder_path_CI99, folder_path_CI95, folder_path_CI90],
            ):
                df_ratio = df_estim[per_high] / df_estim[per_low]
                ratio = df_ratio.values.reshape((self.N, self.D))
                estimator_plot.plot_CI_size(
                    ratio,
                    CI_name,
                    folder_path,
                    self.model_name,
                )

                df_ratio_u = df_estim_u[per_high] / np.abs(df_estim_u[per_low])
                ratio_u = df_ratio_u.values.reshape((self.N, self.L))
                estimator_plot.plot_CI_size_u(
                    ratio_u,
                    CI_name,
                    folder_path,
                    self.model_name,
                    list_lines,
                )

            # self.plot_map_nll_of_estimator(
            #     Theta_MAP_scaled, estimator_name, model_name
            # )
        return df_estim, df_estim_u
