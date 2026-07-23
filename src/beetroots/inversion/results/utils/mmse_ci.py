import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.inversion.results.utils.perf_saver import EstimatorPerfSaver
from beetroots.modelling.component_distribution import ComponentDistribution

# from beetroots.modelling.target_distribution.abstract_target_distribution import (
#     TargetDistribution,
# )
# from beetroots.modelling.target_distribution.posterior import Posterior
from beetroots.space_transform.abstract_transform import Scaler


class ResultsMMSEandCIMyGibbsSampler(ResultsUtil):
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

    def read_data(self, dim_vars: dict) -> pd.DataFrame:
        path_overall_results, df_estim = dict(), dict()
        for k in dim_vars.keys():
            path_overall_results = os.path.join(
                self.path_data_csv_out_mcmc,
                k,
                f"estimation_Var_MMSE_CI_{self.model_name}.csv",
            )

            df_estim[k] = pd.read_csv(path_overall_results, index_col=["n", "d"])
            df_estim[k] = df_estim[k].sort_index().reset_index(drop=False)

            assert len(df_estim[k]) == np.prod(dim_vars[k])
        return df_estim

    def create_folders(
        self, keys: list[str], list_CI: List[int]
    ) -> Tuple[str, dict[int, str]]:
        folder_path_MMSE = dict.fromkeys(keys)
        folder_dict_path_CI = dict.fromkeys(keys)
        folder_path_inter = os.path.join(self.path_img, "estimators", self.model_name)
        for k in keys:
            folder_path_MMSE[k] = os.path.join(folder_path_inter, "MMSE", k)
            folder_dict_path_CI[k] = {
                ci: os.path.join(folder_path_inter, f"CI{ci}", k) for ci in list_CI
            }
            os.makedirs(folder_path_MMSE[k], exist_ok=True)
            for path_ in folder_dict_path_CI[k].values():
                os.makedirs(path_, exist_ok=True)

        return folder_path_MMSE, folder_dict_path_CI

    def main(
        self,
        components_objective: list[ComponentDistribution],
        scalers: dict[str, Scaler],
        estimator_plots: dict[str, PlotsEstimator],
        Vars_true_scaled_full: dict[str, np.ndarray],
        list_vars_idx_sampling: dict[str, List[int]],
        list_vars_fixed_values: dict[str, np.ndarray],
        list_CI: List[int],
    ) -> pd.DataFrame:
        print("starting MMSE and CI estimations and plots")

        dim_vars = {k: (self.N, len(v)) for k, v in list_vars_fixed_values.items()}

        df_estim = self.read_data(dim_vars)
        Vars_mmse_lin = dict.fromkeys(df_estim.keys())
        Vars_mmse_scaled_full = dict.fromkeys(df_estim.keys())
        mse, snr = dict.fromkeys(df_estim.keys()), dict.fromkeys(df_estim.keys())
        mse_component_wise, snr_component_wise = (
            dict.fromkeys(df_estim.keys()),
            dict.fromkeys(df_estim.keys()),
        )
        Vars_mmse_scaled = dict.fromkeys(df_estim.keys())
        for k in df_estim.keys():
            Vars_mmse_lin[k] = df_estim[k]["Var_MMSE"].values.reshape(
                Vars_true_scaled_full[k].shape
            )
            Vars_mmse_scaled_full[k] = scalers[k].from_lin_to_scaled(Vars_mmse_lin[k])

            # save its performance
            perf_saver = EstimatorPerfSaver()
            if Vars_true_scaled_full[k] is not None:
                mse[k] = perf_saver.compute_MSE(
                    Vars_mmse_scaled_full[k],
                    Vars_true_scaled_full[k],
                )
                snr[k] = perf_saver.compute_SNR(
                    Vars_mmse_scaled_full[k],
                    Vars_true_scaled_full[k],
                )
                mse_component_wise[k] = perf_saver.compute_MSE(
                    Vars_mmse_scaled_full[k],
                    Vars_true_scaled_full[k],
                    component_wise=True,
                )
                snr_component_wise[k] = perf_saver.compute_SNR(
                    Vars_mmse_scaled_full[k],
                    Vars_true_scaled_full[k],
                    component_wise=True,
                )
            else:
                mse[k] = None
                snr[k] = None
                mse_component_wise[k] = None
                snr_component_wise[k] = None

            # evaluate objective
            Vars_mmse_scaled[k] = Vars_mmse_scaled_full[k][:, list_vars_idx_sampling[k]]

        current = {k: {"var": Vars_mmse_scaled[k]} for k in Vars_mmse_scaled.keys()}
        objective_mmse = 0
        for comp in components_objective:
            comp.evaluate_all_nlpdf_utils(
                current=current,
                idx_pix=None,
                compute_derivatives=False,
                compute_derivatives_2nd_order=False,
                mtm=False,
            )
            objective_mmse += comp.neglog_pdf(chromatic_gibbs=False)

        folder_path_MMSE, folder_dict_path_CI = self.create_folders(
            df_estim.keys(), list_CI
        )

        for k in df_estim.keys():
            estimator_name_temp = "MMSE"
            EstimatorPerfSaver().save_estimator_performance(
                self.path_data_csv_out_mcmc,
                estimator_name_temp,
                self.model_name,
                mse[k],
                snr[k],
                mse_component_wise[k],
                snr_component_wise[k],
                objective_mmse,
                sub_var=k,
            )

            # plot it
            N, _ = Vars_mmse_lin[k].shape
            if N > 1 and estimator_plots[k] is not None:
                estimator_plots[k].plot_estimator(
                    Vars_mmse_lin[k],
                    estimator_name_temp,
                    folder_path_MMSE[k],
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

                        Var_percentile_lin = df_estim[k][per_name].values
                        Var_percentile_lin = Var_percentile_lin.reshape(
                            Vars_true_scaled_full[k].shape,
                        )

                        estimator_plots[k].plot_estimator(
                            Var_percentile_lin,
                            estimator_name,
                            folder_dict_path_CI[k][ci],
                            self.model_name,
                        )

                    per_low = dict_ci_per[ci]["lower"] * 1
                    per_low_name = f"per_{per_low:.1f}".replace(".", "p")

                    per_up = dict_ci_per[ci]["upper"] * 1
                    per_up_name = f"per_{per_up:.1f}".replace(".", "p")

                    df_ratio = df_estim[k][per_up_name] / df_estim[k][per_low_name]
                    ratio = df_ratio.values.reshape(Vars_true_scaled_full[k].shape)
                    estimator_plots[k].plot_CI_size(
                        ratio,
                        CI_name=f"{ci}% CI size",
                        folder_path=folder_dict_path_CI[k][ci],
                    )

                    estimator_plots[k].plot_CI_size(
                        np.sqrt(ratio),
                        CI_name=f"{ci}% CI uncertainty factor",
                        folder_path=folder_dict_path_CI[k][ci],
                    )

                # self.plot_map_nll_of_estimator(
                #     Var_MAP_scaled, estimator_name, model_name
                # )
        print("MMSE and CI estimations and plots done")
        return df_estim
