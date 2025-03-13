import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class EstimatorPerfSaver:
    __slots__ = ()

    def __init__(self):
        pass

    @staticmethod
    def compute_MSE(
        Theta_estimate: np.ndarray, Theta_true: np.ndarray, component_wise: bool = False
    ):
        if component_wise:
            return np.linalg.norm((Theta_true - Theta_estimate), axis=0) ** 2
        else:
            return np.linalg.norm(Theta_true - Theta_estimate) ** 2

    @staticmethod
    def compute_SNR(
        Theta_estimate: np.ndarray, Theta_true: np.ndarray, component_wise: bool = False
    ):
        if component_wise:
            denom = np.linalg.norm(Theta_true, axis=0) ** 2
            mse_ = EstimatorPerfSaver.compute_MSE(
                Theta_estimate, Theta_true, component_wise=True
            )
        else:
            denom = np.linalg.norm(Theta_true) ** 2
            mse_ = EstimatorPerfSaver.compute_MSE(Theta_estimate, Theta_true)
        res = -10 * np.log10(mse_ / denom)
        return res

    @staticmethod
    def estimate_point_with_lowest_obj(
        list_Theta_lin: np.ndarray,
        list_objective: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        idx_best = np.argmin(list_objective)
        Theta_estimate = list_Theta_lin[idx_best]
        objective_estimate = list_objective[idx_best]
        return Theta_estimate, objective_estimate

    def save_estimator_performance(
        self,
        path_data_csv_out: str,
        estimator_name: str,
        model_name: str,
        mse: Optional[float],
        snr: Optional[float],
        mse_component_wise: Optional[np.ndarray],
        snr_component_wise: Optional[np.ndarray],
        objective: Optional[float],
    ) -> None:
        path_overall_results = f"{path_data_csv_out}/results_overall.csv"
        path_component_wise_results = f"{path_data_csv_out}/results_component_wise.csv"

        list_results_overall = [
            {
                "estimator": estimator_name,
                "model_name": model_name,
                "MSE": mse,
                "SNR": snr,
                "objective": objective,
            }
        ]

        dict_results_component_wise = {
            "estimator": estimator_name,
            "model_name": model_name,
            "MSE": mse,
            "SNR": snr,
            "objective": objective,
        }
        if mse_component_wise is not None:
            dict_results_component_wise.update(
                {f"MSE_{i}": mse_i for i, mse_i in enumerate(mse_component_wise)}
            )
        if snr_component_wise is not None:
            dict_results_component_wise.update(
                {f"SNR_{i}": snr_i for i, snr_i in enumerate(snr_component_wise)}
            )
        list_results_component_wise = [dict_results_component_wise]

        df_results_overall = pd.DataFrame(list_results_overall)
        df_results_overall.to_csv(
            path_overall_results,
            mode="a",
            header=not (os.path.exists(path_overall_results)),
        )

        df_results_component_wise = pd.DataFrame(list_results_component_wise)
        df_results_component_wise.to_csv(
            path_component_wise_results,
            mode="a",
            header=not (os.path.exists(path_component_wise_results)),
        )
