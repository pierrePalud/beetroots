import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class EstimatorPerfSaver:

    __slots__ = ()

    def __init__(self):
        pass

    @staticmethod
    def compute_MSE(Theta_estimate: np.ndarray, Theta_true: np.ndarray):
        return np.linalg.norm(Theta_true - Theta_estimate) ** 2

    @staticmethod
    def compute_SNR(Theta_estimate: np.ndarray, Theta_true: np.ndarray):
        denom = np.linalg.norm(Theta_true) ** 2
        mse_ = EstimatorPerfSaver.compute_MSE(Theta_estimate, Theta_true)
        return -10 * np.log10(mse_ / denom)

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
        objective: Optional[float],
    ) -> None:
        path_overall_results = f"{path_data_csv_out}/results_overall.csv"

        list_results_overall = [
            {
                "estimator": estimator_name,
                "model_name": model_name,
                "MSE": mse,
                "SNR": snr,
                "objective": objective,
            }
        ]

        df_results_overall = pd.DataFrame(list_results_overall)
        df_results_overall.to_csv(
            path_overall_results,
            mode="a",
            header=not (os.path.exists(path_overall_results)),
        )
