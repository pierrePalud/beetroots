import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.inversion.results.abstract_results import ResultsExtractor
from beetroots.inversion.results.utils.perf_saver import EstimatorPerfSaver
from beetroots.space_transform.abstract_transform import Scaler


class ResultsExtractorOptimMLE(ResultsExtractor):
    """This class does not use the same utils as optim_MAP and MCMC to exploit the pixel-wise nature of the optimisation approach.

    .. warning::

        Unfinished class
    """

    def __init__(
        self,
        path_data_csv_out_mle: str,
        path_img: str,
        path_raw: str,
        N_MCMC: int,
        T_OPTI_MLE: int,
        freq_save: int,
        max_workers: int,
        N: int,
        D: int,
    ):
        r"""

        Parameters
        ----------
        path_data_csv_out_mle : str
            path to the csv file in which the performance of estimators is to be saved
        path_img : str
            path to the folder in which images are to be saved
        path_raw : str
            path to the raw ``.hdf5`` files
        N_MCMC : int
            number of optimization procedures to run per posterior distribution
        T_OPTI_MLE : int
            total size of each optimization procedure
        freq_save : int
            frequency of saved iterates, 1 means that all iterates were saved (used to show correct optimization procedure sizes in chain plots)
        max_workers : int
            maximum number of workers that can be used for results extraction
        N : int
            number of pixels / components :math:`n`, i.e., number of inverse problems that were solved
        D : int
            dimension of each vector of physical parameter :math:`\theta_n`
        """
        self.path_data_csv_out_mle = path_data_csv_out_mle
        r"""str: path to the csv file in which the performance of estimators is to be saved"""
        self.path_img = path_img
        r"""str: path to the folder in which images are to be saved"""
        self.path_raw = path_raw
        r"""str: path to the raw ``.hdf5`` files"""

        self.N_MCMC = N_MCMC
        r"""int: number of optimization procedures to run per posterior distribution"""
        self.T_OPTI_MLE = T_OPTI_MLE
        r"""int: total size of each optimization procedure"""
        self.freq_save = freq_save
        r"""int: frequency of saved iterates, 1 means that all iterates were saved (used to show correct optimization procedure sizes in chain plots)"""
        self.effective_T_MLE = self.T_OPTI_MLE // freq_save
        r"""int: effective length of the optimization procedures in ``.hdf5`` files"""

        self.N = N
        r"""number of pixels / components :math:`n`, i.e., number of inverse problems that were solved"""
        self.D = D
        r"""dimension of each vector of physical parameter :math:`\theta_n`"""

        self.max_workers = max_workers
        r"""int: maximum number of workers that can be used for results extraction"""

    @classmethod
    def read_estimator(
        cls,
        path_data_csv_out_mle: str,
        model_name: str,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """reads the value of an already estimated MLE from a csv file.

        Parameters
        ----------
        path_data_csv_out_optim_mle : str
            path to the csv file containing an already estimated MLE
        model_name : str
            name of the model, used to identify the posterior distribution

        Returns
        -------
        np.ndarray
            MLE estimator
        pd.DataFrame
            original DataFrame read from the csv file
        """
        path_file = f"{path_data_csv_out_mle}/results_MLE.csv"
        assert os.path.isfile(path_file), "The MLE has not been computed yet."

        df_results_mle = pd.read_csv(path_file)

        df_mle_best = (
            df_results_mle[df_results_mle["model_name"] == model_name]
            .groupby("n")["objective"]
            .min()
        )
        df_mle_best = df_mle_best.reset_index()

        df_mle_final = pd.merge(
            df_results_mle, df_mle_best, on=["n", "objective"], how="inner"
        )
        df_mle_final = df_mle_final.sort_values("n")
        df_mle_final = df_mle_final.drop_duplicates(["n", "objective"])

        N = df_mle_final["n"].max()
        D = df_mle_final["d"].max()

        Theta_MLE = np.zeros((N, D))
        for d in range(D):
            Theta_MLE[:, d] = df_mle_final.loc[:, f"x_MLE_{d}_lin"].values

        return Theta_MLE, df_mle_final

    def extract_mle_results(
        self,
        list_model_names: List[str],
        scaler: Scaler,
        Theta_true_scaled: Optional[np.ndarray] = None,
    ):
        global _extract_mle_results_1_pix

        if Theta_true_scaled is not None:
            Theta_true_lin = scaler.from_scaled_to_lin(Theta_true_scaled)

        def _extract_mle_results_1_pix(dict_input: dict) -> dict:
            n = dict_input["n"]
            seed = dict_input["seed"]
            model_name = dict_input["model_name"]

            path_results = f"{self.path_raw}/{model_name}/"
            path_results += f"opti_MLE_{seed}/pixel_{n}"

            with h5py.File(f"{path_results}/mc_chains.hdf5", "r") as f:
                list_Theta_lin = np.array(f["list_Theta"])  # (T, 1, D)
                list_objective = np.array(f["list_objective"])  # (T,)

            perf_saver = EstimatorPerfSaver()
            x_MLE_lin, objective = perf_saver.estimate_point_with_lowest_obj(
                list_Theta_lin,
                list_objective,
            )
            x_MLE_scaled = scaler.from_linear_to_scaled(
                x_MLE_lin.reshape((1, self.D)),
            )

            if Theta_true_scaled is not None:
                x_n_scaled_true = Theta_true_scaled[n, :].reshape((1, self.D))
                mse = perf_saver.compute_MSE(x_MLE_scaled, x_n_scaled_true)
                snr = perf_saver.compute_SNR(x_MLE_scaled, x_n_scaled_true)
            else:
                mse = None
                snr = None

            dict_output = {
                "n": n,
                "seed": seed,
                "model_name": model_name,
                "MSE": mse,
                "SNR": snr,
                "objective": objective,
            }
            for d in range(self.D):
                dict_output[f"x_MLE_{d}_lin"] = x_MLE_lin[0, d]
                if Theta_true_scaled is not None:
                    dict_output[f"Theta_true_{d}_lin"] = Theta_true_lin[n, d]

            return dict_output

        # * global function
        list_params = [
            {"n": n, "seed": seed, "model_name": model_name}
            for n in range(self.N)
            for seed in range(self.N_MCMC)
            for model_name in list_model_names
        ]
        with ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=mp.get_context("fork")
        ) as p:
            list_results = list(
                tqdm(
                    p.map(_extract_mle_results_1_pix, list_params),
                    total=len(list_params),
                )
            )

        path_file = f"{self.path_data_csv_out_optim_mle}/results_MLE.csv"
        df_results_mle = pd.DataFrame(list_results)
        df_results_mle.to_csv(path_file)
        return

    def evaluate_mle(
        self,
        list_model_names: List[str],
        scaler: Scaler,
        estimator_plot: PlotsEstimator,
        Theta_true_scaled: Optional[np.ndarray] = None,
    ):
        for model_name in list_model_names:
            Theta_MLE, df_mle_final = self.read_estimator(model_name)
            Theta_MLE_scaled = self.scaler.from_lin_to_scaled(Theta_MLE)

            perf_saver = EstimatorPerfSaver()
            if Theta_true_scaled is not None:
                mse_whole = perf_saver.compute_MSE(Theta_MLE_scaled, Theta_true_scaled)
                snr_whole = perf_saver.compute_SNR(Theta_MLE_scaled, Theta_true_scaled)
            else:
                mse_whole = None
                snr_whole = None

            objective_whole = df_mle_final["objective"].sum()

            # save estimator
            perf_saver.save_estimator_performance(
                self.path_data_csv_out_mle,
                "MLE",
                model_name,
                mse_whole,
                snr_whole,
                objective_whole,
            )
            if self.N > 1:  # and self.index_arr.size > 1:
                folder_path_inter = f"{self.path_img}/estimators"
                folder_path = f"{folder_path_inter}/{model_name}"
                folder_path_MLE = f"{folder_path}/MLE"
                for path_ in [folder_path_inter, folder_path, folder_path_MLE]:
                    if not os.path.isdir(path_):
                        os.mkdir(path_)

                estimator_plot.plot_estimator()
                estimator_plot.plot_estimator(
                    Theta_MLE,
                    "MLE",
                    folder_path_MLE,
                    model_name,
                )

                # self.plot_map_nll_of_estimator(
                #     Theta_MLE_scaled,
                #     "MLE",
                #     model_name,
                # )

        return

    def plot_objectives(self, list_model_names: List[str]):
        global _plot_objectives

        def _plot_objectives(dict_input: dict) -> bool:
            n = dict_input["n"]
            model_name = dict_input["model_name"]

            folder_path_inter = f"{self.path_img}/objective"
            folder_path_inter2 = f"{folder_path_inter}/{model_name}"
            folder_path = f"{folder_path_inter2}/MLE_objectives"

            list_objective_all_seeds = np.zeros(
                (self.N_MCMC, self.effective_T_MLE),
            )

            # read objectives
            for seed in range(self.N_MCMC):
                path_results = f"{self.path_raw}/{model_name}/"
                path_results += f"opti_MLE_{seed}/pixel_{n}"

                with h5py.File(f"{path_results}/mc_chains.hdf5", "r") as f:
                    list_objective_seed = np.array(f["list_objective"])  # (T,)
                    assert list_objective_seed.shape == (self.effective_T_MLE,)

                list_objective_all_seeds[seed] += list_objective_seed

            # plot objectives
            plt.figure(figsize=(8, 6))
            plt.title(f"Objective evolution during optimization for pixel {n}")
            for seed in range(self.N_MCMC):
                plt.plot(
                    range(self.effective_T_MLE),
                    list_objective_all_seeds[seed, :],
                )
            if list_objective_all_seeds.max() <= 0:
                plt.yscale("linear")
            elif list_objective_all_seeds.min() <= 0:
                plt.yscale("symlog")
            else:
                plt.yscale("log")
            plt.grid()
            # plt.tight_layout()
            plt.savefig(
                f"{folder_path}/objective_pixel_{n}.PNG",
                bbox_inches="tight",
            )
            plt.close()

            return

        # * global function
        list_params = [
            {"n": n, "seed": seed, "model_name": model_name}
            for n in range(self.N)
            for seed in range(self.N_MCMC)
            for model_name in list_model_names
        ]
        with ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=mp.get_context("fork")
        ) as p:
            _ = list(
                tqdm(
                    p.map(_plot_objectives, list_params),
                    total=len(list_params),
                )
            )
        return

    def main(
        self,
        list_model_names: List[str],
        scaler: Scaler,
        estimator_plot: PlotsEstimator,
        Theta_true_scaled: Optional[np.ndarray] = None,
    ) -> None:
        """Note: list_model_names = list(self.dict_posteriors.keys())"""

        print("starting MLE results extraction.")
        self.extract_mle_results(list_model_names, scaler, Theta_true_scaled)
        self.evaluate_mle(list_model_names, scaler, estimator_plot)
        self.plot_objectives(list_model_names)
        print("MLE results extraction done")
        return
