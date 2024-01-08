import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.inversion.results.abstract_results import ResultsExtractor
from beetroots.inversion.results.utils.bayes_pval_plots import ResultsBayesPvalues
from beetroots.inversion.results.utils.clppd import ResultsCLPPD
from beetroots.inversion.results.utils.kernel import ResultsKernels
from beetroots.inversion.results.utils.lowest_obj_estimator import (
    ResultsLowestObjective,
)
from beetroots.inversion.results.utils.objective import ResultsObjective
from beetroots.modelling.posterior import Posterior
from beetroots.space_transform.abstract_transform import Scaler


class ResultsExtractorOptimMAP(ResultsExtractor):

    __slots__ = (
        "path_data_csv_out_optim_map",
        "path_img",
        "path_raw",
        "N_MCMC",
        "T_MC",
        "T_BI",
        "freq_save",
        "max_workers",
    )

    def __init__(
        self,
        path_data_csv_out_optim_map: str,
        path_img: str,
        path_raw: str,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        freq_save: int,
        max_workers: int,
    ):
        self.path_data_csv_out_optim_map = path_data_csv_out_optim_map
        self.path_img = path_img
        self.path_raw = path_raw

        self.N_MCMC = N_MCMC
        self.T_MC = T_MC
        self.T_BI = T_BI
        self.freq_save = freq_save

        self.max_workers = max_workers

    @classmethod
    def read_estimator(
        cls,
        path_data_csv_out_optim_map: str,
        model_name: str,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        path_file = f"{path_data_csv_out_optim_map}/"
        path_file += f"estimation_Theta_{model_name}_MAP_optim_map.csv"
        assert os.path.isfile(path_file), f"no MAP at {path_file}"

        df_results_map = pd.read_csv(path_file)
        df_results_map = df_results_map.sort_values(["n", "d"])

        N = df_results_map["n"].max() + 1
        D = df_results_map["d"].max() + 1

        Theta_MAP = np.zeros((N, D))
        for d in range(D):
            Theta_MAP[:, d] = df_results_map.loc[
                df_results_map["d"] == d, "Theta_MAP_optim_map"
            ].values

        return Theta_MAP, df_results_map

    def main(
        self,
        posterior: Posterior,
        model_name: str,
        scaler: Scaler,
        #
        list_idx_sampling: List[int],
        list_fixed_values: List[float],
        #
        estimator_plot: PlotsEstimator,
        Theta_true_scaled: Optional[np.ndarray] = None,
    ):
        list_mcmc_folders = [
            f"{x[0]}/mc_chains.hdf5"
            for x in os.walk(f"{self.path_raw}/{model_name}")
            if "optim_MAP_" in x[0]
        ]
        list_mcmc_folders.sort()

        N = posterior.N * 1
        D_sampling = posterior.D * 1
        D = len(list_fixed_values)
        L = posterior.L

        list_fixed_values_scaled = list(
            posterior.likelihood.forward_map.arr_fixed_values
        )
        list_fixed_values_scaled = [v if v != 0 else None for v in list_fixed_values]

        chain_type = "optim_map"

        # clppd
        ResultsCLPPD(
            model_name=model_name,
            chain_type=chain_type,
            path_img=self.path_img,
            path_data_csv_out=self.path_data_csv_out_optim_map,
            N_MCMC=self.N_MCMC,
            N=N,
            L=L,
        ).main(list_mcmc_folders, estimator_plot.map_shaper)

        # kernel analysis
        ResultsKernels(
            model_name,
            chain_type,
            self.path_img * 1,
            self.N_MCMC * 1,
            self.T_MC * 1,
            self.freq_save * 1,
        ).main(list_mcmc_folders)

        # objective evolution
        if Theta_true_scaled is not None:
            forward_map_evals = posterior.likelihood.evaluate_all_forward_map(
                Theta_true_scaled,
                True,
            )
            nll_utils = posterior.likelihood.evaluate_all_nll_utils(
                forward_map_evals,
            )
            objective_true = posterior.neglog_pdf(
                Theta_true_scaled, forward_map_evals, nll_utils
            )
        else:
            objective_true = None

        idx_lowest_obj, lowest_obj = ResultsObjective(
            model_name,
            chain_type,
            self.path_img * 1,
            self.N_MCMC * 1,
            self.T_MC * 1,
            self.T_BI * 1,
            self.freq_save * 1,
            N=N,
            D=D,
            L=L,
        ).main(list_mcmc_folders, objective_true)

        # MAP estimator from samples
        if Theta_true_scaled is not None:
            Theta_true_scaled_full = np.zeros((N, D))
            Theta_true_scaled_full[:, list_idx_sampling] += Theta_true_scaled

            for d in range(D):
                if list_fixed_values_scaled[d] is not None:
                    Theta_true_scaled_full[:, d] += list_fixed_values_scaled[d]
        else:
            Theta_true_scaled_full = None

        ResultsLowestObjective(
            model_name,
            chain_type,
            self.path_img,
            self.path_data_csv_out_optim_map,
            self.N_MCMC,
            self.T_MC,
            self.freq_save,
        ).main(
            list_mcmc_folders,
            lowest_obj,
            idx_lowest_obj,
            scaler,
            Theta_true_scaled_full,  # (N, D_sampling)
            list_idx_sampling,  # (D_sampling,)
            list_fixed_values,  # (D,)
            estimator_plot,
        )

        ResultsBayesPvalues(
            model_name=model_name,
            chain_type=chain_type,
            path_img=self.path_img,
            path_data_csv_out=self.path_data_csv_out_optim_map,
            N_MCMC=self.N_MCMC,
            N=N,
            D_sampling=D_sampling,
            plot_ESS=True,
        ).main(
            list_idx_sampling=list_idx_sampling,
            map_shaper=estimator_plot.map_shaper if N > 1 else None,
        )
