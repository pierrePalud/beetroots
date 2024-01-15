import os
from typing import List, Optional, Tuple, Union

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
    r"""extractor of inference results for the data of the optimization runs that that was saved."""
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
        r"""

        Parameters
        ----------
        path_data_csv_out_optim_map : str
            path to the csv file in which the performance of estimators is to be saved
        path_img : str
            path to the folder in which images are to be saved
        path_raw : str
            path to the raw ``.hdf5`` files
        N_MCMC : int
            number of optimization procedures to run per posterior distribution
        T_MC : int
            total size of each optimization procedure
        T_BI : int
            duration of the `Burn-in` phase
        freq_save : int
            frequency of saved iterates, 1 means that all iterates were saved (used to show correct optimization procedure sizes in chain plots)
        max_workers : int
            maximum number of workers that can be used for results extraction
        """
        self.path_data_csv_out_optim_map = path_data_csv_out_optim_map
        r"""str: path to the csv file in which the performance of estimators is to be saved"""
        self.path_img = path_img
        r"""str: path to the folder in which images are to be saved"""
        self.path_raw = path_raw
        r"""str: path to the raw ``.hdf5`` files"""

        self.N_MCMC = N_MCMC
        r"""int: number of optimization procedures to run per posterior distribution"""
        self.T_MC = T_MC
        r"""int: total size of each optimization procedure"""
        self.T_BI = T_BI
        r"""int: duration of the `Burn-in` phase"""
        self.freq_save = freq_save
        r"""int: frequency of saved iterates, 1 means that all iterates were saved (used to show correct optimization procedure sizes in chain plots)"""

        self.max_workers = max_workers
        r"""int: maximum number of workers that can be used for results extraction"""

    @classmethod
    def read_estimator(
        cls,
        path_data_csv_out_optim_map: str,
        model_name: str,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        r"""reads the value of an already estimated MAP from a csv file.

        Parameters
        ----------
        path_data_csv_out_optim_map : str
            path to the csv file containing an already estimated MAP
        model_name : str
            name of the model, used to identify the posterior distribution

        Returns
        -------
        np.ndarray
            MAP estimator
        pd.DataFrame
            original DataFrame read from the csv file
        """
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
        list_fixed_values: Union[List[float], np.ndarray],
        #
        estimator_plot: PlotsEstimator,
        Theta_true_scaled: Optional[np.ndarray] = None,
    ):
        r"""performs the data extraction, in this order:

        * step 1 : clppd
        * step 2 : kernel analysis
        * step 3 : objective evolution
        * step 4 : MAP estimator from samples
        * step 5 : model checking with Bayesian p-value

        Parameters
        ----------
        posterior : Posterior
            probability distribution. The goal of the optimization procedure was to find its mode, i.e., the minimum of its negative log pdf.
        model_name : str
            name of the model, used to identify the posterior distribution
        scaler : Scaler
            contains the transformation of the Theta values from their natural space to their scaled space (in which the sampling happens) and its inverse
        list_idx_sampling : List[int]
            indices of the physical parameters that were sampled (the other ones were fixed)
        list_fixed_values : List[float]
            list of used values for the parameters fixed during the sampling
        estimator_plot : PlotsEstimator
            object used to plot the estimator figures
        Theta_true_scaled : Optional[np.ndarray], optional
            true value for the inferred physical parameter :math:`\Theta` (only possible for toy cases), by default None
        """
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
                compute_derivatives=False,
                compute_derivatives_2nd_order=False,
            )
            nll_utils = posterior.likelihood.evaluate_all_nll_utils(
                forward_map_evals,
                compute_derivatives=False,
                compute_derivatives_2nd_order=False,
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
