import copy
import os
from typing import Dict, List, Optional

import numpy as np

from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.inversion.results.abstract_results import ResultsExtractor
from beetroots.inversion.results.utils.bayes_pval_plots import ResultsBayesPvalues
from beetroots.inversion.results.utils.clppd import ResultsCLPPD
from beetroots.inversion.results.utils.ess_plots import ResultsESS
from beetroots.inversion.results.utils.kernel import ResultsKernels
from beetroots.inversion.results.utils.lowest_obj_estimator import (
    ResultsLowestObjective,
)
from beetroots.inversion.results.utils.mc import ResultsMC
from beetroots.inversion.results.utils.mmse_ci import ResultsMMSEandCI
from beetroots.inversion.results.utils.objective import ResultsObjective
from beetroots.inversion.results.utils.regularization_weights import (
    ResultsRegularizationWeights,
)
from beetroots.inversion.results.utils.valid_mc import ResultsValidMC
from beetroots.inversion.results.utils.y_f_Theta import (
    ResultsDistributionComparisonYandFTheta,
)
from beetroots.modelling.posterior import Posterior
from beetroots.space_transform.abstract_transform import Scaler


class ResultsExtractorMCMC(ResultsExtractor):

    __slots__ = (
        "path_data_csv_out_mcmc",
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
        path_data_csv_out_mcmc: str,
        path_img: str,
        path_raw: str,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        freq_save: int,
        max_workers: int,
    ):
        self.path_data_csv_out_mcmc = path_data_csv_out_mcmc
        self.path_img = path_img
        self.path_raw = path_raw

        self.N_MCMC = N_MCMC
        self.T_MC = T_MC
        self.T_BI = T_BI
        self.freq_save = freq_save

        self.max_workers = max_workers

    def read_estimator(self):
        pass

    def main(
        self,
        posterior: Posterior,
        model_name: str,
        scaler: Scaler,
        list_names: List[str],
        list_idx_sampling: List[int],
        list_fixed_values: List[float],
        #
        plot_1D_chains: bool,
        plot_2D_chains: bool,
        plot_ESS: bool,
        plot_comparisons_yspace: bool,
        #
        estimator_plot: Optional[PlotsEstimator] = None,
        analyze_regularization_weight: bool = False,
        Theta_true_scaled: Optional[np.ndarray] = None,
        list_lines_fit: Optional[List[str]] = None,
        #
        list_lines_valid: List[str] = [],
        y_valid: Optional[np.ndarray] = None,
        sigma_a_valid: Optional[np.ndarray] = None,
        omega_valid: Optional[np.ndarray] = None,
        sigma_m_valid: Optional[np.ndarray] = None,
        point_challenger: Dict = {},
        list_CI: List[int] = [68, 90, 95, 99],
    ):
        list_mcmc_folders = [
            f"{x[0]}/mc_chains.hdf5"
            for x in os.walk(f"{self.path_raw}/{model_name}")
            if "mcmc_" in x[0]
        ]
        list_mcmc_folders.sort()

        chain_type = "mcmc"
        N = posterior.N * 1
        D_sampling = posterior.D * 1
        D = len(list_fixed_values)
        L = len(list_lines_fit)
        print(f"N = {N}, L (fit) = {L}, D_sampling = {D_sampling}, D = {D}")

        if estimator_plot is not None:
            map_shaper = copy.copy(estimator_plot.map_shaper)
        else:
            map_shaper = None

        list_fixed_values_scaled = list(
            posterior.likelihood.forward_map.arr_fixed_values
        )
        list_fixed_values_scaled = [v if v != 0 else None for v in list_fixed_values]

        # clppd
        ResultsCLPPD(
            model_name=model_name,
            chain_type=chain_type,
            path_img=self.path_img,
            path_data_csv_out=self.path_data_csv_out_mcmc,
            N_MCMC=self.N_MCMC,
            N=N,
            L=L,
        ).main(list_mcmc_folders, map_shaper)

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
            objective_true = np.nan

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
            self.path_data_csv_out_mcmc,
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

        # deal with whole MC for MMSE and histograms
        if estimator_plot is not None:
            lower_bounds_lin = estimator_plot.lower_bounds_lin * 1
            upper_bounds_lin = estimator_plot.upper_bounds_lin * 1
        else:
            print(list_fixed_values)
            print(posterior.prior_indicator.lower_bounds)
            lower_bounds = posterior.prior_indicator.lower_bounds_full.reshape(
                (1, D),
            )
            upper_bounds = posterior.prior_indicator.upper_bounds_full.reshape(
                (1, D),
            )

            lower_bounds_lin = scaler.from_scaled_to_lin(lower_bounds)
            lower_bounds_lin = lower_bounds_lin.flatten()
            upper_bounds_lin = scaler.from_scaled_to_lin(upper_bounds)
            upper_bounds_lin = upper_bounds_lin.flatten()

        ResultsMC(
            model_name,
            chain_type,
            self.path_img,
            self.path_data_csv_out_mcmc,
            self.max_workers,
            self.N_MCMC,
            self.T_MC,
            self.T_BI,
            self.freq_save,
            N,
            list_idx_sampling,  # (D_sampling,)
            list_fixed_values,  # (D,)
            lower_bounds_lin,
            upper_bounds_lin,
            list_names,
        ).main(
            scaler,
            Theta_true_scaled_full,
            list_mcmc_folders,
            plot_ESS,
            plot_1D_chains,
            plot_2D_chains,
            plot_comparisons_yspace,
            point_challenger=point_challenger,
            list_CI=list_CI,
        )

        # save global MMSE performance
        # (to do now, once the MMSE if computed for all pixels)
        _ = ResultsMMSEandCI(
            model_name,
            self.path_img,
            self.path_data_csv_out_mcmc,
            N,
            D,
        ).main(
            posterior,
            scaler,
            estimator_plot,
            Theta_true_scaled_full,
            list_idx_sampling,
            list_fixed_values,
            list_CI,
        )

        # plot maps of ESS
        if plot_ESS and estimator_plot is not None:
            ResultsESS(
                model_name,
                self.path_img,
                self.path_data_csv_out_mcmc,
                N,
                D_sampling,
            ).main(
                map_shaper,
                list_names,
                list_idx_sampling,
            )

        ResultsBayesPvalues(
            model_name,
            chain_type,
            self.path_img,
            self.path_data_csv_out_mcmc,
            self.N_MCMC,
            N,
            D_sampling,
            plot_ESS,
        ).main(
            list_idx_sampling=list_idx_sampling,
            map_shaper=map_shaper if N > 1 else None,
        )

        # plot how many components have their true value
        # between min and max of MC
        if Theta_true_scaled is not None:
            ResultsValidMC(
                model_name=model_name,
                path_img=self.path_img,
                path_data_csv_out_mcmc=self.path_data_csv_out_mcmc,
                N_MCMC=self.N_MCMC,
                T_MC=self.T_MC,
                T_BI=self.T_BI,
                freq_save=self.freq_save,
                N=N,
                D_sampling=D_sampling,
            ).main(list_names, list_idx_sampling)

        # plot comparison of distributions of y and f(x)
        if plot_comparisons_yspace and list_lines_fit is not None:
            comparison_y_f_Theta = ResultsDistributionComparisonYandFTheta(
                model_name,
                self.path_img,
                self.path_data_csv_out_mcmc,
                self.N_MCMC,
                self.T_MC,
                self.T_BI,
                self.freq_save,
                N,
                D,
                list_idx_sampling,
                self.max_workers,
            )

            if hasattr(posterior.likelihood, "sigma_a"):
                sigma_a = posterior.likelihood.sigma_a
            elif hasattr(posterior.likelihood, "sigma"):
                sigma_a = posterior.likelihood.sigma
            else:
                sigma_a = np.zeros_like(posterior.likelihood.y)

            if hasattr(posterior.likelihood, "sigma_m"):
                sigma_m = posterior.likelihood.sigma_m
            else:
                sigma_m = np.zeros_like(posterior.likelihood.y)

            comparison_y_f_Theta.main(
                list_mcmc_folders,
                scaler,
                posterior.likelihood.forward_map,
                posterior.likelihood.y,
                posterior.likelihood.omega,
                sigma_a,
                sigma_m,
                list_lines_fit,
                "fit",
                point_challenger=point_challenger,
            )

            if len(list_lines_valid) > 0:
                comparison_y_f_Theta.main(
                    list_mcmc_folders,
                    scaler,
                    posterior.likelihood.forward_map,
                    y_valid,
                    omega_valid,
                    sigma_a_valid,
                    sigma_m_valid,
                    list_lines_valid,
                    "valid",
                    point_challenger=point_challenger,
                )
                posterior.likelihood.forward_map.restrict_to_output_subset(
                    list_lines_fit,
                )

        if analyze_regularization_weight:
            ResultsRegularizationWeights(
                model_name,
                self.path_img,
                self.path_data_csv_out_mcmc,
                self.N_MCMC,
                self.T_MC,
                self.T_BI,
                self.freq_save,
                D,
                list_names,
            ).main(list_mcmc_folders)

        print()
        return
