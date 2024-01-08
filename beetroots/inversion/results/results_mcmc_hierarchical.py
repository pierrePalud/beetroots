import os
from typing import List, Optional

import numpy as np

from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.inversion.results.abstract_results import ResultsExtractor
from beetroots.inversion.results.utils.ess_plots import ResultsESS
from beetroots.inversion.results.utils.kernel_hierarchical import (
    ResultsKernelsHierarchical,
)
from beetroots.inversion.results.utils.lowest_obj_estimator_hierarchical import (
    ResultsLowestObjectiveHierarchical,
)
from beetroots.inversion.results.utils.mc_hierarchical import ResultsMCHierarchical
from beetroots.inversion.results.utils.mmse_ci_hierarchical import (
    ResultsMMSEandCIHierarchical,
)
from beetroots.inversion.results.utils.objective_hierarchical import (
    ResultsObjectiveHierarchical,
)
from beetroots.inversion.results.utils.regularization_weights import (
    ResultsRegularizationWeights,
)
from beetroots.inversion.results.utils.valid_mc import ResultsValidMC
from beetroots.inversion.results.utils.y_f_Theta import (
    ResultsDistributionComparisonYandFTheta,
)
from beetroots.modelling.posterior import Posterior
from beetroots.space_transform.abstract_transform import Scaler


class ResultsExtractorHierarchicalMCMC(ResultsExtractor):

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
    ):
        list_mcmc_folders = [
            f"{x[0]}/mc_chains.hdf5"
            for x in os.walk(f"{self.path_raw}/{model_name}")
            if "mcmc_" in x[0]
        ]
        list_mcmc_folders.sort()

        conditional_theta = posterior[1]
        conditional_u = posterior[0]

        chain_type = "mcmc"
        N = posterior[1].N * 1
        D = len(list_names)
        L = len(list_lines_fit)

        if Theta_true_scaled is not None:
            u_true = conditional_theta.likelihood.forward_map.evaluate(
                Theta_true_scaled,
            )
        else:
            u_true = None

        # kernel analysis
        ResultsKernelsHierarchical(
            model_name,
            chain_type,
            self.path_img * 1,
            self.N_MCMC * 1,
            self.T_MC * 1,
            self.freq_save * 1,
        ).main(list_mcmc_folders)

        # objective evolution
        # if Theta_true_scaled is not None:
        #     forward_map_evals = conditional_theta.likelihood.evaluate_all_forward_map(
        #         Theta_true_scaled,
        #         True,
        #     )
        #     nll_utils = conditional_theta.likelihood.evaluate_all_nll_utils(
        #         forward_map_evals,
        #     )
        #     objective_true = conditional_theta.neglog_pdf(
        #         Theta_true_scaled, forward_map_evals, nll_utils
        #     )
        # else:
        objective_true = np.nan

        idx_lowest_obj, lowest_obj = ResultsObjectiveHierarchical(
            model_name,
            chain_type,
            self.path_img * 1,
            self.N_MCMC * 1,
            self.T_MC * 1,
            self.T_BI * 1,
            self.freq_save * 1,
        ).main(list_mcmc_folders, objective_true)

        # MAP estimator from samples
        ResultsLowestObjectiveHierarchical(
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
            Theta_true_scaled,
            estimator_plot,
            list_lines_fit,
        )

        # deal with whole MC for MMSE and histograms
        if estimator_plot is not None:
            lower_bounds_lin = estimator_plot.lower_bounds_lin * 1
            upper_bounds_lin = estimator_plot.upper_bounds_lin * 1
        else:
            lower_bounds = conditional_theta.prior_indicator.lower_bounds.reshape(
                (1, D),
            )
            upper_bounds = conditional_theta.prior_indicator.upper_bounds.reshape(
                (1, D),
            )

            lower_bounds_lin = scaler.from_scaled_to_lin(lower_bounds).flatten()
            upper_bounds_lin = scaler.from_scaled_to_lin(upper_bounds).flatten()

        ResultsMCHierarchical(
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
            lower_bounds_lin,
            upper_bounds_lin,
            list_names=list_names,
            list_lines=list_lines_fit,
        ).main(
            scaler,
            Theta_true_scaled,
            u_true,
            list_mcmc_folders,
            plot_ESS,
            plot_1D_chains,
            plot_2D_chains,
            plot_comparisons_yspace,
        )

        # save global MMSE performance
        # (to do now, once the MMSE if computed for all pixels)
        df_mmse_theta, _ = ResultsMMSEandCIHierarchical(
            model_name,
            self.path_img,
            self.path_data_csv_out_mcmc,
            N,
            D,
            L,
        ).main(
            conditional_theta,
            scaler,
            estimator_plot,
            Theta_true_scaled,
            list_lines_fit,
        )

        # plot maps of ESS
        if plot_ESS and estimator_plot is not None:
            ResultsESS(
                model_name,
                self.path_img,
                self.path_data_csv_out_mcmc,
                N,
                D,
            ).main(estimator_plot.map_shaper, list_names)

        # plot how many components have their true value
        # between min and max of MC
        if Theta_true_scaled is not None:
            ResultsValidMC(
                model_name,
                self.path_img,
                self.path_data_csv_out_mcmc,
                self.N_MCMC,
                self.T_MC,
                self.T_BI,
                self.freq_save,
                N,
                D,
            ).main(list_names)

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
                self.max_workers,
            )
            comparison_y_f_Theta.main(
                df_mmse_theta,
                list_mcmc_folders,
                scaler,
                posterior.likelihood.forward_map,
                posterior.likelihood.y,
                posterior.likelihood.omega,
                posterior.likelihood.sigma_a,
                posterior.likelihood.sigma_m,
                list_lines_fit,
                "fit",
            )

            if len(list_lines_valid) > 0:
                comparison_y_f_Theta.main(
                    df_mmse_theta,
                    list_mcmc_folders,
                    scaler,
                    posterior.likelihood.forward_map,
                    y_valid,
                    omega_valid,
                    sigma_a_valid,
                    sigma_m_valid,
                    list_lines_valid,
                    "valid",
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
            ).main()

        print()
        return
