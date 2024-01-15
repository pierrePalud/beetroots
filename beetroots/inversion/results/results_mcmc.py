import copy
import os
from typing import Dict, List, Optional, Union

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
    r"""extractor of inference results for the Markov chain data that was saved."""

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
        r"""

        Parameters
        ----------
        path_data_csv_out_mcmc : str
            path to the csv file in which the performance of estimators is to be saved
        path_img : str
            path to the folder in which images are to be saved
        path_raw : str
            path to the raw ``.hdf5`` files
        N_MCMC : int
            number of Markov chains to run per posterior distribution
        T_MC : int
            total size of each Markov chain
        T_BI : int
            duration of the `Burn-in` phase
        freq_save : int
            frequency of saved iterates, 1 means that all iterates were saved (used to show correct Markov chain sizes in chain plots)
        max_workers : int
            maximum number of workers that can be used for results extraction
        """
        self.path_data_csv_out_mcmc = path_data_csv_out_mcmc
        r"""str: path to the csv file in which the performance of estimators is to be saved"""
        self.path_img = path_img
        r"""str: path to the folder in which images are to be saved"""
        self.path_raw = path_raw
        r"""str: path to the raw ``.hdf5`` files"""

        self.N_MCMC = N_MCMC
        r"""int: number of Markov chains to run per posterior distribution"""
        self.T_MC = T_MC
        r"""int: total size of each Markov chain"""
        self.T_BI = T_BI
        r"""int: duration of the `Burn-in` phase"""
        self.freq_save = freq_save
        r"""int: frequency of saved iterates, 1 means that all iterates were saved (used to show correct Markov chain sizes in chain plots)"""

        self.max_workers = max_workers
        r"""int: maximum number of workers that can be used for results extraction"""

    def main(
        self,
        posterior: Posterior,
        model_name: str,
        scaler: Scaler,
        list_names: List[str],
        list_idx_sampling: List[int],
        list_fixed_values: Union[List[float], np.ndarray],
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
        r"""performs the data extraction, in this order:

        * step 1 : clppd (see ``beetroots.inversion.results.utils.clppd``)
        * step 2 : kernel analysis (see ``beetroots.inversion.results.utils.kernel``)
        * step 3 : objective evolution (see ``beetroots.inversion.results.utils.objective``) (the objective is the negative log posterior pdf)
        * step 4 : MAP estimator from samples (see ``beetroots.inversion.results.utils.lowest_obj_estimator``)
        * step 5 : deal with whole Markov chain for MMSE and histograms, in a pixel-wise approach to avoid overloading the memory (see ``beetroots.inversion.results.utils.mc``)
        * step 6 : save global MMSE performance (see ``beetroots.inversion.results.utils.mmse_ci``)
        * step 7 (if map) : plot maps of ESS (see ``beetroots.inversion.results.utils.ess_plots``)
        * step 8 : model checking with Bayesian p-value computation (see ``beetroots.inversion.results.utils.bayes_pval_plots``)
        * step 9 (if the true value is known): plot how many components have their true value between min and max of Markov chain (see ``beetroots.inversion.results.utils.valid_mc``)
        * step 10 : plot comparison of distributions of :math:`y_n` and :math:`f(\theta_n)` for all :math:`n \in [\![1, N]\!]` (see ``beetroots.inversion.results.utils.y_f_Theta``)
        * step 11 (if ``analyze_regularization_weight``) : analysis of the regularization weight :math:`\tau` automatic tuning (see ``beetroots.inversion.results.utils.regularization_weights``)

        Parameters
        ----------
        posterior : Posterior
            probability distribution used to generate the Markov chain(s)
        model_name : str
            name of the model, used to identify the posterior distribution
        scaler : Scaler
            contains the transformation of the Theta values from their natural space to their scaled space (in which the sampling happens) and its inverse
        list_names : List[str]
            names of the D physical parameters to appear in plots (for instance, $P_{th}$ for thermal pressure)
        list_idx_sampling : List[int]
            indices of the physical parameters that were sampled (the other ones were fixed)
        list_fixed_values : List[float]
            list of used values for the parameters fixed during the sampling
        plot_1D_chains : bool
            wether to plot each of the :math:`N \times D` 1D chains and histograms for each physical parameter :math:`\theta_{nd}`
        plot_2D_chains : bool
            wether to plot each of the :math:`N \times D \times (D-1)` 2D chains and histograms for pairs of parameters :math:`(\theta_{n d_1}, \theta_{n d_2})` with :math:`1 \leq d_1 < d_2 \leq D`
        plot_ESS : bool
            wether to plot the Effective sample size maps (only used when :math:`N > 1`)
        plot_comparisons_yspace : bool
            whether to plot comparisons of the distribution on :math:`y_n` and :math:`\mathcal{A}(f(\theta_n))` (with :math:`\mathcal{A}` the noise model). Offers a visualization to understand the model checking based on the Bayesian p-value :cite:p:`paludProblemesInversesTest2023a`
        estimator_plot : Optional[PlotsEstimator], optional
            object used to plot the estimator figures, by default None
        analyze_regularization_weight : bool, optional
            wether to analyze the evaluation of the regularization weight :math:`\tau`, by default False
        Theta_true_scaled : Optional[np.ndarray], optional
            true value for the inferred physical parameter :math:`\Theta` (only possible for toy cases), by default None
        list_lines_fit : Optional[List[str]], optional
            names of the observables used for the inversion, by default None
        list_lines_valid : List[str], optional
            names of the available observables not used for the inversion (can be used for model checking), by default []
        y_valid : Optional[np.ndarray], optional
            observation values for the observables not used for inversion. If provided, must have shape (N, L_valid). by default None
        sigma_a_valid : Optional[np.ndarray], optional
            additive noise standard deviation values for the observables not used for inversion. If provided, must have shape (N, L_valid). by default None
        omega_valid : Optional[np.ndarray], optional
            censor threshold values for the observables not used for inversion. If provided, must have shape (N, L_valid)., by default None
        sigma_m_valid : Optional[np.ndarray], optional
            multiplicative noise standard deviation values for the observables not used for inversion. If provided, must have shape (N, L_valid)., by default None
        point_challenger : Dict, optional
            other estimator that can come from the literature, provided to be compared with the inference results, by default {}
        list_CI : List[int], optional
            list of credibility intervals to evaluate (in percent), by default [68, 90, 95, 99]
        """
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

        assert D >= D_sampling, f"should have D={D} >= D_sampling={D_sampling}"
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

        # plot comparison of distributions of y and f(\theta)
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
