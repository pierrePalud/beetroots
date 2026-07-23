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
    ResultsLowestObjectiveGlobal,
)
from beetroots.inversion.results.utils.mc import ResultsMC
from beetroots.inversion.results.utils.mmse_ci import ResultsMMSEandCIMyGibbsSampler
from beetroots.inversion.results.utils.objective import (
    ResultsObjective,
    ResultsObjectiveGlobal,
)
from beetroots.inversion.results.utils.valid_mc import ResultsValidMC
from beetroots.inversion.results.utils.y_f_Var import (
    ResultsDistributionComparisonYandFVar,
)
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.simulations.astro.forward_map_setup.utils import _scale_dict_fixed_params
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
        N,
        D,
        D_sampling,
        L,
        path_data_csv_out_mcmc: str,
        path_img: str,
        path_raw: str,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        freq_save: int,
        max_workers: int,
        sample_variables: dict[str, bool],
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
        self.N = N
        r"""int: number of components :math:`n` in the dataset"""
        self.D = D
        r"""int: number of physical parameters :math:`d` involved in the forward map"""
        self.D_sampling = D_sampling
        r"""int: number of physical parameters :math:`d` that were sampled"""
        self.L = L
        r"""int: number of observables per component :math:`n` used for inversion, e.g., per pixel"""
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

        self.sample_variables = sample_variables
        r"""dict[str, bool]: whether the variable has been sampled or not"""

    def main(
        self,
        target_distributions: dict[str, TargetDistribution],
        model_name: str,
        scalers: Union[Scaler, dict[str, Scaler]],
        list_vars_names: Union[List[str], List[str]],
        param_interest_name: str,
        list_vars_idx_sampling: List[int],
        list_vars_fixed_values: Union[
            List[float], np.ndarray, dict[str, np.ndarray]
        ],  # In linear scale
        #
        plot_2D_chains: dict[bool],
        plot_1D_chains: dict[bool],
        plot_ESS: bool,
        plot_comparisons_yspace: bool,
        #
        estimator_plots: Optional[
            Union[PlotsEstimator, dict[str, PlotsEstimator]]
        ] = None,
        analyze_regularization_weight: bool = False,
        Vars_true_scaled: Optional[np.ndarray] = None,
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
        var_name_interest : str
            name of the variable of interest (for instance, usually denoted as :math:`\Theta` in the literature)
        list_vars_idx_sampling : List[int]
            indices of the physical parameters that were sampled (the other ones were fixed)
        list_vars_fixed_values : List[float]
            list of used values for the parameters fixed during the sampling
        plot_1D_chains : bool
            wether to plot each of the :math:`N \times D` 1D chains and histograms for each physical parameter :math:`\theta_{nd}`
        plot_2D_chains : bool
            wether to plot each of the :math:`N \times D \times (D-1)` 2D chains and histograms for pairs of parameters :math:`(\theta_{n d_1}, \theta_{n d_2})` with :math:`1 \leq d_1 < d_2 \leq D`
        plot_ESS : dict
            wether to plot the Effective sample size maps for each variable (only used when :math:`N > 1`)
        plot_comparisons_yspace : bool
            whether to plot comparisons of the distribution on :math:`y_n` and :math:`\mathcal{A}(f(\theta_n))` (with :math:`\mathcal{A}` the noise model). Offers a visualization to understand the model checking based on the Bayesian p-value :cite:p:`paludProblemesInversesTest2023a`
        estimator_plots : Optional[PlotsEstimator], optional
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
        current_true = None  # current is used to compute the objective function for the true variables. If true variables are provided, it will be set to a dictionnary afterwards.

        # Verify assertions on the types of the inputs
        assert self.D >= self.D_sampling, (
            f"should have D={self.D} >= D_sampling={self.D_sampling}"
        )
        print(
            f"N = {self.N}, L (fit) = {self.L}, D_sampling = {self.D_sampling}, D = {self.D}"
        )

        assert isinstance(target_distributions, dict)
        for val in [list_vars_fixed_values, list_vars_idx_sampling, list_vars_names]:
            assert isinstance(val, dict), (
                "list_vars_fixed_values, list_vars_idx_sampling, list_vars_names, scalers should be dictionaries"
            )

        # Fetch the map shaper object.
        for key, estimator_plot in estimator_plots.items():
            if estimator_plot is not None:
                map_shaper = copy.copy(estimator_plots[key].map_shaper)
                break
            else:
                map_shaper = None

        list_vars_fixed_values_scaled = dict.fromkeys(target_distributions.keys())
        for key in target_distributions.keys():
            list_vars_fixed_values_scaled[key] = list(
                _scale_dict_fixed_params(
                    scaler=scalers[key],
                    dict_fixed_params=dict(
                        zip(list_vars_names[key], list_vars_fixed_values[key])
                    ),
                ).values()
            )

        # clppd
        ResultsCLPPD(
            model_name=model_name,
            chain_type=chain_type,
            path_img=self.path_img,
            path_data_csv_out=self.path_data_csv_out_mcmc,
            N_MCMC=self.N_MCMC,
            N=self.N,
            L=self.L,
        ).main(list_mcmc_folders, map_shaper)

        components = set(
            [
                component
                for target_distribution in target_distributions.values()
                for component in target_distribution.distribution_components.values()
            ]
        )
        # objective global evolution
        if all(
            [Vars_true_scaled[key] is not None for key in target_distributions.keys()]
        ):
            current_true = {
                key: {"var": Vars_true_scaled[key]}
                for key in target_distributions.keys()
            }
            objective_true = 0
            for comp in components:
                comp.evaluate_all_nlpdf_utils(
                    current=current_true,
                    idx_pix=None,
                    compute_derivatives=False,
                    compute_derivatives_2nd_order=False,
                )
                objective_true += comp.neglog_pdf(chromatic_gibbs=False)
        else:
            objective_true = np.nan

        idx_lowest_obj, lowest_obj = ResultsObjectiveGlobal(
            model_name,
            chain_type,
            self.path_img,
            self.N_MCMC,
            self.T_MC,
            self.T_BI,
            self.freq_save,
        ).main(list_mcmc_folders, objective_true)

        Vars_true_scaled_full = dict.fromkeys(target_distributions.keys())
        # MAP estimator from samples
        for key in target_distributions.keys():
            if Vars_true_scaled[key] is not None:
                Var_true_scaled_full = np.zeros(
                    (self.N, len(list_vars_names[key]))
                )  # len(list_vars_names[key]) gives the total number of components for this variable (even the one not sampled)
                Var_true_scaled_full[:, list_vars_idx_sampling[key]] += (
                    Vars_true_scaled[key]
                )

                for d in range(self.D):
                    if list_vars_fixed_values_scaled[key][d] is not None:
                        Var_true_scaled_full[:, d] += list_vars_fixed_values_scaled[
                            key
                        ][d]
            else:
                Var_true_scaled_full = None
            Vars_true_scaled_full[key] = Var_true_scaled_full

        ResultsLowestObjectiveGlobal(
            model_name,
            chain_type,
            self.path_img,
            self.path_data_csv_out_mcmc,
            self.N_MCMC,
            self.T_MC,
            self.freq_save,
            target_distributions.keys(),
        ).main(
            list_mcmc_folders,
            lowest_obj,
            idx_lowest_obj,
            scalers,
            Vars_true_scaled_full,  # (N, M)               M being the total number of components (sampled or not) M is in general L or D
            list_vars_idx_sampling,  # (M_sampling,)   M_sampling is the number of sampled component. It is in general L or D_sampling
            list_vars_fixed_values,  # (M,)
            estimator_plots,
        )

        for key in target_distributions.keys():
            if self.sample_variables[
                key
            ]:  # We only plot the kernel analysis for the sampled variables, it makes no sense for other variables.
                # kernel analysis
                ResultsKernels(
                    model_name,
                    chain_type,
                    self.path_img * 1,
                    self.N_MCMC * 1,
                    self.T_MC * 1,
                    self.T_BI * 1,
                    self.freq_save * 1,
                    sub_var=key,
                ).main(list_mcmc_folders, map_shaper)

            if current_true is not None:
                # objective evolution
                objective_true = target_distributions[key].neglog_pdf(
                    current=current_true, update_nlpdf_utils=True, chromatic_gibbs=False
                )
            else:
                objective_true = np.nan

            ResultsObjective(
                model_name,
                chain_type,
                self.path_img * 1,
                self.N_MCMC * 1,
                self.T_MC * 1,
                self.T_BI * 1,
                self.freq_save * 1,
                N=self.N,
                D=self.D,
                L=self.L,
                sub_var=key,
            ).main(list_mcmc_folders, objective_true)

            # deal with whole MC for MMSE and histograms
            if estimator_plots[key] is not None:
                lower_bounds_lin = estimator_plots[key].lower_bounds_lin * 1
                upper_bounds_lin = estimator_plots[key].upper_bounds_lin * 1
            elif any(
                [
                    isinstance(component, SmoothIndicatorPrior)
                    for component in target_distributions[key].components
                ]
            ):
                # Fetch the indicator prior in the target distribution
                indicator_prior = [
                    component
                    for component in target_distributions[key].components
                    if isinstance(component, SmoothIndicatorPrior)
                ][0]

                lower_bounds = indicator_prior.lower_bounds_full.reshape(
                    (1, self.D),
                )
                upper_bounds = indicator_prior.upper_bounds_full.reshape(
                    (1, self.D),
                )

                lower_bounds_lin = scalers[key].from_scaled_to_lin(lower_bounds)
                lower_bounds_lin = lower_bounds_lin.flatten()
                upper_bounds_lin = scalers[key].from_scaled_to_lin(upper_bounds)
                upper_bounds_lin = upper_bounds_lin.flatten()
            else:
                raise ValueError(
                    "No prior found in the target distribution. Please provide a 'PlotEstimator' object or a 'SmoothIndicatorPrior' in the target distribution."
                )

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
                self.N,
                list_vars_idx_sampling[key],  # (D_sampling,)
                list_vars_fixed_values[key],  # (D,) Values in linear scale
                lower_bounds_lin,
                upper_bounds_lin,
                list_vars_names[key],
                sub_var=key,
            ).main(
                scalers[key],
                Vars_true_scaled_full[key],
                list_mcmc_folders,
                plot_ESS[key],
                plot_1D_chains[key],
                plot_2D_chains[key],
                plot_comparisons_yspace,
                point_challenger=point_challenger,
                list_CI=list_CI,
            )

            # plot how many components have their true value
            # between min and max of MC
            if Vars_true_scaled[key] is not None:
                ResultsValidMC(
                    model_name=model_name,
                    path_img=self.path_img,
                    path_data_csv_out_mcmc=self.path_data_csv_out_mcmc,
                    N_MCMC=self.N_MCMC,
                    T_MC=self.T_MC,
                    T_BI=self.T_BI,
                    freq_save=self.freq_save,
                    N=self.N,
                    D_sampling=len(list_vars_idx_sampling[key]),
                    sub_var=key,
                ).main(list_vars_names[key], list_vars_idx_sampling[key])

            # plot maps of ESS
            if plot_ESS[key] and estimator_plots[key] is not None:
                ResultsESS(
                    model_name,
                    self.path_img,
                    self.path_data_csv_out_mcmc,
                    self.N,
                    D_sampling=len(list_vars_idx_sampling[key]),
                    sub_var=key,
                ).main(
                    map_shaper,
                    list_vars_names[key],
                    list_vars_idx_sampling[key],
                )

        ResultsBayesPvalues(
            model_name,
            chain_type,
            self.path_img,
            self.path_data_csv_out_mcmc,
            self.N_MCMC,
            N=self.N,
            D_sampling=len(list_vars_idx_sampling[param_interest_name]),
            plot_ESS=plot_ESS[param_interest_name],
            sub_var=param_interest_name,
        ).main(
            list_idx_sampling=list_vars_idx_sampling[param_interest_name],
            map_shaper=map_shaper if self.N > 1 else None,
        )

        # save global MMSE performance
        # (to do now, once the MMSE is computed for all pixels)
        _ = ResultsMMSEandCIMyGibbsSampler(
            model_name,
            self.path_img,
            self.path_data_csv_out_mcmc,
            self.N,
            self.D,
        ).main(
            components,
            scalers,
            estimator_plots,
            Vars_true_scaled_full,
            list_vars_idx_sampling,
            list_vars_fixed_values,
            list_CI,
        )
        # plot comparison of distributions of y and f(\theta)
        if plot_comparisons_yspace and list_lines_fit is not None:
            comparison_y_f_Var = ResultsDistributionComparisonYandFVar(
                model_name,
                self.path_img,
                self.path_data_csv_out_mcmc,
                self.N_MCMC,
                self.T_MC,
                self.T_BI,
                self.freq_save,
                self.N,
                self.D,
                list_vars_idx_sampling[param_interest_name],
                self.max_workers,
                param_interest_name=param_interest_name,
            )

            # Retrieve the required elements.
            found_sigma_a, found_sigma_m, found_omega, found_y, found_forward_map = (
                False,
                False,
                False,
                False,
                False,
            )
            for comp in components:
                if hasattr(comp, "sigma_a"):
                    sigma_a = comp.sigma_a
                    found_sigma_a = True
                elif not found_sigma_a and hasattr(comp, "sigma"):
                    sigma_a = comp.sigma
                    found_sigma_a = True
                elif not found_sigma_a:
                    sigma_a = np.zeros((self.N, self.L))

                if hasattr(comp, "sigma_m"):
                    sigma_m = comp.sigma_m
                    found_sigma_m = True
                elif not found_sigma_m:
                    sigma_m = np.zeros((self.N, self.L))

                if hasattr(comp, "omega"):
                    omega = comp.omega
                    found_omega = True
                elif not found_omega:
                    omega = np.zeros((self.N, self.L))

                if hasattr(comp, "y"):
                    y = comp.y
                    found_y = True
                elif not found_y:
                    y = np.zeros((self.N, self.L))

                if hasattr(comp, "forward_map"):
                    forward_map = comp.forward_map
                    found_forward_map = True
                elif not found_forward_map:
                    forward_map = None

            if not found_y or not found_forward_map:
                raise ValueError(
                    "One of the target distributions must have a 'y' and a 'forward_map' attribute in one of its component distributions."
                )

            comparison_y_f_Var.main(
                list_mcmc_folders,
                scalers[param_interest_name],
                forward_map,
                y,
                omega,
                sigma_a,
                sigma_m,
                list_lines_fit,
                "fit",
                point_challenger=point_challenger,
            )

            if len(list_lines_valid) > 0:
                comparison_y_f_Var.main(
                    list_mcmc_folders,
                    scalers[param_interest_name],
                    forward_map,
                    y_valid,
                    omega_valid,
                    sigma_a_valid,
                    sigma_m_valid,
                    list_lines_valid,
                    "valid",
                    point_challenger=point_challenger,
                )
                forward_map.restrict_to_output_subset(
                    list_lines_fit,
                )
        return

    """


        if analyze_regularization_weight:
            ResultsRegularizationWeights(
                model_name,
                self.path_img,
                self.path_data_csv_out_mcmc,
                self.N_MCMC,
                self.T_MC,
                self.T_BI,
                self.freq_save,
                self.D,
                list_names,
            ).main(list_mcmc_folders)

        return
    """
