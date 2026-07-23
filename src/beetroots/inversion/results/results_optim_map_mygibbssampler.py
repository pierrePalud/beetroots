import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.inversion.results.abstract_results import ResultsExtractor
from beetroots.inversion.results.utils.bayes_pval_plots import ResultsBayesPvalues
from beetroots.inversion.results.utils.clppd import ResultsCLPPD
from beetroots.inversion.results.utils.kernel import ResultsKernels
from beetroots.inversion.results.utils.lowest_obj_estimator import (  # ResultsLowestObjective,
    ResultsLowestObjectiveGlobal,
)
from beetroots.inversion.results.utils.objective import (
    ResultsObjective,
    ResultsObjectiveGlobal,
)
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)

# from beetroots.modelling.target_distribution.posterior import Posterior
from beetroots.simulations.astro.forward_map_setup.utils import _scale_dict_fixed_params
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
        N,
        D,
        D_sampling,
        L,
        path_data_csv_out_optim_map: str,
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
        raise NotImplementedError(
            "This class must be adapted to the latest hierarchical code strucure."
        )  # TODO: Implement the class with the hierarchical model code structure.
        self.N = N
        r"""int: number of components :math:`n` in the dataset"""
        self.D = D
        r"""int: number of physical parameters :math:`d` involved in the forward map"""
        self.D_sampling = D_sampling
        r"""int: number of physical parameters :math:`d` that were sampled"""
        self.L = L

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
        self.sample_variables = sample_variables
        r"""dict[str, bool]: dictionary of booleans indicating whether the variable is sampled or not. The keys are the names of the variables and the values are booleans indicating whether the variable is sampled or not."""

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
        target_distributions: dict[str, TargetDistribution],
        model_name: str,
        scalers: Union[Scaler, dict[str, Scaler]],
        param_interest_name: str,
        #
        list_vars_idx_sampling: List[int],
        list_vars_fixed_values: Union[
            List[float], np.ndarray, dict[str, np.ndarray]
        ],  # In linear scale
        list_vars_names: Union[List[str], List[str]],
        #
        estimator_plots: Optional[
            Union[PlotsEstimator, dict[str, PlotsEstimator]]
        ] = None,
        Vars_true_scaled: Optional[Dict[str, np.ndarray]] = None,
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
        chain_type = "optim_map"
        current_true = None  # current is used to compute the objective function for the true variables. If true variables are provided, it will be set to a dictionnary afterwards.

        # Fetch the map shaper object.
        for key, estimator_plot in estimator_plots.items():
            if estimator_plot is not None:
                map_shaper = copy.copy(estimator_plots[key].map_shaper)
                break
            else:
                map_shaper = None

        # clppd
        ResultsCLPPD(
            model_name=model_name,
            chain_type=chain_type,
            path_img=self.path_img,
            path_data_csv_out=self.path_data_csv_out_optim_map,
            N_MCMC=self.N_MCMC,
            N=self.N,
            L=self.L,
        ).main(list_mcmc_folders, map_shaper)

        # objective global evolution
        components = set(
            [
                component
                for target_distribution in target_distributions.values()
                for component in target_distribution.distribution_components.values()
            ]
        )
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

        # MAP estimator from samples
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
            self.path_data_csv_out_optim_map,
            self.N_MCMC,
            self.T_MC,
            self.freq_save,
            target_distributions.keys(),
        ).main(
            list_mcmc_folders,
            lowest_obj,
            idx_lowest_obj,
            scalers,
            Vars_true_scaled_full,  # (N, D_sampling)
            list_vars_idx_sampling,  # (D_sampling,)
            list_vars_fixed_values,  # (D,)
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

            idx_lowest_obj, lowest_obj = ResultsObjective(
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

        ResultsBayesPvalues(
            model_name=model_name,
            chain_type=chain_type,
            path_img=self.path_img,
            path_data_csv_out=self.path_data_csv_out_optim_map,
            N_MCMC=self.N_MCMC,
            N=self.N,
            D_sampling=len(list_vars_idx_sampling[param_interest_name]),
            plot_ESS=False,
            sub_var=param_interest_name,
        ).main(
            list_idx_sampling=list_vars_idx_sampling[param_interest_name],
            map_shaper=map_shaper if self.N > 1 else None,
        )
