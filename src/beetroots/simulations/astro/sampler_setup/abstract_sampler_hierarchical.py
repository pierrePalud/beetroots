import abc
import time
from typing import Dict, List, Optional, Union

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from beetroots.inversion.results.results_mcmc_mygibbssampler import ResultsExtractorMCMC
from beetroots.inversion.results.results_optim_map_mygibbssampler import (
    ResultsExtractorOptimMAP,
)
from beetroots.inversion.run.run_mcmc import RunMCMC
from beetroots.inversion.run.run_optim_map import RunOptimMAP
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.my_gibbs_sampler import MyGibbsSampler
from beetroots.sampler.saver.my_gibbs_saver import MyGibbsSaver
from beetroots.sampler.utils.sampler_params import MyGibbsSamplerParams
from beetroots.simulations.astro.sampler_setup.abstract_sampler_approach import (
    SimulationTargetDistributionType,
)
from beetroots.space_transform.transform import MyScaler


class SimulationMyGibbsSampler(SimulationTargetDistributionType):
    @abc.abstractmethod
    def setup_full_conditionals(self, **kwargs) -> dict:
        pass

    def setup_target_distribution(self, **kwargs) -> tuple:
        outputs = self.setup_full_conditionals(**kwargs)

        for k in ["dict_models", "scalers"]:
            if k not in outputs.keys():
                raise TypeError(
                    f"output of 'setup_full_conditionals' method should be a dict comprising a key '{k}' with a dictionary as value."
                )
            elif not isinstance(outputs[k], dict):
                raise TypeError(
                    f"the value of key '{k}' should be a dictionary with one key/value pair for each target distribution."
                )
        return outputs

    def restore_kernel_states(self, kernels_list, checkpoint_kernels: Dict) -> None:
        """Restore kernel hyperparameter states from checkpoint for a list of kernels.

        Parameters
        ----------
        kernels_list : list
            List of kernel objects to restore
        checkpoint_kernels : dict
            Checkpoint dictionary containing kernel states with structure:
            {kernel_name: {param_name: value, ...}, ...}
        """
        for kernel in kernels_list:
            kernel_name = kernel.__class__.__name__
            if kernel_name in checkpoint_kernels:
                kernel_params = checkpoint_kernels[kernel_name]
                if hasattr(kernel, "set_parameters_from_dict"):
                    try:
                        kernel.set_parameters_from_dict(kernel_params)
                    except Exception:
                        # best-effort fallback to attribute assignment
                        for param_name, param_value in kernel_params.items():
                            if hasattr(kernel, param_name):
                                setattr(kernel, param_name, param_value)
                else:
                    for param_name, param_value in kernel_params.items():
                        if hasattr(kernel, param_name):
                            setattr(kernel, param_name, param_value)

    def restore_model_check_states(self, dict_model_check: Dict, checkpoint_model_check: Dict) -> None:
        """Restore model checking statistics from checkpoint.

        Parameters
        ----------
        dict_model_check : dict
            Model check dictionary to update with checkpoint values
        checkpoint_model_check : dict
            Checkpoint dictionary containing model check states
        """
        for key, value in checkpoint_model_check.items():
            dict_model_check[key] = value

    def inversion_optim(
        self,
        dict_models: Dict[str, Dict[str, TargetDistribution]],
        scalers: dict[str, MyScaler],
        my_sampler_params: MyGibbsSamplerParams,
        param_interest_name: str,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        rng_list=List[xp.random.Generator],
        batch_size: int = 10,
        freq_save: int = 1,
        start_from: Optional[str] = None,
        can_run_in_parallel: bool = True,
        save_forward_map_evals: bool = False,
        debugging: dict = {},
        **kwargs,
    ) -> None:
        tps_init = time.time()

        samplers_ = [
            MyGibbsSampler(my_sampler_params, self.D_sampling, self.L, self.N, rng_)
            for rng_ in rng_list
        ]

        saver_ = MyGibbsSaver(
            N=self.N,
            D=self.D,
            D_sampling=self.D_sampling,
            L=self.L,
            scaler=scalers,
            batch_size=batch_size,
            freq_save=freq_save,
            list_idx_sampling=self.list_idx_sampling_param_interest,
            save_forward_map_evals=save_forward_map_evals,
        )

        if debugging.get("run_sampling", True):
            run_optim_map = RunOptimMAP(self.path_data_csv_out, self.max_workers)
            run_optim_map.main(
                dict_models,
                samplers_,
                saver_,
                scalers,
                N_MCMC,
                T_MC,
                path_raw=self.path_raw,
                Vars_0=start_from,
                freq_save=freq_save,
                can_run_in_parallel=can_run_in_parallel,
            )

        if debugging.get("run_results_extraction", True):
            results_optim_map = ResultsExtractorOptimMAP(
                self.N,
                self.D,
                self.D_sampling,
                self.L,
                self.path_data_csv_out_optim_map,
                self.path_img,
                self.path_raw,
                N_MCMC,
                T_MC,
                T_BI,
                freq_save,
                self.max_workers,
                my_sampler_params.sample_variables,
            )
            for model_name, target_distributions in dict_models.items():
                results_optim_map.main(
                    target_distributions=target_distributions,
                    model_name=model_name,
                    scalers=scalers,
                    #
                    param_interest_name=param_interest_name,
                    #
                    list_vars_idx_sampling=self.list_vars_idx_sampling,
                    list_vars_fixed_values=self.list_vars_fixed_values,  # Values in linear scale
                    list_vars_names=self.list_vars_names_plots,
                    #
                    estimator_plots=self.plots_estimator,
                    Vars_true_scaled=self.Vars_true_scaled,
                )

        duration = time.time() - tps_init  # is seconds
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        msg = "Simulation and analysis finished. Total duration : "
        msg += f"{duration_str} s\n"
        print(msg)

        list_model_names = list(dict_models.keys())
        return list_model_names

    def inversion_mcmc(
        self,
        dict_models: Dict[str, Dict[str, TargetDistribution]],
        scalers: dict[str, MyScaler],
        my_sampler_params: MyGibbsSamplerParams,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        plot_1D_chains: dict[bool],
        plot_2D_chains: dict[bool],
        plot_ESS: dict[str, bool],
        plot_comparisons_yspace: bool,
        rng_list: List[xp.random.Generator],
        #
        param_interest_name: str,
        #
        debugging: dict = {},
        #
        batch_size: int = 10,
        freq_save: int = 1,
        start_from: Optional[Union[str, dict]] = None,
        #
        regu_spatial_N0: Union[int, float] = xp.inf,
        regu_spatial_scale: Optional[float] = 1.0,
        regu_spatial_vmin: Optional[float] = 1e-8,
        regu_spatial_vmax: Optional[float] = 1e8,
        #
        y_valid: Optional[xp.ndarray] = None,
        sigma_a_valid: Optional[xp.ndarray] = None,
        omega_valid: Optional[xp.ndarray] = None,
        sigma_m_valid: Optional[xp.ndarray] = None,
        #
        can_run_in_parallel: bool = True,
        point_challenger: Dict = {},
        list_CI: List[int] = [68, 90, 95, 99],
        #
        save_forward_map_evals: bool = False,
        **kwargs,
    ) -> None:
        tps_init = time.time()

        samplers_ = [
            MyGibbsSampler(my_sampler_params, self.D_sampling, self.L, self.N, rng_)
            for rng_ in rng_list
        ]

        saver_ = MyGibbsSaver(
            self.N,
            self.D,
            self.D_sampling,
            self.L,
            scalers,
            batch_size=batch_size,
            freq_save=freq_save,
            list_idx_sampling=self.list_idx_sampling_param_interest,
            save_forward_map_evals=save_forward_map_evals,
        )

        if debugging.get("run_sampling", True):
            run_mcmc = RunMCMC(self.path_data_csv_out, self.max_workers)
            run_mcmc.main(
                dict_models,
                samplers_,
                saver_,
                scalers,
                N_MCMC,
                T_MC,
                T_BI=T_BI,
                path_raw=self.path_raw,
                Vars_0=start_from,
                freq_save=freq_save,
                #
                regu_spatial_N0=regu_spatial_N0,
                regu_spatial_scale=regu_spatial_scale,
                regu_spatial_vmin=regu_spatial_vmin,
                regu_spatial_vmax=regu_spatial_vmax,
                #
                can_run_in_parallel=can_run_in_parallel,
                #
            )

        if debugging.get("run_results_extraction", True):
            print("Start results extraction...")
            results_mcmc = ResultsExtractorMCMC(
                self.N,
                self.D,
                self.D_sampling,
                self.L,
                self.path_data_csv_out_mcmc,
                self.path_img,
                self.path_raw,
                N_MCMC,
                T_MC,
                T_BI,
                freq_save,
                self.max_workers,
                my_sampler_params.sample_variables,
            )
            for model_name, target_distributions in dict_models.items():
                results_mcmc.main(
                    target_distributions=target_distributions,
                    model_name=model_name,
                    scalers=scalers,
                    list_vars_names=self.list_vars_names_plots,
                    param_interest_name=param_interest_name,
                    list_vars_idx_sampling=self.list_vars_idx_sampling,
                    list_vars_fixed_values=self.list_vars_fixed_values,  # Values in linear scale
                    #
                    plot_1D_chains=plot_1D_chains,
                    plot_2D_chains=plot_2D_chains,
                    plot_ESS=plot_ESS,
                    plot_comparisons_yspace=plot_comparisons_yspace,
                    #
                    estimator_plots=self.plots_estimator,
                    analyze_regularization_weight=xp.isfinite(regu_spatial_N0),
                    list_lines_fit=self.list_lines_fit,
                    Vars_true_scaled=self.Vars_true_scaled,
                    list_lines_valid=self.list_lines_valid,
                    y_valid=y_valid,
                    sigma_a_valid=sigma_a_valid,
                    omega_valid=omega_valid,
                    sigma_m_valid=sigma_m_valid,
                    point_challenger=point_challenger,
                    list_CI=list_CI,
                )

        duration = time.time() - tps_init  # is seconds
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        msg = "Simulation and analysis finished. Total duration : "
        msg += f"{duration_str} s\n"
        print(msg)

        list_model_names = list(dict_models.keys())
        return list_model_names
