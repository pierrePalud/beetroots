from typing import Dict, List, Optional, Union

import numpy as np

from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.sampler.utils.my_sampler_params import MySamplerParams
from beetroots.simulations.abstract_simulation import Simulation
from beetroots.simulations.astro import data_validation
from beetroots.simulations.astro.forward_map.abstract_nn import SimulationNN
from beetroots.simulations.astro.observation.abstract_real_data import (
    SimulationRealData,
)
from beetroots.simulations.astro.posterior_type.abstract_direct import (
    SimulationMySampler,
)


class SimulationRealDataNN(SimulationNN, SimulationRealData, SimulationMySampler):
    __slots__ = (
        "path_output_sim",
        "path_img",
        "path_raw",
        "path_data_csv",
        "path_data_csv_in",
        "path_data_csv_out",
        "path_data_csv_out_mcmc",
        "path_data_csv_out_optim_map",
        "path_data_csv_out_optim_mle",
        "N",
        "D",
        "D_no_kappa",
        "L",
        "list_names",
        "list_names_plot",
        "cloud_name",
        "max_workers",
        "list_lines_fit",
        "list_lines_valid",
        "Theta_true_scaled",
        "map_shaper",
        "plots_estimator",
    )

    def setup(
        self,
        forward_model_name: str,
        force_use_cpu: bool,
        fixed_params: Dict[str, Optional[float]],
        is_log_scale_params: Dict[str, bool],
        #
        data_int_path: str,
        data_err_path: str,
        sigma_m_float: float,
        #
        indicator_margin_scale: float,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
        #
        with_spatial_prior: bool = True,
        spatial_prior_params: Union[None, SpatialPriorParams] = None,
        list_gaussian_approx_params: List[bool] = [],
        list_mixing_model_params: List[Dict[str, str]] = [],
    ):
        self.list_lines_valid = []

        (
            df_int_fit,
            y_fit,
            sigma_a_fit,
            omega_fit,
            y_valid,
            sigma_a_valid,
            omega_valid,
        ) = self.setup_observation(
            data_int_path=data_int_path,
            data_err_path=data_err_path,
        )
        scaler, forward_map = self.setup_forward_map(
            forward_model_name=forward_model_name,
            force_use_cpu=force_use_cpu,
            dict_fixed_params=fixed_params,
            dict_is_log_scale_params=is_log_scale_params,
        )
        sigma_m_fit = sigma_m_float * np.ones((self.N, self.L))

        # run setup
        print(f"lower_bounds_lin = {lower_bounds_lin}")

        dict_posteriors, scaler, prior_indicator_1pix = self.setup_posteriors(
            scaler=scaler,
            forward_map=forward_map,
            y=y_fit,
            sigma_a=sigma_a_fit,
            sigma_m=sigma_m_fit,
            omega=omega_fit,
            syn_map=df_int_fit,
            with_spatial_prior=with_spatial_prior,
            spatial_prior_params=spatial_prior_params,
            indicator_margin_scale=indicator_margin_scale,
            lower_bounds_lin=lower_bounds_lin,
            upper_bounds_lin=upper_bounds_lin,
            list_gaussian_approx_params=list_gaussian_approx_params,
            list_mixing_model_params=list_mixing_model_params,
        )
        return (
            dict_posteriors,
            scaler,
            prior_indicator_1pix,
            y_valid,
            sigma_a_valid,
            omega_valid,
        )

    def main(
        self,
        params: dict,
        path_data_cloud: str,
        point_challenger: dict = {},
    ) -> None:
        if params["with_spatial_prior"]:
            spatial_prior_params = SpatialPriorParams(
                **params["spatial_prior"],
            )
        else:
            spatial_prior_params = None

        sigma_m_float = np.log(params["sigma_m_float_linscale"])
        (
            dict_posteriors,
            scaler,
            prior_indicator_1pix,
            y_valid,
            sigma_a_valid,
            omega_valid,
        ) = self.setup(
            **params["forward_model"],
            #
            data_int_path=f"{path_data_cloud}/{params['filename_int']}",
            data_err_path=f"{path_data_cloud}/{params['filename_err']}",
            sigma_m_float=sigma_m_float,
            #
            **params["prior_indicator"],
            #
            with_spatial_prior=params["with_spatial_prior"],
            spatial_prior_params=spatial_prior_params,
            list_gaussian_approx_params=params["list_gaussian_approx_params"],
            list_mixing_model_params=[
                {"path_transition_params": f"{path_data_cloud}/{filename}"}
                for filename in params["mixing_model_params_filename"]
            ],
        )
        self.save_and_plot_setup(
            dict_posteriors,
            params["prior_indicator"]["lower_bounds_lin"],
            params["prior_indicator"]["upper_bounds_lin"],
            scaler,
        )

        # * Optim MAP
        if params["to_run_optim_map"]:
            list_model_names = self.inversion_optim_map(
                dict_posteriors=dict_posteriors,
                scaler=scaler,
                my_sampler_params=MySamplerParams(**params["sampling_params"]["map"]),
                can_run_in_parallel=params["forward_model"]["force_use_cpu"],
                **params["run_params"]["map"],
            )

        # * MCMC
        if params["to_run_mcmc"]:
            list_model_names = self.inversion_mcmc(
                dict_posteriors=dict_posteriors,
                scaler=scaler,
                my_sampler_params=MySamplerParams(**params["sampling_params"]["mcmc"]),
                can_run_in_parallel=params["forward_model"]["force_use_cpu"],
                #
                y_valid=y_valid,
                sigma_a_valid=sigma_a_valid,
                omega_valid=omega_valid,
                sigma_m_valid=sigma_m_float * np.ones_like(sigma_a_valid),
                #
                point_challenger=point_challenger,
                **params["run_params"]["mcmc"],
            )

        return
