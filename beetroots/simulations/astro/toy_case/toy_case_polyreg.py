import os
from typing import Dict, List, Optional, Union

import numpy as np

from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.sampler.utils.my_sampler_params import MySamplerParams
from beetroots.simulations.abstract_simulation import Simulation
from beetroots.simulations.astro import data_validation
from beetroots.simulations.astro.forward_map.abstract_poly_reg import (
    SimulationPolynomialReg,
)
from beetroots.simulations.astro.observation.abstract_toy_case import SimulationToyCase
from beetroots.simulations.astro.posterior_type.abstract_direct import (
    SimulationMySampler,
)


class SimulationToyCasePolyReg(
    SimulationPolynomialReg, SimulationToyCase, SimulationMySampler
):
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
        fixed_params: Dict[str, Optional[float]],
        is_log_scale_params: Dict[str, bool],
        #
        sigma_a_float: float,
        sigma_m_float: float,
        omega_float: float,
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

        scaler, forward_map = self.setup_forward_map(
            forward_model_name=forward_model_name,
            dict_fixed_params=fixed_params,
            dict_is_log_scale_params=is_log_scale_params,
        )

        sigma_a = sigma_a_float * np.ones((self.N, self.L))
        sigma_m = sigma_m_float * np.ones((self.N, self.L))
        omega = omega_float * np.ones((self.N, self.L))

        syn_map, y = self.setup_observation(
            scaler=scaler,
            forward_map=forward_map,
            sigma_a=sigma_a,
            sigma_m=sigma_m,
            omega=omega,
        )

        # run setup
        dict_posteriors, scaler, prior_indicator_1pix = self.setup_posteriors(
            scaler=scaler,
            forward_map=forward_map,
            y=y,
            sigma_a=sigma_a,
            sigma_m=sigma_m,
            omega=omega,
            syn_map=syn_map,
            with_spatial_prior=with_spatial_prior,
            spatial_prior_params=spatial_prior_params,
            indicator_margin_scale=indicator_margin_scale,
            lower_bounds_lin=lower_bounds_lin,
            upper_bounds_lin=upper_bounds_lin,
            list_gaussian_approx_params=list_gaussian_approx_params,
            list_mixing_model_params=list_mixing_model_params,
        )

        y_valid = None
        sigma_a_valid = None
        omega_valid = None

        return (
            dict_posteriors,
            scaler,
            prior_indicator_1pix,
            y_valid,
            sigma_a_valid,
            omega_valid,
        )

    def main(self, params: dict, path_data_cloud: str) -> None:
        if params["with_spatial_prior"]:
            spatial_prior_params = SpatialPriorParams(
                **params["spatial_prior"],
            )
        else:
            spatial_prior_params = None

        (
            dict_posteriors,
            scaler,
            prior_indicator_1pix,
            y_valid,  # None
            sigma_a_valid,  # None
            omega_valid,  # None
        ) = simulation.setup(
            **params["forward_model"],
            #
            sigma_a_float=params["sigma_a_float"],
            sigma_m_float=np.log(params["sigma_m_float_linscale"]),
            omega_float=3 * params["sigma_a_float"],
            #
            **params["prior_indicator"],
            #
            with_spatial_prior=params["with_spatial_prior"],
            spatial_prior_params=spatial_prior_params,
            #
            list_gaussian_approx_params=params["list_gaussian_approx_params"],
            list_mixing_model_params=[
                {"path_transition_params": f"{path_data_cloud}/{filename}"}
                for filename in params["mixing_model_params_filename"]
            ],
        )
        simulation.save_and_plot_setup(
            dict_posteriors,
            params["prior_indicator"]["lower_bounds_lin"],
            params["prior_indicator"]["upper_bounds_lin"],
            scaler,
        )
        # * Optim MAP
        if params["to_run_optim_map"]:
            simulation.inversion_optim_map(
                dict_posteriors=dict_posteriors,
                scaler=scaler,
                my_sampler_params=MySamplerParams(**params["sampling_params"]["map"]),
                can_run_in_parallel=params["forward_model"]["force_use_cpu"],
                **params["run_params"]["map"],
            )

        # * MCMC
        if params["to_run_mcmc"]:
            simulation.inversion_mcmc(
                dict_posteriors=dict_posteriors,
                scaler=scaler,
                my_sampler_params=MySamplerParams(**params["sampling_params"]["mcmc"]),
                can_run_in_parallel=params["forward_model"]["force_use_cpu"],
                **params["run_params"]["mcmc"],
            )
        return


if __name__ == "__main__":
    (
        yaml_file,
        path_data,
        path_models,
        path_outputs,
    ) = SimulationToyCasePolyReg.parse_args()

    params = SimulationToyCasePolyReg.load_params(path_data, yaml_file)

    SimulationToyCasePolyReg.check_input_params_file(
        params,
        data_validation.schema,
    )

    simulation = SimulationToyCasePolyReg(
        **params["simu_init"],
        yaml_file=yaml_file,
        path_data=path_data,
        path_outputs=path_outputs,
        path_models=path_models,
        forward_model_fixed_params=params["forward_model"]["fixed_params"],
    )

    simulation.main(
        params=params,
        path_data_cloud=path_data,
    )
