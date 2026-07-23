# DEPRECATED:

import os
from typing import Dict, Optional

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

import importlib

from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.modelling.target_distribution.full_conditional import FullConditional
from beetroots.sampler.utils.sampler_params import MyGibbsSamplerParams
from beetroots.simulations.astro import data_validation
from beetroots.simulations.astro.forward_map_setup.abstract_nn import SimulationNN
from beetroots.simulations.astro.observation_setup.abstract_toy_case import (
    SimulationToyCase,
)
from beetroots.simulations.astro.sampler_setup.abstract_sampler_hierarchical import (
    SimulationMyGibbsSampler,
)
from beetroots.space_transform.log_transform import LogScaler


class SimulationToyCaseNNHierachical(
    SimulationNN, SimulationToyCase, SimulationMyGibbsSampler
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

    def setup_full_conditionals(
        self,
        scaler,
        forward_map,
        y,
        sigma_a,
        sigma_m,
        omega,
        syn_map,
        params_component_distributions: dict[str, dict],
        dict_target_distributions_match_components: dict[str, list],
        mult_f_Theta=None,
    ) -> None:
        # component distributions
        component_distributions_names = params_component_distributions.keys()
        component_distributions = dict()

        if "theta_indicator_prior" in component_distributions_names:
            dic = params_component_distributions["theta_indicator_prior"]
            module_ = importlib.import_module(dic["module"])
            class_ = getattr(module_, dic["class_name"])
            kwargs = dic["params"]
            lower_bounds_lin = xp.array(kwargs["lower_bounds_lin"])  # Linear space
            upper_bounds_lin = xp.array(kwargs["upper_bounds_lin"])  # Linear space

            kwargs.update(
                {
                    "D": self.D_sampling,
                    "L": self.L,
                    "N": self.N,
                    "list_idx_sampling": self.list_idx_sampling_param_interest,
                }
            )

            if dic[
                "do_scaling"
            ]:  # TODO: check if really needed or if we could do it right away without checking the condition
                lower_bounds = scaler.from_lin_to_scaled(
                    lower_bounds_lin.reshape((1, self.D)),
                ).flatten()
                upper_bounds = scaler.from_lin_to_scaled(
                    upper_bounds_lin.reshape((1, self.D)),
                ).flatten()
            kwargs["lower_bounds"] = lower_bounds
            kwargs["upper_bounds"] = upper_bounds

            component_distributions["theta_indicator_prior"] = class_(**kwargs)
        else:
            raise ValueError("'theta_indicator_prior' must be a component distribution")

        if "theta_spatial_prior" in component_distributions_names:
            dic = params_component_distributions["theta_spatial_prior"]
            module_ = importlib.import_module(dic["module"])
            class_ = getattr(module_, dic["class_name"])
            kwargs = dic["params"]
            kwargs["cloud_name"] = self.cloud_name
            kwargs["list_idx_sampling"] = self.list_idx_sampling_param_interest
            kwargs["df"] = syn_map
            kwargs["spatial_prior_params"] = SpatialPriorParams(
                kwargs["name"],
                kwargs["use_next_nearest_neighbors"],
                kwargs["initial_regu_weights"],
            )
            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N})
            component_distributions["theta_spatial_prior"] = class_(**kwargs)

            dict_sites = component_distributions["theta_spatial_prior"].dict_sites
            list_edges = component_distributions["theta_spatial_prior"].list_edges
            weights = component_distributions["theta_spatial_prior"].weights
        else:
            dict_sites, list_edges, weights = None, None, None
            raise Warning(
                "no spatial prior for theta will be used as no component distribution with the key 'theta_spatial_prior' has been provided."
            )

        if "log_aux_given_theta" in component_distributions_names:
            dic = params_component_distributions["log_aux_given_theta"]
            module_ = importlib.import_module(dic["module"])
            class_ = getattr(module_, dic["class_name"])
            kwargs = dic["params"]
            kwargs["forward_map"] = forward_map
            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N})
            kwargs["sigma_m"] = sigma_m

            component_distributions["log_aux_given_theta"] = class_(**kwargs)
        else:
            raise ValueError("'log_aux_given_theta' must be a component distribution")

        if "obs_given_log_aux" in component_distributions_names:
            dic = params_component_distributions["obs_given_log_aux"]
            module_ = importlib.import_module(dic["module"])
            class_ = getattr(module_, dic["class_name"])
            kwargs = dic["params"]
            kwargs["y"] = y
            kwargs["sigma_a"] = sigma_a
            kwargs["omega"] = omega
            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N})
            component_distributions["obs_given_log_aux"] = class_(**kwargs)
        else:
            raise ValueError("'obs_given_log_aux' must be a component distribution")

        # target distributions
        target_distributions = {}

        for (
            target_dist_name,
            list_component_dists,
        ) in dict_target_distributions_match_components.items():
            component_distributions_target = {
                component_dist_name: component_distributions[component_dist_name]
                for component_dist_name in list_component_dists
            }
            if "theta" in target_dist_name:
                var_shape = (self.N, self.D_sampling)
            elif "log_aux" in target_dist_name:
                var_shape = (self.N, self.L)
            else:
                raise ValueError(
                    f"target_dist_name must contain 'theta' or 'log_aux' but got {target_dist_name}"
                )
            separable, sites = (
                (False, dict_sites) if "theta" in target_dist_name else (True, None)
            )
            target_distribution = FullConditional(
                D=self.D_sampling,
                L=self.L,
                N=self.N,
                distribution_components=component_distributions_target,
                var_name=target_dist_name,
                var_shape=var_shape,
                dict_sites=sites,
                separable=separable,
            )
            target_distributions[target_dist_name] = target_distribution

        dict_models = {
            "hierarchical_log_aux_theta": target_distributions
        }  # We could instantiate several versions of the problem (different hyperparams for example similarly to the posterior model with the approximation)

        # Proposal distributions parameters
        proposal_distribution_params_theta = {
            "list_edges": list_edges,
            "weights": weights,
            "upper_bounds": upper_bounds[self.list_idx_sampling_param_interest],
            "lower_bounds": lower_bounds[self.list_idx_sampling_param_interest],
            "indicator_margin_scale": params_component_distributions[
                "theta_indicator_prior"
            ]["params"]["indicator_margin_scale"],
        }
        # proposal_distribution_params_log_aux = {
        #     "list_edges": list_edges,
        #     "weights": 100
        #     * xp.ones(
        #         y.shape[-1]
        #     ),  # xp.mean(xp.log(y), axis=0) ** 2 / xp.var(xp.log(y), axis=0),
        # }
        # proposal_distribution_params_log_aux = {
        #     "std": 0.001 * xp.ones_like(y),  # sigma_a,
        #     "weights": False,
        # }
        proposal_distribution_params_log_aux = {
            "s_m": sigma_m,
        }
        # proposal_distribution_params_log_aux = {
        #     "s_m": sigma_m,
        #     "s_a": sigma_a,
        #     "omega": omega,
        #     "exponential_transform": True,
        # }

        kwargs_proposal_distributions = {
            "full_conditional_theta": proposal_distribution_params_theta,
            "full_conditional_log_auxiliary": proposal_distribution_params_log_aux,
        }

        # Set scalers
        scalers = {
            "full_conditional_theta": scaler,
            "full_conditional_log_auxiliary": LogScaler(),
        }

        ## -- SET VALUES FOR RESULTS EXTRACTION -- ##
        # Set true values
        self.full_cond_aux_true = (
            scalers["full_conditional_log_auxiliary"].from_lin_to_scaled(mult_f_Theta)
            if mult_f_Theta is not None
            else None
        )
        self.Vars_true_scaled = {key: None for key in target_distributions.keys()}
        self.Vars_true_scaled["full_conditional_theta"] = self.Theta_true_scaled
        self.Vars_true_scaled["full_conditional_log_auxiliary"] = (
            self.full_cond_aux_true
        )
        # list_var_names_plot
        self.list_vars_names_plots = {key: None for key in target_distributions.keys()}
        self.list_vars_names_plots["full_conditional_theta"] = self.list_names_plots
        self.list_vars_names_plots["full_conditional_log_auxiliary"] = (
            self.list_lines_fit
        )
        # Set list_vars_idx_sampling
        self.list_vars_idx_sampling = {key: None for key in target_distributions.keys()}
        self.list_vars_idx_sampling["full_conditional_theta"] = (
            self.list_idx_sampling_param_interest
        )
        self.list_vars_idx_sampling["full_conditional_log_auxiliary"] = list(
            range(self.L)
        )
        # Set list_fixed_values
        self.list_vars_fixed_values = {key: None for key in target_distributions.keys()}
        self.list_vars_fixed_values["full_conditional_theta"] = self.list_fixed_values
        self.list_vars_fixed_values["full_conditional_log_auxiliary"] = [
            None for _ in range(self.L)
        ]

        params_plot_setup = {
            "dict_sites_": dict_sites,
            "y": y * 1,
            "sigma_a": sigma_a * 1,
            "omega": omega * 1,
        }

        # NOTE: lower and upper bounds in linear scale ! If not provided it's gonna search in an indicator prior to get the bounds in scaled space and the transform them.
        params_plots_estimator = {
            "full_conditional_theta": {
                "lower_bounds_lin": lower_bounds_lin,
                "upper_bounds_lin": upper_bounds_lin,
                "scaler": scalers["full_conditional_theta"],
                "Var_true_scaled": self.Vars_true_scaled["full_conditional_theta"],
                "list_idx_sampling": self.list_vars_idx_sampling[
                    "full_conditional_theta"
                ],
                "list_names_plots": self.list_vars_names_plots[
                    "full_conditional_theta"
                ],
                "list_fixed_values": self.list_vars_fixed_values[
                    "full_conditional_theta"
                ],
                "sub_var": "full_conditional_theta",
            },
            "full_conditional_log_auxiliary": {
                "lower_bounds_lin": y.min(axis=0)
                * 0.9,  # scalers["full_conditional_log_auxiliary"].from_lin_to_scaled(y).min(axis=0) * 0.9,
                "upper_bounds_lin": y.max(axis=0)
                * 1.1,  # scalers["full_conditional_log_auxiliary"].from_lin_to_scaled(y).max(axis=0) * 1.1,
                "scaler": scalers["full_conditional_log_auxiliary"],
                "Var_true_scaled": self.Vars_true_scaled[
                    "full_conditional_log_auxiliary"
                ],
                "list_idx_sampling": self.list_vars_idx_sampling[
                    "full_conditional_log_auxiliary"
                ],
                "list_names_plots": self.list_vars_names_plots[
                    "full_conditional_log_auxiliary"
                ],
                "list_fixed_values": self.list_vars_fixed_values[
                    "full_conditional_log_auxiliary"
                ],
                "sub_var": "full_conditional_log_auxiliary",
            },
        }

        return {
            "dict_models": dict_models,
            "scalers": scalers,
            "params_plot_setup": params_plot_setup,
            "params_plots_estimator": params_plots_estimator,
            "kwargs_proposal_distributions": kwargs_proposal_distributions,
        }

    def setup(
        self,
        forward_model_name: str,
        force_use_cpu: bool,
        fixed_params: Dict[str, Optional[float]],
        is_log_scale_params: Dict[str, bool],
        #
        sigma_a_float: float,
        sigma_m_float: float,
        omega_float: float,
        #
        params_component_distributions,
        dict_target_distributions_match_components,
    ):
        self.N = int(self.cloud_name.split("N")[1]) ** 2

        self.list_lines_valid = []

        scaler, forward_map = self.setup_forward_map(
            forward_model_name=forward_model_name,
            force_use_cpu=force_use_cpu,
            dict_fixed_params=fixed_params,
            dict_is_log_scale_params=is_log_scale_params,
        )

        sigma_a = sigma_a_float * xp.ones((self.N, self.L))
        sigma_m = sigma_m_float * xp.ones((self.N, self.L))
        omega = omega_float * xp.ones((self.N, self.L))

        self.sigma_m = sigma_m

        syn_map, y, mult_f_Theta, Theta_true_scaled, censored, forward_map = (
            self.setup_observation(
                scaler=scaler,
                forward_map=forward_map,
                sigma_a=sigma_a,
                sigma_m=sigma_m,
                omega=omega,
                return_true_scaled_params=True,
            )
        )
        self.y = y * 1

        # run setup
        (
            dict_models,
            scalers,
            params_plot_setup,
            params_plots_estimator,
            kwargs_proposal_distributions,
        ) = self.setup_target_distribution(
            scaler=scaler,
            forward_map=forward_map,
            y=y,
            sigma_a=sigma_a,
            sigma_m=sigma_m,
            omega=omega,
            syn_map=syn_map,
            params_component_distributions=params_component_distributions,
            dict_target_distributions_match_components=dict_target_distributions_match_components,
            mult_f_Theta=mult_f_Theta,
        ).values()

        return (
            dict_models,
            scalers,
            params_plot_setup,
            params_plots_estimator,
            kwargs_proposal_distributions,
            mult_f_Theta,  # to initialize the auxiliary variable.
            Theta_true_scaled,
            censored,
            forward_map,
        )

    def main(self, params: dict, path_data_cloud: str) -> None:
        (
            dict_models,
            scalers,
            params_plot_setup,
            params_plots_estimator,
            kwargs_proposal_distributions,
            mult_f_Theta,  # to initialize the auxiliary variable.
            Theta_true_scaled,  # to initialize the physical parameters (theta).
            censored,
            forward_map,
        ) = simulation.setup(
            **params["forward_model"],
            #
            sigma_a_float=params["sigma_a_float"],
            sigma_m_float=xp.log(params["sigma_m_float_linscale"]),
            omega_float=3 * params["sigma_a_float"],
            #
            params_component_distributions=params["component_distributions"],
            dict_target_distributions_match_components=params["target_distributions"],
        )

        for model_name in list(dict_models.keys()):
            folder_path = f"{self.path_raw}/{model_name}"
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)

        simulation.save_and_plot_setup(
            **params_plot_setup,
        )
        self.plots_estimator = dict.fromkeys(params_plots_estimator.keys())
        for k in self.plots_estimator.keys():
            self.plots_estimator[k] = simulation.save_and_plot_estimator(
                **params_plots_estimator[k],
            )

        # * Optim MAP
        if params["to_run_optim_map"]:
            raise ValueError("optimization not implemented yet.")

        # * MCMC
        if params["to_run_mcmc"]:
            param_interest_name = params["sampling_params"]["mcmc"][
                "param_interest_name"
            ]

            if (
                params["debugging"]["sampling_start"]["full_conditional_log_auxiliary"]
                == "obs"
            ):
                params["run_params"]["mcmc"]["start_from"][
                    "full_conditional_log_auxiliary"
                ] = scalers["full_conditional_log_auxiliary"].from_lin_to_scaled(
                    self.y
                )  # Set starting vector for the auxiliary variable.
            elif (
                params["debugging"]["sampling_start"]["full_conditional_log_auxiliary"]
                == "omega"
            ):
                params["run_params"]["mcmc"]["start_from"][
                    "full_conditional_log_auxiliary"
                ] = xp.log(3 * params["sigma_a_float"]) * xp.ones(
                    (self.N, self.L)
                )  # Set starting vector for the auxiliary variable (set to omega).
            elif (
                params["debugging"]["sampling_start"]["full_conditional_log_auxiliary"]
                == "true"
            ):
                params["run_params"]["mcmc"]["start_from"][
                    "full_conditional_log_auxiliary"
                ] = scalers["full_conditional_log_auxiliary"].from_lin_to_scaled(
                    mult_f_Theta
                )  # Set starting vector for the auxiliary variable.
            elif (
                params["debugging"]["sampling_start"]["full_conditional_log_auxiliary"]
                == "random"
            ):
                params["run_params"]["mcmc"]["start_from"]["full_conditional_theta"] = (
                    xp.random.rand(self.N, self.D_sampling)
                )  # Set starting vector for the physical parameters.
                mult_f_Theta_start = xp.random.lognormal(
                    mean=-(self.sigma_m**2) / 2,
                    sigma=self.sigma_m,
                ) * forward_map.evaluate(
                    params["run_params"]["mcmc"]["start_from"]["full_conditional_theta"]
                )  # Set starting vector for the auxiliary variable.
                params["run_params"]["mcmc"]["start_from"][
                    "full_conditional_log_auxiliary"
                ] = scalers["full_conditional_log_auxiliary"].from_lin_to_scaled(
                    mult_f_Theta_start
                )

            if (
                params["debugging"]["sampling_start"]["full_conditional_theta"]
                == "true"
            ):
                params["run_params"]["mcmc"]["start_from"]["full_conditional_theta"] = (
                    Theta_true_scaled * 1
                )  # Set starting vector for the physical parameters.
            for k in kwargs_proposal_distributions.keys():
                params["sampling_params"]["mcmc"]["proposal_distributions_mtm_params"][
                    k
                ]["params"].update(kwargs_proposal_distributions[k])
            simulation.inversion_mcmc(
                dict_models=dict_models,
                scalers=scalers,
                my_sampler_params=MyGibbsSamplerParams(
                    **params["sampling_params"]["mcmc"]
                ),
                can_run_in_parallel=params["forward_model"]["force_use_cpu"],
                param_interest_name=param_interest_name,
                debugging=params.get("debugging", dict()),
                **params["run_params"]["mcmc"],
            )
        return


if __name__ == "__main__":
    yaml_file, path_data, path_models, path_outputs = (
        SimulationToyCaseNNHierachical.parse_args()
    )

    params = SimulationToyCaseNNHierachical.load_params(path_data, yaml_file)

    SimulationToyCaseNNHierachical.check_input_params_file(
        params,
        data_validation.schema_astro_hierarchical,
    )

    simulation = SimulationToyCaseNNHierachical(
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
