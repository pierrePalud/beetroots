import os
import sys
import warnings
from typing import Dict, List, Optional

import numba as nb
import numpy as np

from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

import importlib
from pathlib import Path

import scipy.stats as stats

from beetroots.modelling.forward_maps.abstract_base import ForwardMap
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

    def setup_vars_0(
        self,
        start_type_Theta: str,
        start_type_LogXi: str,
        y: xp.ndarray,
        mult_f_Theta: xp.ndarray,
        Theta_true_scaled: xp.ndarray,
        forward_map: ForwardMap,
        scalers: dict,
        sigma_m: xp.ndarray,
        sigma_a: xp.ndarray,
        censored: xp.ndarray,
        rng_list: List[xp.random.Generator],
        N_MCMC: int,
        indicator_prior: Optional[SmoothIndicatorPrior] = None,
    ) -> dict:
        def _broadcast_or_validate(
            arr: xp.ndarray,
            target_shape_no_chain: tuple[int, int],
            var_name: str,
        ) -> xp.ndarray:
            if arr.ndim == 2 and arr.shape == target_shape_no_chain:
                return xp.repeat(arr[None, :, :], N_MCMC, axis=0)
            if arr.ndim == 3 and arr.shape == (N_MCMC,) + target_shape_no_chain:
                return arr
            raise ValueError(
                f"Invalid shape for {var_name}: got {arr.shape}. Expected either {target_shape_no_chain} or {(N_MCMC,) + target_shape_no_chain}."
            )

        if len(rng_list) != N_MCMC:
            raise ValueError(
                f"Expected {N_MCMC} random generators, got {len(rng_list)}."
            )

        theta_shape = (self.N, self.D_sampling)
        log_xi_shape = (self.N, self.L)

        # Initialize Theta for all chains
        if start_type_Theta == "true":
            theta_init = xp.asarray(Theta_true_scaled * 1)
            theta_values = xp.repeat(theta_init[None, :, :], N_MCMC, axis=0)
            print(
                f"Initializing Theta for all N_MCMC={N_MCMC} chains with the true scaled values (same initialization for all chains)."
            )
        elif Path(start_type_Theta).exists():
            if Path(start_type_Theta).suffix.lower() != ".csv":
                raise ValueError(
                    f"Theta start file must be a .csv file, got: {start_type_Theta}"
                )
            theta_loaded = self._load_csv_array(start_type_Theta, key="Theta")
            theta_loaded_scaled = scalers["full_conditional_theta"].from_lin_to_scaled(
                np.concatenate(
                    [theta_loaded, np.zeros((theta_loaded.shape[0], 1))], axis=1
                )
            )[:, :-1]
            theta_values = _broadcast_or_validate(
                theta_loaded_scaled,
                theta_shape,
                "full_conditional_theta",
            )
            print(
                f"Initializing Theta from file {start_type_Theta} (same initialization for all chains if the input array fits the variable's shape, different initializations if the file contains an additional dimension for each chain)."
            )
        elif start_type_Theta == "random":
            if indicator_prior is not None:
                print(
                    f"Initializing Theta randomly from indicator prior bounds (applied independently to all N_MCMC={N_MCMC} chains)."
                )
                theta_chains = [
                    rng_list[n].uniform(0.0, 1.0, size=theta_shape)
                    * (
                        indicator_prior.upper_bounds[None, :]
                        - indicator_prior.lower_bounds[None, :]
                    )
                    + indicator_prior.lower_bounds[None, :]
                    for n in range(N_MCMC)
                ]
            else:
                print(
                    f"Initializing Theta randomly from a standard Gaussian (applied independently to all N_MCMC={N_MCMC} chains)."
                )
                theta_chains = [
                    rng_list[n].standard_normal(size=theta_shape) for n in range(N_MCMC)
                ]
            theta_values = xp.asarray(theta_chains)
        else:
            warnings.warn(
                "Invalid start_type_Theta or not provided. Falling back to random standard Gaussian initialization for all chains."
            )
            theta_chains = [
                rng_list[n].standard_normal(size=theta_shape) for n in range(N_MCMC)
            ]
            theta_values = xp.asarray(theta_chains)

        # Initialize log_xi for all chains
        if start_type_LogXi == "true":
            f_theta_true = forward_map.evaluate(Theta_true_scaled)
            xi_true = mult_f_Theta / f_theta_true
            log_xi_init = scalers["full_conditional_log_xi"].from_lin_to_scaled(xi_true)
            log_xi_values = xp.repeat(log_xi_init[None, :, :], N_MCMC, axis=0)
            print(
                f"Initializing log_xi for all N_MCMC={N_MCMC} chains with the true values."
            )
        elif Path(start_type_LogXi).exists():
            if Path(start_type_LogXi).suffix.lower() != ".csv":
                raise ValueError(
                    f"log_xi start file must be a .csv file, got: {start_type_LogXi}"
                )
            xi_loaded = self._load_csv_array(start_type_LogXi, key="log_xi")
            log_xi_loaded = scalers["full_conditional_log_xi"].from_lin_to_scaled(
                xp.maximum(xi_loaded, 1e-12)
            )
            log_xi_values = _broadcast_or_validate(
                log_xi_loaded,
                log_xi_shape,
                "full_conditional_log_xi",
            )
            print(
                f"Initializing log_xi from file {start_type_LogXi} (same initialization for all chains if the input array fits the variable's shape, different initializations if the file contains an additional dimension for each chain)."
            )
        elif start_type_LogXi == "random":
            if start_type_Theta != "random":
                warnings.warn(
                    "'start_type_LogXi' is 'random' while 'start_type_Theta' is not. Random multiplicative-noise initialization is applied independently for log_xi."
                )

            log_xi_chains = []
            for n in range(N_MCMC):
                # xi_start = rng_list[n].lognormal(
                #     mean=-(sigma_m**2) / 2,
                #     sigma=sigma_m,
                # )
                f_theta = forward_map.evaluate(theta_values[n])

                sigma = xp.where(
                    sigma_a < f_theta * (xp.exp(sigma_m**2) - 1) ** 0.5,
                    sigma_a / f_theta,  # No log since close to 1
                    sigma_m,
                )

                mean = xp.log(y) - xp.log(f_theta) - (sigma**2) / 2

                # Broadcast censor fallback along the MTM axis without allocating repeated arrays.
                mean[censored] = -(sigma_m[censored] ** 2) / 2
                sigma[censored] = sigma_m[censored]

                log_xi_n = mean + sigma * rng_list[n].standard_normal(size=mean.shape)
                # xi_start = xp.clip(
                #     y / f_theta,
                #     stats.lognorm.ppf(0.05, s=sigma_m, scale=xp.exp(-(sigma_m**2) / 2)),
                #     stats.lognorm.ppf(0.95, s=sigma_m, scale=xp.exp(-(sigma_m**2) / 2)),
                # )
                # log_xi_n = scalers["full_conditional_log_xi"].from_lin_to_scaled(
                #     xp.maximum(xi_start, 1e-12)
                # )
                log_xi_chains.append(log_xi_n)
            log_xi_values = xp.asarray(log_xi_chains)
            print(
                f"Initializing log_xi for all N_MCMC={N_MCMC} chains with random multiplicative perturbations (applied independently to all chains)."
            )
        elif start_type_LogXi == "observations":
            log_xi_chains = []
            for n in range(N_MCMC):
                theta_n = theta_values[n]
                f_theta_n = xp.maximum(forward_map.evaluate(theta_n), 1e-12)
                xi_obs = xp.maximum(y / f_theta_n, 1e-12)
                log_xi_chains.append(
                    scalers["full_conditional_log_xi"].from_lin_to_scaled(xi_obs)
                )
            log_xi_values = xp.asarray(log_xi_chains)
            print(
                f"Initializing log_xi for all N_MCMC={N_MCMC} chains from the ratio y/f(theta) (chain-wise with each theta initialization)."
            )
        else:
            warnings.warn(
                "Invalid start_type_LogXi or not provided. Falling back to random multiplicative-noise initialization for log_xi."
            )
            log_xi_chains = []
            for n in range(N_MCMC):
                xi_start = rng_list[n].lognormal(
                    mean=-(sigma_m**2) / 2,
                    sigma=sigma_m,
                )
                log_xi_n = scalers["full_conditional_log_xi"].from_lin_to_scaled(
                    xp.maximum(xi_start, 1e-12)
                )
                log_xi_chains.append(log_xi_n)
            log_xi_values = xp.asarray(log_xi_chains)

        start_from = {
            seed: {
                "full_conditional_theta": theta_values[seed],
                "full_conditional_log_xi": log_xi_values[seed],
            }
            for seed in range(N_MCMC)
        }

        return start_from

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
        else:
            dict_sites = None
            warnings.warn(
                "no spatial prior for theta will be used as no component distribution with the key 'theta_spatial_prior' has been provided."
            )

        if "obs_given_target_log_xi" in component_distributions_names:
            dic = params_component_distributions["obs_given_target_log_xi"]
            module_ = importlib.import_module(dic["module"])
            class_ = getattr(module_, dic["class_name"])
            kwargs = dic["params"]
            kwargs["forward_map"] = forward_map
            kwargs["y"] = y
            kwargs["sigma_a"] = sigma_a
            kwargs["sigma_m"] = sigma_m
            kwargs["omega"] = omega
            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N})

            component_distributions["obs_given_target_log_xi"] = class_(**kwargs)
        else:
            raise ValueError(
                "'obs_given_target_log_xi' must be a component distribution"
            )

        if "log_xi_prior" in component_distributions_names:
            dic = params_component_distributions["log_xi_prior"]
            module_ = importlib.import_module(dic["module"])
            class_ = getattr(module_, dic["class_name"])
            kwargs = dic["params"]
            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N})
            component_distributions["log_xi_prior"] = class_(**kwargs)
        else:
            raise ValueError("'log_xi_prior' must be a component distribution")

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
            elif "log_xi" in target_dist_name:
                var_shape = (self.N, self.L)
            else:
                raise ValueError(
                    f"target_dist_name must contain 'theta' or 'log_xi' but got {target_dist_name}"
                )

            target_distribution = FullConditional(
                D=self.D_sampling,
                L=self.L,
                N=self.N,
                distribution_components=component_distributions_target,
                var_name=target_dist_name,
                var_shape=var_shape,
                dict_sites=dict_sites,
            )
            target_distributions[target_dist_name] = target_distribution

        # Set scalers
        scalers = {
            "full_conditional_theta": scaler,
            "full_conditional_log_xi": LogScaler(),
        }

        ## -- SET VALUES FOR RESULTS EXTRACTION -- ##
        # Set true values
        f_theta_true = xp.maximum(forward_map.evaluate(self.Theta_true_scaled), 1e-12)
        xi_true = xp.maximum(mult_f_Theta / f_theta_true, 1e-12)
        self.full_cond_log_xi_true = scalers[
            "full_conditional_log_xi"
        ].from_lin_to_scaled(xi_true)
        self.Vars_true_scaled = {key: None for key in target_distributions.keys()}
        self.Vars_true_scaled["full_conditional_theta"] = self.Theta_true_scaled
        self.Vars_true_scaled["full_conditional_log_xi"] = self.full_cond_log_xi_true
        # list_var_names_plot
        self.list_vars_names_plots = {key: None for key in target_distributions.keys()}
        self.list_vars_names_plots["full_conditional_theta"] = self.list_names_plots
        self.list_vars_names_plots["full_conditional_log_xi"] = self.list_lines_fit
        # Set list_vars_idx_sampling
        self.list_vars_idx_sampling = {key: None for key in target_distributions.keys()}
        self.list_vars_idx_sampling["full_conditional_theta"] = (
            self.list_idx_sampling_param_interest
        )
        self.list_vars_idx_sampling["full_conditional_log_xi"] = list(range(self.L))
        # Set list_fixed_values
        self.list_vars_fixed_values = {key: None for key in target_distributions.keys()}
        self.list_vars_fixed_values["full_conditional_theta"] = self.list_fixed_values
        self.list_vars_fixed_values["full_conditional_log_xi"] = [
            None for _ in range(self.L)
        ]

        params_plot_setup = {
            "dict_sites_": dict_sites,
            "y": y * 1,
            "sigma_a": sigma_a * 1,
            "omega": omega * 1,
        }

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
            "full_conditional_log_xi": {
                "lower_bounds_lin": xi_true.min(axis=0) * 0.9,
                "upper_bounds_lin": xi_true.max(axis=0) * 1.1,
                "scaler": scalers["full_conditional_log_xi"],
                "Var_true_scaled": self.Vars_true_scaled["full_conditional_log_xi"],
                "list_idx_sampling": self.list_vars_idx_sampling[
                    "full_conditional_log_xi"
                ],
                "list_names_plots": self.list_vars_names_plots[
                    "full_conditional_log_xi"
                ],
                "list_fixed_values": self.list_vars_fixed_values[
                    "full_conditional_log_xi"
                ],
                "sub_var": "full_conditional_log_xi",
            },
        }

        self.model_name = "hierarchical_log_xi_theta"
        dict_models = {self.model_name: target_distributions}

        return {
            "dict_models": dict_models,
            "scalers": scalers,
            "params_plot_setup": params_plot_setup,
            "params_plots_estimator": params_plots_estimator,
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
        self.sigma_m = sigma_m
        omega = omega_float * xp.ones((self.N, self.L))

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

        # run setup
        (
            dict_models,
            scalers,
            params_plot_setup,
            params_plots_estimator,
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
            mult_f_Theta,  # to initialize the auxiliary variable.
            Theta_true_scaled,
            censored,
            forward_map,
            y,
        )

    def main(self, params: dict, path_data_cloud: str) -> None:
        self.path_data_cloud = path_data_cloud
        (
            dict_models,
            scalers,
            params_plot_setup,
            params_plots_estimator,
            mult_f_Theta,  # to initialize the auxiliary variable.
            Theta_true_scaled,  # to initialize the physical parameters (theta).
            censored,
            forward_map,
            y,  # to initialize the auxiliary variable.
        ) = self.setup(
            **params["forward_model"],
            #
            sigma_a_float=params["sigma_a_float"],
            sigma_m_float=xp.log(params["sigma_m_float_linscale"]),
            omega_float=3 * params["sigma_a_float"],
            #
            params_component_distributions=params["component_distributions"],
            dict_target_distributions_match_components=params["target_distributions"],
        )

        folder_path = f"{self.path_raw}/{self.model_name}"
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        if params["debugging"]["plot_setup"]:
            self.save_and_plot_setup(
                **params_plot_setup,
            )

        self.plots_estimator = dict.fromkeys(params_plots_estimator.keys())
        for k in self.plots_estimator.keys():
            self.plots_estimator[k] = self.save_and_plot_estimator(
                **params_plots_estimator[k],
            )

        param_interest_name = params["param_interest_name"]

        # * Optim MAP
        if params["to_run_optim_map"]:
            # Set starting point
            n_mcmc_map = params["run_params"]["map"]["N_MCMC"]
            seed_mcmc_map = params["run_params"]["map"].get("rng_seed", 0)
            if seed_mcmc_map is None:
                rng_list = [xp.random.default_rng(seed=None) for _ in range(n_mcmc_map)]
            else:
                rng_list = [
                    xp.random.default_rng(seed=seed_mcmc_map + i)
                    for i in range(n_mcmc_map)
                ]
            params["run_params"]["map"]["start_from"] = self.setup_vars_0(
                start_type_Theta=params["sampling_params"]["map"]["start_from_type"][
                    "full_conditional_theta"
                ],
                start_type_LogXi=params["sampling_params"]["map"]["start_from_type"][
                    "full_conditional_log_xi"
                ],
                y=y,
                mult_f_Theta=mult_f_Theta,
                Theta_true_scaled=Theta_true_scaled,
                forward_map=forward_map,
                scalers=scalers,
                sigma_m=self.sigma_m,
                sigma_a=params["sigma_a_float"],
                censored=censored,
                rng_list=rng_list,
                N_MCMC=n_mcmc_map,
            )

            my_sampler_params = MyGibbsSamplerParams(
                target_distributions=dict_models.get(self.model_name),
                is_stochastic=False,
                param_interest_name=param_interest_name,
                **params["sampling_params"]["map"],
            )

            self.inversion_optim(
                dict_models=dict_models,
                scalers=scalers,
                my_sampler_params=my_sampler_params,
                can_run_in_parallel=params["forward_model"]["force_use_cpu"],
                param_interest_name=param_interest_name,
                debugging=params.get("debugging", dict()),
                **params["run_params"]["map"],
                rng_list=rng_list,
            )

        param_interest_name = params["param_interest_name"]

        # * MCMC
        if params["to_run_mcmc"]:
            # Set starting point
            n_mcmc = params["run_params"]["mcmc"]["N_MCMC"]
            seed_mcmc = params["run_params"]["mcmc"].get("rng_seed", 0)
            if seed_mcmc is None:
                rng_list = [xp.random.default_rng(seed=None) for _ in range(n_mcmc)]
            else:
                rng_list = [
                    xp.random.default_rng(seed=seed_mcmc + i) for i in range(n_mcmc)
                ]
            params["run_params"]["mcmc"]["start_from"] = self.setup_vars_0(
                start_type_Theta=params["sampling_params"]["mcmc"]["start_from_type"][
                    "full_conditional_theta"
                ],
                start_type_LogXi=params["sampling_params"]["mcmc"]["start_from_type"][
                    "full_conditional_log_xi"
                ],
                y=y,
                N_MCMC=n_mcmc,
                mult_f_Theta=mult_f_Theta,
                Theta_true_scaled=Theta_true_scaled,
                forward_map=forward_map,
                scalers=scalers,
                sigma_m=self.sigma_m,
                sigma_a=params["sigma_a_float"],
                censored=censored,
                rng_list=rng_list,
                indicator_prior=dict_models[self.model_name][
                    "full_conditional_theta"
                ].distribution_components["theta_indicator_prior"],
            )
            my_sampler_params = MyGibbsSamplerParams(
                target_distributions=dict_models.get(self.model_name),
                is_stochastic=True,
                param_interest_name=param_interest_name,
                **params["sampling_params"]["mcmc"],
            )

            self.inversion_mcmc(
                dict_models=dict_models,
                scalers=scalers,
                my_sampler_params=my_sampler_params,
                can_run_in_parallel=params["forward_model"]["force_use_cpu"],
                param_interest_name=param_interest_name,
                debugging=params.get("debugging", dict()),
                **params["run_params"]["mcmc"],
                rng_list=rng_list,
            )
        return


if __name__ == "__main__":
    yaml_file, path_data, path_models, path_outputs = (
        SimulationToyCaseNNHierachical.parse_args()
    )

    params = SimulationToyCaseNNHierachical.load_params(path_data, yaml_file)
    max_workers = params.get("simu_init", {}).get("max_workers", -1)
    if max_workers > 0:
        print(f"[Init] Dynamically setting thread limits to {max_workers}")

        # 1. Limit Numba threads dynamically
        try:
            nb.set_num_threads(max_workers)
        except ImportError:
            pass

        # 2. Limit NumPy/SciPy (OpenBLAS/MKL/OMP) threads dynamically
        try:
            pass
            # from threadpoolctl import threadpool_limits

            # threadpool_limits(limits=max_workers)
        except ImportError:
            print(
                "[Warning] 'threadpoolctl' is not installed. NumPy thread limits might not apply."
            )
            print("Run 'pip install threadpoolctl' to fix this.")
    else:
        print(f"[Init] max_workers is {max_workers}, using default thread limits")

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
    for key, val in params["sampling_params"]["mcmc"]["sample_variables"].items():
        if not val:
            params["run_params"]["mcmc"]["plot_1D_chains"][key] = False
            params["run_params"]["mcmc"]["plot_2D_chains"][key] = False
            params["run_params"]["mcmc"]["plot_ESS"][key] = False

    simulation.main(
        params=params,
        path_data_cloud=path_data,
    )
