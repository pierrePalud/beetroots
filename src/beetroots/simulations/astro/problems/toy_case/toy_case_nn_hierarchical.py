import os
import sys

# import yaml
# # We parse the YAML early to get max_workers
# # Extract YAML filename from command line args
# if len(sys.argv) > 1 and os.path.exists(os.path.join(sys.argv[2], sys.argv[1])):
#     yaml_file = os.path.join(sys.argv[2], sys.argv[1])
#     try:
#         with open(yaml_file, "r", encoding="utf-8") as f:
#             yaml_config = yaml.safe_load(f)
#             max_workers = yaml_config.get("simu_init", {}).get("max_workers", -1)
#             if max_workers > 0:
#                 os.environ["NUMBA_NUM_THREADS"] = str(max_workers)
#                 os.environ["OMP_NUM_THREADS"] = str(max_workers)
#                 os.environ["OPENBLAS_NUM_THREADS"] = str(max_workers)
#                 os.environ["MKL_NUM_THREADS"] = str(max_workers)
#                 print(f"[Early init] Setting thread limits to {max_workers} from YAML")
#             else:
#                 print(
#                     f"[Early init] max_workers in YAML is {max_workers}, not setting thread limits"
#                 )
#     except Exception as e:
#         print(f"[Warning] Could not extract max_workers from YAML: {e}, using defaults")
# else:
#     print(
#         "[Warning] YAML file not found in command line args, using default thread limits"
#     )
import warnings
from typing import Dict, List, Optional, Tuple

import numba as nb

from beetroots.modelling.component_distribution import ComponentDistribution
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

import importlib
from pathlib import Path

from beetroots.modelling.forward_maps.abstract_base import ForwardMap
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.modelling.target_distribution.full_conditional import FullConditional
from beetroots.sampler.kernels.gamma_approx_aux import (
    estimate_mode_from_local_grid,
    gamma_approximation_parameters,
    newton_raphson_method,
)
from beetroots.sampler.utils.sampler_params import MyGibbsSamplerParams
from beetroots.simulations.astro import data_validation
from beetroots.simulations.astro.forward_map_setup.abstract_nn import SimulationNN
from beetroots.simulations.astro.observation_setup.abstract_toy_case import (
    SimulationToyCase,
)
from beetroots.simulations.astro.sampler_setup.abstract_sampler_hierarchical import (
    SimulationMyGibbsSampler,
)
from beetroots.space_transform.id_transform import IdScaler


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
        "checkpoint_data",
    )

    def setup_vars_0(
        self,
        start_type_Theta: str,
        start_type_Aux: str,
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
        aux_shape = (self.N, self.L)

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
                xp.concatenate(
                    [theta_loaded, xp.zeros((theta_loaded.shape[0], 1))], axis=1
                )
            )[:, :-1]
            theta_values = _broadcast_or_validate(
                theta_loaded_scaled,
                theta_shape,
                "full_conditional_theta",
            )
            print(
                f"Initializing Theta from file {start_type_Theta} (same initialization for all chains if the ixp.t array fits the variable's shape, different initializations if the file contains an additional dimension for each chain)."
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

        # Initialize Aux for all chains
        if start_type_Aux == "true":
            aux_init = xp.asarray(mult_f_Theta * 1)
            aux_values = xp.repeat(aux_init[None, :, :], N_MCMC, axis=0)
            print(
                f"Initializing auxiliary variable for all N_MCMC={N_MCMC} chains with the true values."
            )
        elif Path(start_type_Aux).exists():
            if Path(start_type_Aux).suffix.lower() != ".csv":
                raise ValueError(
                    f"Aux start file must be a .csv file, got: {start_type_Aux}"
                )
            aux_loaded = self._load_csv_array(start_type_Aux, key="Aux")
            aux_values = _broadcast_or_validate(
                aux_loaded,
                aux_shape,
                "full_conditional_auxiliary",
            )
            print(
                f"Initializing auxiliary variable from file {start_type_Aux} (same initialization for all chains if the ixp.t array fits the variable's shape, different initializations if the file contains an additional dimension for each chain)."
            )
        elif start_type_Aux == "random":
            if start_type_Theta != "random":
                warnings.warn(
                    "'start_type_Aux' is 'random' while 'start_type_Theta' is not. Random multiplicative-noise initialization is applied on the provided Theta initialization following the observation model."
                )

            # Old initialization (kept as comment as requested):
            # aux_chains = []
            # for n in range(N_MCMC):
            #     theta_n = theta_values[n]
            #     mult_f_theta_start = rng_list[n].lognormal(
            #         mean=-(sigma_m**2) / 2,
            #         sigma=sigma_m,
            #     ) * forward_map.evaluate(theta_n)
            #     aux_n = scalers["full_conditional_auxiliary"].from_lin_to_scaled(
            #         mult_f_theta_start
            #     )
            #     aux_chains.append(aux_n)
            # aux_values = xp.asarray(aux_chains)

            # New initialization: use gamma-approx proposal (based on formulas in gamma_approx_aux).
            # For censored pixels use multiplicative lognormal * f_theta (as before),
            # for uncensored pixels draw from a Gamma with parameters computed from
            # a local mode estimate and second-derivative (Newton refinement).

            aux_chains = []
            for n in range(N_MCMC):
                theta_n = theta_values[n]
                f_theta = forward_map.evaluate(theta_n)
                f_theta = xp.maximum(f_theta, 1e-300)
                log_f_theta = xp.log(f_theta)
                sigma_a2 = sigma_a**2
                sigma_m2 = sigma_m**2

                u0_approx = estimate_mode_from_local_grid(
                    f_theta=f_theta[~censored],
                    log_ftheta=log_f_theta[~censored],
                    sigma_m=sigma_m[~censored],
                    sigma_a=sigma_a[~censored],
                    sigma_m2=sigma_m2[~censored],
                    y=y[~censored],
                    K=10,
                )

                mode_gamma_approx = newton_raphson_method(
                    initial_guess=u0_approx,
                    log_ftheta=log_f_theta[~censored],
                    sigma_m2=sigma_m2[~censored],
                    sigma_a2=sigma_a2[~censored],
                    y=y[~censored],
                    max_iter=5,
                )
                alpha_candidate_given_current, theta_candidate_given_current = (
                    gamma_approximation_parameters(
                        mode_gamma_approx,
                        log_f_theta[~censored],
                        sigma_m2[~censored],
                        sigma_a2[~censored],
                    )
                )

                # samples = xp.where(censored, mult_samples, gamma_samples)
                samples = xp.zeros_like(f_theta)
                samples[censored] = rng_list[n].lognormal(
                    mean=log_f_theta[censored] - (sigma_m2[censored] / 2),
                    sigma=sigma_m[censored],
                )
                samples[~censored] = rng_list[n].gamma(
                    shape=alpha_candidate_given_current,
                    scale=theta_candidate_given_current,
                )
                aux_n = scalers["full_conditional_auxiliary"].from_lin_to_scaled(
                    samples
                )
                aux_chains.append(aux_n)
            aux_values = xp.asarray(aux_chains)
            print(
                f"Initializing auxiliary variable for all N_MCMC={N_MCMC} chains using gamma-approx proposal (censored: multiplicative lognormal, uncensored: gamma-approx)."
            )
        elif start_type_Aux == "observations":
            aux_values = xp.repeat(y[None, :], N_MCMC, axis=0)
            print(
                f"Initializing auxiliary variable for all N_MCMC={N_MCMC} chains with the observed values (same initialization for all chains). Can leads to mixing issues due to coupling non matched with the Theta initialization."
            )
        else:
            warnings.warn(
                "Invalid start_type_Aux or not provided. Falling back to random initialization for all chains, i.e. adding multiplicative perturbation to the forward model evaluation of Theta."
            )
            aux_chains = []
            for n in range(N_MCMC):
                theta_n = theta_values[n]
                mult_f_theta_start = rng_list[n].lognormal(
                    mean=-(sigma_m**2) / 2,
                    sigma=sigma_m,
                ) * forward_map.evaluate(theta_n)
                aux_n = scalers["full_conditional_auxiliary"].from_lin_to_scaled(
                    mult_f_theta_start
                )
                aux_chains.append(aux_n)
            aux_values = xp.asarray(aux_chains)

        start_from = {
            seed: {
                "full_conditional_theta": theta_values[seed],
                "full_conditional_auxiliary": aux_values[seed],
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
            list_edges = component_distributions["theta_spatial_prior"].list_edges
            weights = component_distributions["theta_spatial_prior"].weights
        else:
            dict_sites, list_edges, weights = None, None, None
            warnings.warn(
                "no spatial prior for theta will be used as no component distribution with the key 'theta_spatial_prior' has been provided."
            )

        if "aux_given_theta" in component_distributions_names:
            dic = params_component_distributions["aux_given_theta"]
            module_ = importlib.import_module(dic["module"])
            class_ = getattr(module_, dic["class_name"])
            kwargs = dic["params"]
            kwargs["forward_map"] = forward_map
            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N})
            kwargs["sigma_m"] = sigma_m

            component_distributions["aux_given_theta"] = class_(**kwargs)
        else:
            raise ValueError("'aux_given_theta' must be a component distribution")

        if "obs_given_aux" in component_distributions_names:
            dic = params_component_distributions["obs_given_aux"]
            module_ = importlib.import_module(dic["module"])
            class_ = getattr(module_, dic["class_name"])
            kwargs = dic["params"]
            kwargs["y"] = y
            kwargs["sigma_a"] = sigma_a
            kwargs["omega"] = omega
            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N})
            component_distributions["obs_given_aux"] = class_(**kwargs)
        else:
            raise ValueError("'obs_given_aux' must be a component distribution")

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
            elif "aux" in target_dist_name:
                var_shape = (self.N, self.L)
            else:
                raise ValueError(
                    f"target_dist_name must contain 'theta' or 'aux' but got {target_dist_name}"
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
            "full_conditional_auxiliary": IdScaler(),
        }

        ## -- SET VALUES FOR RESULTS EXTRACTION -- ##
        # Set true values
        self.full_cond_aux_true = mult_f_Theta
        self.Vars_true_scaled = {key: None for key in target_distributions.keys()}
        self.Vars_true_scaled["full_conditional_theta"] = self.Theta_true_scaled
        self.Vars_true_scaled["full_conditional_auxiliary"] = self.full_cond_aux_true
        # list_var_names_plot
        self.list_vars_names_plots = {key: None for key in target_distributions.keys()}
        self.list_vars_names_plots["full_conditional_theta"] = self.list_names_plots
        self.list_vars_names_plots["full_conditional_auxiliary"] = self.list_lines_fit
        # Set list_vars_idx_sampling
        self.list_vars_idx_sampling = {key: None for key in target_distributions.keys()}
        self.list_vars_idx_sampling["full_conditional_theta"] = (
            self.list_idx_sampling_param_interest
        )
        self.list_vars_idx_sampling["full_conditional_auxiliary"] = list(range(self.L))
        # Set list_fixed_values
        self.list_vars_fixed_values = {key: None for key in target_distributions.keys()}
        self.list_vars_fixed_values["full_conditional_theta"] = self.list_fixed_values
        self.list_vars_fixed_values["full_conditional_auxiliary"] = [
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
            "full_conditional_auxiliary": {
                "lower_bounds_lin": y.min(axis=0),
                "upper_bounds_lin": y.max(axis=0),
                "scaler": scalers["full_conditional_auxiliary"],
                "Var_true_scaled": self.Vars_true_scaled["full_conditional_auxiliary"],
                "list_idx_sampling": self.list_vars_idx_sampling[
                    "full_conditional_auxiliary"
                ],
                "list_names_plots": self.list_vars_names_plots[
                    "full_conditional_auxiliary"
                ],
                "list_fixed_values": self.list_vars_fixed_values[
                    "full_conditional_auxiliary"
                ],
                "sub_var": "full_conditional_auxiliary",
            },
        }

        self.model_name = "hierarchical_aux_theta"
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
        self.sigma_a = sigma_a
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
            omega_float=3 * params["sigma_a_float"]
            if params.get("omega_float", None) is None
            else params["omega_float"],
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
                start_type_Aux=params["sampling_params"]["map"]["start_from_type"][
                    "full_conditional_auxiliary"
                ],
                y=y,
                mult_f_Theta=mult_f_Theta,
                Theta_true_scaled=Theta_true_scaled,
                forward_map=forward_map,
                scalers=scalers,
                sigma_m=self.sigma_m,
                sigma_a=self.sigma_a,
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
                start_type_Aux=params["sampling_params"]["mcmc"]["start_from_type"][
                    "full_conditional_auxiliary"
                ],
                y=y,
                N_MCMC=n_mcmc,
                mult_f_Theta=mult_f_Theta,
                Theta_true_scaled=Theta_true_scaled,
                forward_map=forward_map,
                scalers=scalers,
                sigma_m=self.sigma_m,
                sigma_a=self.sigma_a,
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
    result = SimulationToyCaseNNHierachical.parse_args()

    # Handle both old-style (4 return values) and new-style (5 return values) parse_args output
    yaml_file, path_data, path_models, path_outputs, checkpoint_data = result

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

    # Store checkpoint data for use in main()
    simulation.checkpoint_data = checkpoint_data

    for key, val in params["sampling_params"]["mcmc"]["sample_variables"].items():
        if not val:
            params["run_params"]["mcmc"]["plot_1D_chains"][key] = False
            params["run_params"]["mcmc"]["plot_2D_chains"][key] = False
            params["run_params"]["mcmc"]["plot_ESS"][key] = False

    simulation.main(
        params=params,
        path_data_cloud=path_data,
    )
