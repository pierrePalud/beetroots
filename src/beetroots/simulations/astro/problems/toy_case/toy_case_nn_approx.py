import os
import warnings
from typing import Dict, Optional

import numba as nb

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

import importlib
from pathlib import Path

from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.modelling.target_distribution.posterior import Posterior
from beetroots.sampler.utils.sampler_params import MyGibbsSamplerParams
from beetroots.simulations.astro import data_validation
from beetroots.simulations.astro.forward_map_setup.abstract_nn import SimulationNN
from beetroots.simulations.astro.observation_setup.abstract_toy_case import (
    SimulationToyCase,
)
from beetroots.simulations.astro.sampler_setup.abstract_sampler_hierarchical import (
    SimulationMyGibbsSampler,
)


class SimulationToyCaseNNApprox(
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
        Theta_true_scaled: xp.ndarray,
        rng_list: list[xp.random.Generator],
        N_MCMC: int,
        indicator_prior: Optional[SmoothIndicatorPrior] = None,
    ) -> dict:

        def _broadcast_or_validate(
            arr: xp.ndarray,
            target_shape_no_chain: tuple[int, int],
        ) -> xp.ndarray:
            if arr.ndim == 2 and arr.shape == target_shape_no_chain:
                return xp.repeat(arr[None, :, :], N_MCMC, axis=0)
            if arr.ndim == 3 and arr.shape == (N_MCMC,) + target_shape_no_chain:
                return arr
            raise ValueError(
                f"Invalid shape for Theta: got {arr.shape}. Expected either {target_shape_no_chain} or {(N_MCMC,) + target_shape_no_chain}."
            )

        if len(rng_list) != N_MCMC:
            raise ValueError(
                f"Expected {N_MCMC} random generators, got {len(rng_list)}."
            )

        theta_shape = (self.N, self.D_sampling)

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
            theta_values = _broadcast_or_validate(
                theta_loaded,
                theta_shape,
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

        start_from = {
            seed: {
                "Theta": theta_values[seed],
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
            raise Warning(
                "no spatial prior for theta will be used as no component distribution with the key 'theta_spatial_prior' has been provided."
            )

        dict_posteriors = {}
        if "approx-likelihood" in component_distributions_names:
            dic = params_component_distributions["approx-likelihood"]
            kwargs = dic["params"]
            list_mixing_model_params = [
                {"path_transition_params": f"{self.path_data_cloud}/{filename}"}
                for filename in kwargs["mixing_model_params_filename"]
            ]
            kwargs_mixing = dict()
            for i, dict_params in enumerate(list_mixing_model_params):
                model_name = f"mixing_{i}"
                module_ = importlib.import_module(dic["module"][0])
                class_ = getattr(module_, dic["class_name"][0])
                kwargs_mixing.update({"D": self.D_sampling, "L": self.L, "N": self.N})
                kwargs_mixing.update(dict_params)
                kwargs_mixing.update({"forward_map": forward_map})
                kwargs_mixing.update({"y": y})
                kwargs_mixing.update({"sigma_a": sigma_a})
                kwargs_mixing.update({"sigma_m": sigma_m})
                kwargs_mixing.update({"omega": omega})
                kwargs_mixing.update({"list_lines_fit": self.list_lines_fit * 1})
                kwargs_mixing.update({"var_name": "Theta"})

                component_distributions["approx-likelihood"] = class_(**kwargs_mixing)

                posterior_mixing = Posterior(
                    self.D_sampling,
                    self.L,
                    self.N,
                    likelihood=component_distributions["approx-likelihood"],
                    prior_spatial=component_distributions["theta_spatial_prior"],
                    prior_indicator=component_distributions["theta_indicator_prior"],
                    separable=True,  # only used if no spatial prior
                    var_name="Theta",
                )
                dict_posteriors[model_name] = {"Theta": posterior_mixing}

            list_gaussian_approx_params = kwargs["list_gaussian_approx_params"]
            kwargs_gaussian = dict()
            for is_raw in list_gaussian_approx_params:
                name = "raw" if is_raw else "transformed"
                model_name = f"gaussian_approx_{name}"

                if is_raw:
                    m_a = xp.zeros((self.N, self.L))
                    s_a = sigma_a * 1
                else:
                    m_a = 0  # (np.exp(sigma_m**2 / 2) - 1) * y
                    s_a = xp.sqrt(
                        y**2
                        # * np.exp(sigma_m**2)
                        * (xp.exp(sigma_m**2) - 1)
                        + sigma_a**2
                    )

                module_ = importlib.import_module(dic["module"][1])
                class_ = getattr(module_, dic["class_name"][1])
                kwargs_gaussian.update({"D": self.D_sampling, "L": self.L, "N": self.N})
                kwargs_gaussian.update({"forward_map": forward_map})
                kwargs_gaussian.update({"y": y})
                kwargs_gaussian.update({"sigma": s_a})
                kwargs_gaussian.update({"bias": m_a})
                kwargs_gaussian.update({"omega": omega})

                component_distributions["censored_likelihood"] = class_(
                    **kwargs_gaussian
                )
                # separable+True: if no spatial prior, still possible to run
                # chromatic Gibbs on MTM
                posterior_censor = Posterior(
                    self.D_sampling,
                    self.L,
                    self.N,
                    likelihood=component_distributions["censored_likelihood"],
                    prior_spatial=component_distributions["theta_spatial_prior"],
                    prior_indicator=component_distributions["theta_indicator_prior"],
                    separable=False,
                    var_name="Theta",
                )
                dict_posteriors[model_name] = {"Theta": posterior_censor}

        # Set scalers
        scalers = {"Theta": scaler}

        ## -- SET VALUES FOR RESULTS EXTRACTION -- ##
        # Set true values
        self.Vars_true_scaled = dict()
        self.Vars_true_scaled["Theta"] = self.Theta_true_scaled
        # list_var_names_plot
        self.list_vars_names_plots = dict()
        self.list_vars_names_plots["Theta"] = self.list_names_plots
        # Set list_vars_idx_sampling
        self.list_vars_idx_sampling = dict()
        self.list_vars_idx_sampling["Theta"] = self.list_idx_sampling_param_interest
        # Set list_fixed_values
        self.list_vars_fixed_values = dict()
        self.list_vars_fixed_values["Theta"] = self.list_fixed_values

        params_plot_setup = {
            "dict_sites_": dict_sites,
            "y": y * 1,
            "sigma_a": sigma_a * 1,
            "omega": omega * 1,
        }

        params_plots_estimator = {
            "Theta": {
                "lower_bounds_lin": lower_bounds_lin,
                "upper_bounds_lin": upper_bounds_lin,
                "scaler": scalers["Theta"],
                "Var_true_scaled": self.Vars_true_scaled["Theta"],
                "list_idx_sampling": self.list_vars_idx_sampling["Theta"],
                "list_names_plots": self.list_vars_names_plots["Theta"],
                "list_fixed_values": self.list_vars_fixed_values["Theta"],
                "sub_var": "Theta",
            }
        }

        return {
            "dict_models": dict_posteriors,
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
            Theta_true_scaled,
            censored,
            forward_map,
            y,
        ) = simulation.setup(
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

        for model_name in list(dict_models.keys()):
            folder_path = f"{self.path_raw}/{model_name}"
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)

        if params["debugging"]["plot_setup"]:
            self.save_and_plot_setup(
                **params_plot_setup,
            )

        self.plots_estimator = dict.fromkeys(params_plots_estimator.keys())
        for k in self.plots_estimator.keys():
            self.plots_estimator[k] = simulation.save_and_plot_estimator(
                **params_plots_estimator[k],
            )

        param_interest_name = params["param_interest_name"]

        # * Optim MAP
        if params["to_run_optim_map"]:
            n_mcmc_map = params["run_params"]["map"]["N_MCMC"]
            seed_map = params["run_params"]["map"].get("rng_seed", None)
            if seed_map is None:
                rng_list = [xp.random.default_rng(seed=None) for _ in range(n_mcmc_map)]
            else:
                rng_list = [
                    xp.random.default_rng(seed=seed_map + i) for i in range(n_mcmc_map)
                ]
            params["run_params"]["map"]["start_from"] = self.setup_vars_0(
                start_type_Theta=params["sampling_params"]["map"]["start_from_type"][
                    "Theta"
                ],
                Theta_true_scaled=Theta_true_scaled,
                rng_list=rng_list,
                N_MCMC=n_mcmc_map,
                indicator_prior=dict_models[list(dict_models.keys())[0]][
                    "Theta"
                ].prior_indicator,
            )

            my_sampler_params = MyGibbsSamplerParams(
                target_distributions=dict_models.get(list(dict_models.keys())[0]),
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
                rng_list=rng_list,
                **params["run_params"]["map"],
            )

        # * MCMC
        if params["to_run_mcmc"]:
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
                    "Theta"
                ],
                Theta_true_scaled=Theta_true_scaled,
                rng_list=rng_list,
                N_MCMC=n_mcmc,
                indicator_prior=dict_models[list(dict_models.keys())[0]][
                    "Theta"
                ].prior_indicator,
            )
            my_sampler_params = MyGibbsSamplerParams(
                target_distributions=dict_models.get(list(dict_models.keys())[0]),
                is_stochastic=True,
                param_interest_name=param_interest_name,
                **params["sampling_params"]["mcmc"],
            )

            simulation.inversion_mcmc(
                dict_models=dict_models,
                scalers=scalers,
                my_sampler_params=my_sampler_params,
                can_run_in_parallel=params["forward_model"]["force_use_cpu"],
                param_interest_name=param_interest_name,
                debugging=params.get("debugging", dict()),
                rng_list=rng_list,
                **params["run_params"]["mcmc"],
            )
        return


if __name__ == "__main__":
    yaml_file, path_data, path_models, path_outputs, checkpoint_data = (
        SimulationToyCaseNNApprox.parse_args()
    )

    params = SimulationToyCaseNNApprox.load_params(path_data, yaml_file)
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
    SimulationToyCaseNNApprox.check_input_params_file(
        params,
        data_validation.schema_astro_hierarchical,
    )

    simulation = SimulationToyCaseNNApprox(
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
