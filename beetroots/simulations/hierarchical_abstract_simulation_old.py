import copy
import multiprocessing as mp
import os
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple, Union

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from tqdm.auto import tqdm

from beetroots import utils
from beetroots.modelling.likelihoods.gaussian_censored import CensoredGaussianLikelihood
from beetroots.modelling.likelihoods.log_normal import LogNormalLikelihood
from beetroots.modelling.posterior import Posterior
from beetroots.modelling.priors.l22_laplacian_prior import L22LaplacianSpatialPrior
from beetroots.modelling.priors.l22_prior import L22SpatialPrior
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.sampler.hierarchical_sampler import HierarchicalSampler
from beetroots.sampler.saver.hierarchical_saver import HierarchicalSaver
from beetroots.sampler.utils.psgldparams import PSGLDParams
from beetroots.simulations.analysis import ess, plots
from beetroots.simulations.analysis import utils as su

warnings.filterwarnings("ignore")

# TODO: to keep for now, will probably have to adapt code in multiple mini classes for results extraction.

# TODO: have dict_psterior with a single entry


class HierarchicalSimulation(ABC):
    def __init__(
        self,
        max_workers: int,
        N_MCMC: int,
        T_MC: int,
        T_OPTI: int,
        T_OPTI_MLE: int,
        T_BI: int,
        batch_size: int,
        signal_2dim: bool,
        grid_name: str = "PDR17G1E20_P_cte_grid.dat",
        with_spatial_prior: bool = True,
        spatial_prior_params: Union[None, SpatialPriorParams] = None,
    ):
        assert (not (with_spatial_prior) and spatial_prior_params is None) or (
            with_spatial_prior and spatial_prior_params is not None
        ), "incoherent input : if you don't want to use spatial regularization, set spatial_prior_params to None"

        self.path_output = ""

        self.max_workers = max_workers
        self.N_MCMC = N_MCMC
        self.T_MC = T_MC
        self.T_OPTI = T_OPTI
        self.T_OPTI_MLE = T_OPTI_MLE
        self.T_BI = T_BI
        self.batch_size = batch_size

        # wether the signal is an image or a time series
        self.signal_2dim = signal_2dim

        # tmp values, for initialization
        self.D = 0
        self.D_no_kappa = 0
        self.L = 0
        self.N = 0

        self.y = np.zeros((self.N, self.L))

        self.sigma_a = np.zeros((self.N, self.L))
        self.sigma_m = np.zeros((self.N, self.L))
        self.omega = np.zeros((self.N, self.L))

        self.indicator_margin_scale = 1.0
        self.lower_bounds = np.zeros((self.D,))
        self.upper_bounds = np.ones((self.D,))
        self.lower_bounds_lin = np.zeros((self.D,))  # (for plots only)
        self.upper_bounds_lin = np.ones((self.D,))  # (for plots only)

        self.scaler = None
        self.prior_indicator_1pix = None
        self.forward_map = None

        self.dict_posteriors = {}

        self.grid_name = grid_name

        self.with_spatial_prior = with_spatial_prior
        self.spatial_prior_params = spatial_prior_params

        # self.posterior_mixing = None  # MixingModelsLikelihood()
        # self.posterior_censor = None  # CensoredGaussianLikelihood()

        # if Theta_true remains Nonen then we don't have access to truth and don't
        # compute MSE or SNR
        # (scaled because it is the one used for MSE and SNR)
        self.Theta_true_scaled = None

        self.map_shaper = None  # used to reshape when plotting
        self.list_names = [f"x_{d}" for d in range(self.D)]
        self.list_lines = [f"y_{ell}" for ell in range(self.L)]

        self.cloud_name = ""

    def setup_output_folders(self):
        id_simulation = self.cloud_name * 1

        grid_name_short = self.grid_name.split("_")[0]
        id_simulation += f"_{grid_name_short}"

        if self.with_spatial_prior:
            if self.spatial_prior_params.use_clustering:
                id_simulation += f"_{self.spatial_prior_params.name}_{self.spatial_prior_params.n_clusters}clusters"
            else:
                id_simulation += f"_{self.spatial_prior_params.name}_noclusters"
        else:
            id_simulation += f"_no_spatial"

        (
            path_output,
            path_img,
            path_raw,
            path_data_csv_in,
            path_data_csv_out,
            path_data_csv_out_mcmc,
            path_data_csv_out_optim_map,
            path_data_csv_out_optim_mle,
        ) = utils.initialize_output_folders(id_simulation)

        self.path_img = path_img
        self.path_output = path_output
        self.path_raw = path_raw
        self.path_data_csv_in = path_data_csv_in
        self.path_data_csv_out = path_data_csv_out
        self.path_data_csv_out_mcmc = path_data_csv_out_mcmc
        self.path_data_csv_out_optim_map = path_data_csv_out_optim_map
        self.path_data_csv_out_optim_mle = path_data_csv_out_optim_mle

    def get_spatial_prior_from_name(
        self,
        syn_map: pd.DataFrame,
        initial_regu_weights: np.ndarray,
    ):
        """chooses the spatial prior to be used in function of the given parameters

        Parameters
        ----------
        syn_map : pd.DataFrame
            contains the pixels neighbouring information

        Returns
        -------
        prior_spatial :
            spatial prior, either None or instance of a daughter class of SpatialPrior
        """
        if self.N > 1 and self.with_spatial_prior:
            if self.signal_2dim and self.spatial_prior_params.name == "L2-gradient":
                return L22SpatialPrior(
                    self.spatial_prior_params,
                    self.cloud_name,
                    self.D,
                    self.N,
                    df=syn_map,
                    initial_weights=initial_regu_weights,
                )
            if self.signal_2dim and self.spatial_prior_params.name == "L2-laplacian":
                return L22LaplacianSpatialPrior(
                    self.spatial_prior_params,
                    self.cloud_name,
                    self.D,
                    self.N,
                    df=syn_map,
                    initial_weights=initial_regu_weights,
                )

            if (
                not (self.signal_2dim)
                and self.spatial_prior_params.name == "L2-gradient"
            ):
                return L22SpatialPrior(
                    self.spatial_prior_params,
                    self.cloud_name,
                    self.D,
                    self.N,
                    df=syn_map,
                )
        if not self.with_spatial_prior:
            return None

        raise ValueError(
            f"you said you wanted a spatial prior (with_spatial_prior = True), but the indicated name is invalid {self.spatial_prior_params.name}"
        )

    @abstractmethod
    def setup_forward_map(self):
        pass

    @abstractmethod
    def setup_observation(self):
        pass

    @abstractmethod
    def setup_posteriors(self):
        pass

    @abstractmethod
    def save_setup_to_csv(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    def run_mcmc_simulations(
        self,
        psgld_params,
        start_mcmc_from: Optional[str],
        # sample_regu_weights: bool = True,
        # T_BI_reguweights: Optional[int] = None,
        #
        regu_spatial_N0: Union[int, float] = np.infty,
        regu_spatial_scale: Optional[float] = 1.0,
        regu_spatial_vmin: Optional[float] = 1e-8,
        regu_spatial_vmax: Optional[float] = 1e8,
        #
        freq_save_mcmc: int = 1,
    ) -> None:
        global _run_one_simulation_mcmc_all_pixels

        assert start_mcmc_from in [None, "MLE", "MAP"]

        for k in range(len(psgld_params)):
            psgld_params[k].save_to_file(
                self.path_data_csv_in, "algo{}_params_sampling.csv".format(k)
            )

        for seed in range(self.N_MCMC):
            for model_name in list(self.dict_posteriors.keys()):
                folder_path = f"{self.path_raw}/{model_name}/mcmc_{seed}"

                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)

        def _run_one_simulation_mcmc_all_pixels(dict_input: dict) -> dict:
            model_name = dict_input["model_name"]
            seed = dict_input["seed"]

            folder_path = f"{self.path_raw}/{model_name}/mcmc_{seed}"

            saver_ = HierarchicalSaver(
                self.N,
                self.D,
                self.L,
                folder_path,
                self.scaler,
                self.batch_size,
                freq_save=freq_save_mcmc,
            )

            sampler = HierarchicalSampler(
                psgld_params,
                self.D,
                self.D_no_kappa,
                self.L,
                self.N,
                np.random.default_rng(seed),
            )

            v0 = None

            # TODO: the "MLE" amd "MAP" options possibly need to be updated
            if start_mcmc_from == "MLE":
                x0, _ = self.read_MLE_from_csv_file(model_name)
                x0 = self.scaler.from_lin_to_scaled(x0)
                # x0 += psgld_params.sigma_mtm * sampler.rng.normal(size=x0.shape)

            elif start_mcmc_from == "MAP":
                mc_path = (
                    f"{self.path_raw}/{model_name}/optim_MAP_{seed}/mc_chains.hdf5"
                )
                with h5py.File(mc_path, "r") as f:
                    Theta_MAP_lin = np.array(f["list_Theta"][-1])
                    # T_v = 100
                    v0_1pix = np.max(
                        np.array(f["list_v"]).reshape((-1, self.N, self.D)),
                        axis=(0, 1),
                    )  # [-1]).flatten()  * 1e2

                v0 = v0_1pix[None, :] * np.ones((self.N, self.D))
                v0 = v0.flatten()
                x0 = self.scaler.from_lin_to_scaled(Theta_MAP_lin)

                # x0 += psgld_params.sigma_mtm * sampler.rng.normal(size=x0.shape)

            else:
                x0 = None

            tps0 = time.time()

            sampler.sample(
                self.dict_posteriors[model_name],
                saver=saver_,
                max_iter=self.T_MC,
                # sample_regu_weights=sample_regu_weights,
                # T_BI_reguweights=T_BI_reguweights,
                x0=x0,
                v0=v0,  # ! replace by None? (to be investigated), plugging u0, v_u0?
                regu_spatial_N0=regu_spatial_N0,
                regu_spatial_scale=regu_spatial_scale,
                regu_spatial_vmin=regu_spatial_vmin,
                regu_spatial_vmax=regu_spatial_vmax,
            )

            # return input dict with duration information
            total_duration = time.time() - tps0

            dict_output = {
                "seed": seed,
                "model_name": model_name,
                "total_duration": total_duration,
            }
            return dict_output

        print("starting sampling")
        list_params = [
            {"seed": seed, "model_name": model_name}
            for seed in range(self.N_MCMC)
            for model_name in list(self.dict_posteriors.keys())
        ]
        if len(list_params) == 1:
            list_simulations_durations = [
                _run_one_simulation_mcmc_all_pixels(list_params[0])
            ]
        else:
            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=mp.get_context("fork")
            ) as p:
                list_simulations_durations = list(
                    p.map(_run_one_simulation_mcmc_all_pixels, list_params)
                )

        df_results_sampling = pd.DataFrame(list_simulations_durations)
        df_results_sampling.to_csv(f"{self.path_data_csv_out_mcmc}/durations_MCMC.csv")
        print("sampling done")
        print()

    def estimate_best_regu_weights_from_mcmc(self, model_name: str) -> np.ndarray:
        assert (
            self.N > 1 and self.with_spatial_prior
        )  # if N == 1 : then no spatial regu, and no weight

        list_mcmc_folders = [
            f"{x[0]}/mc_chains.hdf5"
            for x in os.walk(f"{self.path_raw}/{model_name}")
            if "mcmc_" in x[0]
        ]
        for i, mc_path in enumerate(list_mcmc_folders):
            if i == 0:
                with h5py.File(mc_path, "r") as f:
                    list_tau = np.array(f["list_tau"][self.T_BI :])

            else:
                with h5py.File(mc_path, "r") as f:
                    list_tau = np.concatenate(
                        [list_tau, np.array(f["list_tau"][self.T_BI :])]
                    )

        estimated_regu_weights = list_tau.mean(0)
        assert estimated_regu_weights.shape == (self.D,)
        return estimated_regu_weights

    # TODO (@Pierre): update l. 384 - 650 if necessary
    # def run_optimization_MAP(
    #     self,
    #     psgld_params,
    #     start_map_from: Optional[str],
    #     freq_save_map: int = 1,
    # ) -> None:
    #     global _run_one_simulation_optim_map_all_pixels

    #     assert start_map_from in [None, "MLE", "MAP"]

    #     psgld_params.save_to_file(self.path_data_csv_in, "algo_params_optim_MAP.csv")

    #     for seed in range(self.N_MCMC):
    #         for model_name in list(self.dict_posteriors.keys()):
    #             folder_path = f"{self.path_raw}/{model_name}/optim_MAP_{seed}"
    #             if not os.path.isdir(folder_path):
    #                 os.mkdir(folder_path)

    #     def _run_one_simulation_optim_map_all_pixels(dict_input: dict) -> dict:
    #         model_name = dict_input["model_name"]
    #         seed = dict_input["seed"]

    #         folder_path = f"{self.path_raw}/{model_name}/optim_MAP_{seed}"

    #         saver_ = hierarchical_saver.HierarchicalSaver(
    #             self.N,
    #             self.D,
    #             self.L,
    #             folder_path,
    #             self.scaler,
    #             self.batch_size,
    #             freq_save=freq_save_map,
    #         )
    #         sampler = mysampler.MySampler(
    #             psgld_params,
    #             self.D,
    #             self.D_no_kappa,
    #             self.L,
    #             self.N,
    #             np.random.default_rng(seed),
    #         )

    #         if start_map_from == "MLE":
    #             x0, _ = self.read_MLE_from_csv_file(model_name)
    #             x0 = self.scaler.from_lin_to_scaled(x0)
    #             # x0 += psgld_params.sigma_mtm * sampler.rng.normal(size=x0.shape)
    #             v0 = None

    #         #! start MAP estimation from a previous MAP estimation
    #         elif start_map_from == "MAP":
    #             mc_path = (
    #                 f"{self.path_raw}/{model_name}/optim_MAP_{seed}/mc_chains.hdf5"
    #             )
    #             with h5py.File(mc_path, "r") as f:
    #                 Theta_MAP_lin = np.array(f["list_Theta"][-1])
    #                 v0_1pix = np.max(
    #                     np.array(f["list_v"][-1]).reshape((self.N, self.D)), axis=0
    #                 )  # [-1]).flatten()  * 1e2

    #             v0 = v0_1pix[None, :] * np.ones((self.N, self.D))
    #             v0 = v0.flatten()
    #             x0 = self.scaler.from_lin_to_scaled(Theta_MAP_lin)

    #         else:
    #             x0 = None
    #             v0 = None
    #             print("starting from random initialization")

    #         tps0 = time.time()

    #         sampler.sample(
    #             self.dict_posteriors[model_name],
    #             saver=saver_,
    #             max_iter=self.T_OPTI,
    #             # sample_regu_weights=False,
    #             x0=x0,
    #             v0=v0,
    #         )

    #         # return input dict with duration information
    #         total_duration = time.time() - tps0

    #         dict_output = {
    #             "seed": seed,
    #             "model_name": model_name,
    #             "total_duration": total_duration,
    #         }
    #         return dict_output

    #     print("starting optimization MAP")
    #     list_params = [
    #         {"seed": seed, "model_name": model_name}
    #         for seed in range(self.N_MCMC)
    #         for model_name in list(self.dict_posteriors.keys())
    #     ]
    #     with ProcessPoolExecutor(
    #         max_workers=self.max_workers, mp_context=mp.get_context("fork")
    #     ) as p:
    #         list_simulations_durations = list(
    #             p.map(_run_one_simulation_optim_map_all_pixels, list_params)
    #         )

    #     df_results_sampling = pd.DataFrame(list_simulations_durations)
    #     df_results_sampling.to_csv(
    #         f"{self.path_data_csv_out_optim_map}/durations_optim_MAP.csv"
    #     )
    #     print("optimization MAP done")
    #     print()
    #     return

    # def run_optimization_MLE(self, psgld_params, freq_save_mle: int = 1) -> None:
    #     global _run_one_simulation_one_pixel

    #     psgld_params.save_to_file(self.path_data_csv_in, "algo_params_optim_MLE.csv")

    #     for n in range(self.N):
    #         for model_name in list(self.dict_posteriors.keys()):
    #             for seed in range(self.N_MCMC):
    #                 folder_path_inter = f"{self.path_raw}/{model_name}/opti_MLE_{seed}"

    #                 folder_path = f"{folder_path_inter}/pixel_{n}"
    #                 for path_ in [folder_path_inter, folder_path]:
    #                     if not os.path.isdir(path_):
    #                         os.mkdir(path_)

    #     def _run_one_simulation_one_pixel(dict_input: dict) -> dict:
    #         n = dict_input["n"]
    #         #! change seed between pixels
    #         #! interest: avoid starting at the same point for all n
    #         seed_folder = dict_input["seed"]
    #         seed_run = (dict_input["seed"] + 1) * (n + 1)
    #         model_name = dict_input["model_name"]

    #         idx_model = list(self.dict_posteriors.keys()).index(model_name)
    #         params = (self.list_mixing_model_params + self.list_gaussian_approx_params)[
    #             idx_model
    #         ]

    #         folder_path_inter = f"{self.path_raw}/{model_name}/opti_MLE_{seed_folder}"
    #         folder_path = f"{folder_path_inter}/pixel_{n}"

    #         saver_ = saver.Saver(
    #             1,
    #             self.D,
    #             self.L,
    #             folder_path,
    #             self.scaler,
    #             self.batch_size,
    #             freq_save=freq_save_mle,
    #         )
    #         sampler = mysampler.MySampler(
    #             psgld_params,
    #             self.D,
    #             self.D_no_kappa,
    #             self.L,
    #             1,
    #             np.random.default_rng(seed_run),
    #         )
    #         y_pix = self.y[n, :].reshape((1, self.L))
    #         # print(f"L={self.L}, D={self.D}, N={self.N}")
    #         # print(f"omega.shape = {self.omega.shape}")
    #         omega_pix = self.omega[n, :].reshape((1, self.L))

    #         if "mixing" in model_name:
    #             sigma_a_pix = self.sigma_a[n, :].reshape((1, self.L))
    #             sigma_m_pix = self.sigma_m[n, :].reshape((1, self.L))

    #             likelihood_1pix = MixingModelsLikelihood(
    #                 self.forward_map,
    #                 self.D,
    #                 self.L,
    #                 1,
    #                 y_pix,
    #                 sigma_a_pix,
    #                 sigma_m_pix,
    #                 omega_pix,
    #                 path_transition_params=params["path_transition_params"],
    #             )
    #         elif "gaussian_approx" in model_name:
    #             is_raw = params * 1
    #             if is_raw:
    #                 m_a_pix = 0
    #                 s_a_pix = self.sigma_a[n, :].reshape((1, self.L))
    #             else:
    #                 m_a = (np.exp(self.sigma_m ** 2 / 2) - 1) * self.y
    #                 s_a = np.sqrt(
    #                     self.y ** 2
    #                     * np.exp(self.sigma_m ** 2)
    #                     * (np.exp(self.sigma_m ** 2) - 1)
    #                     + self.sigma_a ** 2
    #                 )

    #                 m_a_pix = m_a[n, :].reshape((1, self.L))
    #                 s_a_pix = s_a[n, :].reshape((1, self.L))

    #             likelihood_1pix = CensoredGaussianLikelihood(
    #                 self.forward_map,
    #                 self.D,
    #                 self.L,
    #                 1,
    #                 y_pix,
    #                 s_a_pix,
    #                 omega_pix,
    #                 bias=m_a_pix,
    #             )
    #         else:
    #             raise ValueError(f"invalid model name : {model_name}")

    #         posterior_1pix = Posterior(
    #             self.D,
    #             self.L,
    #             1,
    #             likelihood_1pix,
    #             None,
    #             self.prior_indicator_1pix,
    #         )

    #         tps0 = time.time()
    #         sampler.sample(
    #             posterior_1pix,
    #             saver=saver_,
    #             max_iter=self.T_OPTI_MLE,
    #             # sample_regu_weights=False,
    #             disable_progress_bar=True,
    #         )

    #         # return input dict with duration information
    #         total_duration = time.time() - tps0

    #         dict_output = {
    #             "n": n,
    #             "seed": seed_folder,
    #             "seed_run": seed_run,
    #             "model_name": model_name,
    #             "total_duration": total_duration,
    #         }
    #         return dict_output

    #     print("starting optimization MLE")
    #     list_params = [
    #         {"n": n, "seed": seed, "model_name": model_name}
    #         for n in range(self.N)
    #         for seed in range(self.N_MCMC)
    #         for model_name in list(self.dict_posteriors.keys())
    #     ]

    #     n_models = len(self.list_gaussian_approx_params)
    #     n_models += len(self.list_mixing_model_params)

    #     with ProcessPoolExecutor(
    #         max_workers=self.max_workers, mp_context=mp.get_context("fork")
    #     ) as p:
    #         list_simulations_durations = list(
    #             tqdm(
    #                 p.map(
    #                     _run_one_simulation_one_pixel,
    #                     list_params,
    #                 ),
    #                 total=self.N * self.N_MCMC * n_models,
    #             )
    #         )
    #     df_results_optim_mle = pd.DataFrame(list_simulations_durations)
    #     df_results_optim_mle.to_csv(
    #         f"{self.path_data_csv_out_optim_mle}/durations_optim_MLE.csv"
    #     )
    #     print("optimization MLE done")

    def plot_estimator(
        self,
        Theta_estimated,
        estimator_name,
        folder_path: str,
        model_name=None,
        is_CI: bool = False,
        Theta_true: Union[np.ndarray, None] = None,
        list_others_to_plot: Union[list] = [],
        is_theta: bool = True,
    ):
        # folder_path = f"{self.path_img}/estimators"
        # if not os.path.isdir(folder_path):
        #     os.mkdir(folder_path)

        # if self.Theta_true_scaled is not None:
        #     Theta_true = self.scaler.from_scaled_to_lin(self.Theta_true_scaled)

        assert isinstance(self.cloud_name, str) and len(self.cloud_name) > 0
        origin = "upper" if "toy" in self.cloud_name else "lower"

        list_names = self.list_names if is_theta else self.list_lines

        for d, name in enumerate(list_names):
            x_estimator_d_plot = utils.reshape_vector_for_plots(
                Theta_estimated[:, d], self.index_arr, self.N
            )

            if is_theta:
                if is_CI:
                    if self.upper_bounds_lin[-1] - self.lower_bounds_lin[-1] > 20:
                        vmin = 1
                        vmax = None
                        # vmax = self.upper_bounds_lin[d] / self.lower_bounds_lin[d]
                    else:
                        vmin = 0
                        vmax = None
                        # vmax = self.upper_bounds_lin[d] - self.lower_bounds_lin[d]
                else:
                    vmin = self.lower_bounds_lin[d] / 1.1
                    vmax = self.upper_bounds_lin[d] * 1.1
            else:
                vmin, vmax = None, None

            if "toy" not in self.cloud_name:
                x_estimator_d_plot = x_estimator_d_plot.T

            plt.figure(figsize=(8, 6))
            plt.title(f"{estimator_name} for {name}")

            if self.signal_2dim:
                plt.imshow(
                    x_estimator_d_plot,
                    norm=colors.LogNorm(vmin, vmax),
                    origin=origin,
                    cmap="jet",
                )
                plt.colorbar()
            else:
                plt.plot(x_estimator_d_plot, label="estimation")

                if len(list_others_to_plot) > 0:
                    for dict_ in list_others_to_plot:
                        per_estimator_d_plot = utils.reshape_vector_for_plots(
                            dict_["per"][:, d], self.index_arr, self.N
                        )
                        plt.plot(
                            range(self.N),
                            per_estimator_d_plot,
                            dict_["style"],
                            label=dict_["label"],
                        )

                if (self.Theta_true_scaled is not None) and (estimator_name != "true"):
                    if Theta_true is None:
                        Theta_true = self.scaler.from_scaled_to_lin(
                            self.Theta_true_scaled
                        )
                    plt.plot(Theta_true[:, d], "r--", label="true")

                vmin = self.lower_bounds_lin[d] - 0.2
                vmax = self.upper_bounds_lin[d] + 0.2
                if is_CI:
                    vmin = 0
                    vmax = None
                    # vmax = self.upper_bounds_lin[d] - self.lower_bounds_lin[d]

                plt.ylim([vmin, vmax])
                plt.grid()
                plt.legend()

            # plt.tight_layout()
            if model_name is None:
                plt.savefig(
                    f"{folder_path}/{estimator_name}_{d}.PNG",
                    bbox_inches="tight",
                )
            else:
                plt.savefig(
                    f"{folder_path}/{estimator_name}_{model_name}_{d}.PNG",
                    bbox_inches="tight",
                )

            plt.close()

            if self.signal_2dim:
                plt.figure(figsize=(8, 6))
                plt.title(f"{estimator_name} for {name}")
                plt.imshow(x_estimator_d_plot, origin=origin, cmap="jet")
                plt.colorbar()

                if model_name is None:
                    plt.savefig(
                        f"{folder_path}/{estimator_name}_linscale_{d}.PNG",
                        bbox_inches="tight",
                    )
                else:
                    plt.savefig(
                        f"{folder_path}/{estimator_name}_{model_name}_linscale_{d}.PNG",
                        bbox_inches="tight",
                    )
                plt.close()

    def plot_map_nll_of_estimator(
        self,
        Theta_estimated_scaled: np.ndarray,
        estimator_name: str,
        model_name: str,
    ):
        """DEPRECATED - not to be used"""
        folder_path = f"{self.path_img}/nll_maps"
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        posterior = copy.copy(self.dict_posteriors[model_name])
        forward_map_evals = posterior.likelihood.evaluate_all_forward_map(
            Theta_estimated_scaled, False
        )
        nll_utils = posterior.likelihood.evaluate_all_nll_utils(
            forward_map_evals, idx=None, compute_derivatives=False
        )
        nll = posterior.likelihood.neglog_pdf(
            forward_map_evals, nll_utils, pixelwise=True
        )

        # to avoid negative values : re-add the log of the variances
        if "mixing" in model_name:
            cte = -(
                nll_utils["lambda_"] * np.log(nll_utils["s_a"])
                + (1 - nll_utils["lambda_"]) * np.log(nll_utils["s_m"])
            )  # (N, L)
            cte = np.where(self.y <= self.omega, 0.0, cte)  # (N, L)
            cte = np.sum(cte, axis=1)  # (N,)
            assert cte.shape == ((self.N,)), f"{cte.shape}"
            nll += cte

        # assert nll.min() >= 0

        nll_shaped = utils.reshape_vector_for_plots(nll, self.index_arr, self.N)

        plt.figure(figsize=(8, 6))
        plt.title(f"Map of pixelwise nll of {estimator_name}")

        if self.signal_2dim:
            origin = "upper" if "toy" in self.cloud_name else "lower"
            if "toy" not in self.cloud_name:
                nll_shaped = nll_shaped.T

            plt.imshow(nll_shaped, origin=origin, norm=colors.LogNorm(), cmap="jet")
            plt.colorbar()
        else:
            plt.plot(nll_shaped)
            plt.grid()

        # plt.tight_layout()
        plt.savefig(
            f"{folder_path}/{estimator_name}_{model_name}.PNG",
            bbox_inches="tight",
        )
        plt.close()

    def save_estimator_performance(
        self, estimator_name, model_name, mse, snr, objective
    ):
        list_results_overall = [
            {
                "estimator": estimator_name,
                "model_name": model_name,
                "MSE": mse,
                "SNR": snr,
                "objective": objective,
            }
        ]
        df_results_overall = pd.DataFrame(list_results_overall)
        path_overall_results = f"{self.path_data_csv_out}/results_overall.csv"
        df_results_overall.to_csv(
            path_overall_results,
            mode="a",
            header=not (os.path.exists(path_overall_results)),
        )

    def mcmc_results_analysis_kernels(
        self,
        model_name: str,
        list_mcmc_folders: List[str],
        freq_save_mcmc: int,
    ) -> None:
        # plots
        # - freq accepted plots per kernel
        # - log proba accept per kernel
        print("starting plot of accepted frequencies")

        # theta
        list_type = np.zeros((self.N_MCMC, self.T_MC // freq_save_mcmc))
        list_accepted = np.zeros((self.N_MCMC, self.T_MC // freq_save_mcmc))
        list_log_proba_accept = np.zeros((self.N_MCMC, self.T_MC // freq_save_mcmc))

        # u
        list_type_u = np.zeros((self.N_MCMC, self.T_MC // freq_save_mcmc))
        list_accepted_u = np.zeros((self.N_MCMC, self.T_MC // freq_save_mcmc))
        list_log_proba_accept_u = np.zeros((self.N_MCMC, self.T_MC // freq_save_mcmc))

        for seed, mc_path in enumerate(list_mcmc_folders):
            with h5py.File(mc_path, "r") as f:
                list_type[seed] = np.array(f["list_type_t"])

                # on theta
                list_accepted[seed] = np.array(f["list_accepted_t"])
                list_log_proba_accept[seed] = np.array(f["list_log_proba_accept_t"])

                # on u
                list_accepted_u[seed] = np.array(f["list_u_accepted_t"])
                list_log_proba_accept_u[seed] = np.array(f["list_u_log_proba_accept_t"])

        # mobile mean size
        k_mm_mtm = 20  # MTM
        k_mm_mala = 20  # P-MALA

        folder_path_inter = f"{self.path_img}/accepted_freq"
        folder_path_inter2 = f"{folder_path_inter}/mcmc"
        folder_path = f"{folder_path_inter2}/{model_name}"
        for path_ in [folder_path_inter, folder_path_inter2, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        # * MTM
        for seed in range(self.N_MCMC):
            idx_mtm = list_type[seed] == 0
            accepted_mtm = list_accepted[seed, idx_mtm]

            if accepted_mtm.size > k_mm_mtm:
                accepted_mtm_smooth = np.convolve(
                    accepted_mtm, np.ones(k_mm_mtm) / k_mm_mtm, mode="valid"
                )

                plt.figure(figsize=(8, 6))
                plt.title(f"MTM : {100 * np.nanmean(accepted_mtm):.2f} % accepted")
                plt.plot(accepted_mtm_smooth, label="mobile mean")
                # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
                plt.grid()
                plt.legend()
                plt.xticks(rotation=45)
                # plt.tight_layout()
                plt.savefig(
                    f"{folder_path}/freq_accept_{model_name}_seed{seed}_MTM.PNG",
                    bbox_inches="tight",
                )
                plt.close()

        # * PMALA
        for seed in range(self.N_MCMC):
            idx_pmala = list_type[seed] == 1
            accepted_pmala = list_accepted[seed, idx_pmala]

            if accepted_pmala.size > k_mm_mala:
                accepted_pmala_smooth = np.convolve(
                    accepted_pmala, np.ones(k_mm_mala) / k_mm_mala, mode="valid"
                )

                plt.figure(figsize=(8, 6))
                plt.title(f"PMALA : {100 * np.nanmean(accepted_pmala):.2f} % accepted")
                plt.plot(accepted_pmala_smooth, label="mobile mean")
                # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
                plt.grid()
                plt.legend()
                plt.xticks(rotation=45)
                # plt.tight_layout()
                plt.savefig(
                    f"{folder_path}/freq_accept_{model_name}_seed{seed}_PMALA.PNG",
                    bbox_inches="tight",
                )
                plt.close()

        # * PMALA for u
        for seed in range(self.N_MCMC):
            # idx_pmala = list_type[seed] == 1
            accepted_pmala = list_accepted_u[seed, :] * 1  # [seed, idx_pmala]

            if accepted_pmala.size > k_mm_mala:
                accepted_pmala_smooth = np.convolve(
                    accepted_pmala, np.ones(k_mm_mala) / k_mm_mala, mode="valid"
                )

                plt.figure(figsize=(8, 6))
                plt.title(f"PMALA : {100 * np.nanmean(accepted_pmala):.2f} % accepted")
                plt.plot(accepted_pmala_smooth, label="mobile mean")
                # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
                plt.grid()
                plt.legend()
                plt.xticks(rotation=45)
                # plt.tight_layout()
                plt.savefig(
                    f"{folder_path}/freq_accept_{model_name}_seed{seed}_PMALA_u.PNG",
                    bbox_inches="tight",
                )
                plt.close()

        print("plots of accepted frequencies done")

        print("starting plot of log proba accept")
        folder_path_inter = f"{self.path_img}/log_proba_accept"
        folder_path_inter2 = f"{folder_path_inter}/mcmc"
        folder_path = f"{folder_path_inter2}/{model_name}"
        for path_ in [folder_path_inter, folder_path_inter2, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        # * MTM
        for seed in range(self.N_MCMC):
            idx_mtm = list_type[seed] == 0
            list_log_proba_accept_mtm = list_log_proba_accept[seed, idx_mtm]

            if list_log_proba_accept_mtm.size > k_mm_mtm:
                list_log_proba_accept_mtm_smooth = np.convolve(
                    list_log_proba_accept_mtm,
                    np.ones(k_mm_mtm) / k_mm_mtm,
                    mode="valid",
                )

                plt.figure(figsize=(8, 6))
                plt.title(
                    f"MTM: log proba accept avg: {np.nanmean(list_log_proba_accept_mtm):.3e}, median: {np.nanmedian(list_log_proba_accept_mtm):.3e}"
                )
                plt.plot(list_log_proba_accept_mtm_smooth, label="mobile mean")
                # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
                plt.grid()
                plt.legend()
                plt.yscale("symlog")
                plt.xticks(rotation=45)
                # plt.tight_layout()
                plt.savefig(
                    f"{folder_path}/log_proba_accept_{model_name}_seed{seed}_MTM.PNG",
                    bbox_inches="tight",
                )
                plt.close()

        # * PMALA
        for seed in range(self.N_MCMC):
            idx_pmala = list_type[seed] == 1
            list_log_proba_accept_pmala = list_log_proba_accept[seed, idx_pmala]

            if list_log_proba_accept_pmala.size > k_mm_mala:
                list_log_proba_accept_pmala_smooth = np.convolve(
                    list_log_proba_accept_pmala,
                    np.ones(k_mm_mala) / k_mm_mala,
                    mode="valid",
                )

                plt.figure(figsize=(8, 6))
                plt.title(
                    f"PMALA: log proba accept avg: {np.nanmean(list_log_proba_accept_pmala):.3e}, median: {np.nanmedian(list_log_proba_accept_pmala):.3e}"
                )
                plt.plot(list_log_proba_accept_pmala_smooth, label="mobile mean")
                # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
                plt.grid()
                plt.legend()
                plt.yscale("symlog")
                plt.xticks(rotation=45)
                # plt.tight_layout()
                plt.savefig(
                    f"{folder_path}/log_proba_accept_{model_name}_seed{seed}_PMALA.PNG",
                    bbox_inches="tight",
                )
                plt.close()

        # * PMALA on u
        for seed in range(self.N_MCMC):
            # idx_pmala = list_type[seed] == 1
            list_log_proba_accept_pmala = list_log_proba_accept_u[seed, :]

            if list_log_proba_accept_pmala.size > k_mm_mala:
                list_log_proba_accept_pmala_smooth = np.convolve(
                    list_log_proba_accept_pmala,
                    np.ones(k_mm_mala) / k_mm_mala,
                    mode="valid",
                )

                plt.figure(figsize=(8, 6))
                plt.title(
                    f"PMALA: log proba accept avg: {np.nanmean(list_log_proba_accept_pmala):.3e}, median: {np.nanmedian(list_log_proba_accept_pmala):.3e}"
                )
                plt.plot(list_log_proba_accept_pmala_smooth, label="mobile mean")
                # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
                plt.grid()
                plt.legend()
                plt.yscale("symlog")
                plt.xticks(rotation=45)
                # plt.tight_layout()
                plt.savefig(
                    f"{folder_path}/log_proba_accept_{model_name}_seed{seed}_PMALA_u.PNG",
                    bbox_inches="tight",
                )
                plt.close()
        print("plots of log proba accept done")
        return

    def mcmc_results_analysis_regu_param(
        self, model_name: str, list_mcmc_folders: List[str], freq_save_mcmc: int
    ) -> None:
        # TODO: implement freq_save_mcmc > 1 behavior
        # step 2 : regularization weights : plot MC + hist, and save (per simulation and for the combination) : MMSE, IC90%, IC95% IC99%

        # import spatial regularization parameter
        list_tau = np.zeros((self.N_MCMC, self.T_MC, self.D))
        for seed, mc_path in enumerate(list_mcmc_folders):
            with h5py.File(mc_path, "r") as f:
                try:
                    list_tau[seed] = np.array(f["list_tau"])
                except:
                    pass

        if not np.allclose(list_tau, 0) and self.with_spatial_prior:
            print("starting plots of regularization weights")
            folder_path_inter = f"{self.path_img}/regularization_weights"
            folder_path = f"{folder_path_inter}/{model_name}"
            for path_ in [folder_path_inter, folder_path]:
                if not os.path.isdir(path_):
                    os.mkdir(path_)

            # individual plots
            for seed in range(self.N_MCMC):
                for d in range(self.D):
                    list_tau_sd = list_tau[seed, :, d]
                    assert list_tau_sd.shape == (self.T_MC,)

                    list_tau_sd_no_BI = list_tau_sd[self.T_BI :]
                    tau_MMSE, _ = su.estimate_MMSE(list_tau_sd_no_BI)
                    IC_2p5, IC_97p5 = su.estimate_IC(list_tau_sd_no_BI, 95)
                    assert isinstance(tau_MMSE, float), tau_MMSE
                    assert isinstance(IC_2p5, float), IC_2p5
                    assert isinstance(IC_97p5, float), IC_97p5

                    plots.plot_1D_chain(
                        list_tau_sd,
                        None,
                        d,
                        folder_path,
                        f"MC of spatial regularization weight of {self.list_names[d]}",
                        lower_bounds_lin=1e-8 * np.ones((1,)),
                        upper_bounds_lin=1e8 * np.ones((1,)),
                        N_MCMC=self.N_MCMC,
                        T_MC=self.T_MC,
                        T_BI=self.T_BI,
                        # seed,
                        # tau_MMSE,
                        # IC_2p5,
                        # IC_97p5,
                        # True,
                    )
                    plots.plot_1D_hist(
                        list_tau_sd[self.T_BI :],
                        None,
                        d,
                        folder_path,
                        title=f"posterior distribution of spatial regularization weight of {self.list_names[d]}",
                        lower_bounds_lin=1e-8 * np.ones((1,)),
                        upper_bounds_lin=1e8 * np.ones((1,)),
                        seed=seed,
                        estimator=tau_MMSE,
                        IC_low=IC_2p5,
                        IC_high=IC_97p5,
                    )

            # altogether
            list_tau_flatter = list_tau.reshape((self.N_MCMC * self.T_MC, self.D))
            list_tau_flatter_no_BI = list_tau[:, self.T_BI :].reshape(
                (self.N_MCMC * (self.T_MC - self.T_BI), self.D)
            )

            tau_MMSE, _ = su.estimate_MMSE(list_tau_flatter_no_BI)  # (D,)
            IC_2p5, IC_97p5 = su.estimate_IC(list_tau_flatter_no_BI, 95)  # (D,) each
            assert tau_MMSE.shape == (self.D,)
            assert IC_2p5.shape == (self.D,)
            assert IC_97p5.shape == (self.D,)

            plt.figure(figsize=(8, 6))
            plt.title("regularization weights sampling")
            for d, name in enumerate(self.list_names):
                plt.semilogy(list_tau_flatter[:, d], label=name)

            for seed in range(self.N_MCMC):
                if seed == 0:
                    plt.axvline(
                        seed * self.T_MC + self.T_BI, c="k", ls="--", label="T_BI"
                    )
                elif seed == 1:
                    plt.axvline(seed * self.T_MC, c="k", ls="-", label="new MC")
                    plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--")

                else:
                    plt.axvline(seed * self.T_MC, c="k", ls="-")
                    plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--")

            plt.grid()
            plt.legend()
            # plt.tight_layout()
            plt.savefig(
                f"{folder_path}/mc_regu_weights.PNG",
                bbox_inches="tight",
            )
            plt.close()

            for d in range(self.D):
                plots.plot_1D_hist(
                    list_tau_flatter_no_BI[:, d],
                    None,
                    d,
                    folder_path,
                    f"posterior distribution of spatial regularization weight of {self.list_names[d]}",
                    self.lower_bounds_lin,
                    self.upper_bounds_lin,
                    None,
                    tau_MMSE[d],
                    IC_2p5[d],
                    IC_97p5[d],
                )

            list_estimation_tau = []
            for d in range(self.D):
                dict_ = {
                    "model_name": model_name,
                    "d": d,
                    "MMSE": tau_MMSE[d],
                }
                for alpha in [90, 95, 99]:
                    IC_low, IC_high = su.estimate_IC(list_tau_flatter_no_BI, alpha)
                    dict_[f"IC_{alpha}_low"] = IC_low[d]
                    dict_[f"IC_{alpha}_high"] = IC_high[d]

                list_estimation_tau.append(dict_)

            df_estimation_tau = pd.DataFrame(list_estimation_tau)
            path_file = f"{self.path_data_csv_out_mcmc}/estimation_tau.csv"
            df_estimation_tau.to_csv(
                path_file, mode="a", header=not (os.path.exists(path_file))
            )
            print("plots of regularization weights done")
        return

    def mcmc_results_analysis_objective(
        self,
        model_name: str,
        list_mcmc_folders: List[str],
        freq_save_mcmc: int,
    ):
        # TODO: implement freq_save_mcmc > 1 behavior
        list_objective_theta = np.zeros((self.N_MCMC, self.T_MC // freq_save_mcmc))
        list_objective_u = np.zeros((self.N_MCMC, self.T_MC // freq_save_mcmc))

        for seed, mc_path in enumerate(list_mcmc_folders):
            with h5py.File(mc_path, "r") as f:
                list_objective_theta[seed] = np.array(f["list_objective"])
                list_objective_u[seed] = np.array(f["list_u_objective"])

        list_objective = list_objective_theta + list_objective_u

        list_objective_flat = list_objective.flatten()
        list_objective_no_BI_flat = list_objective[:, self.T_BI :].flatten()

        list_objective_theta_flat = list_objective_theta.flatten()
        list_objective_theta_no_BI_flat = list_objective_theta[:, self.T_BI :].flatten()

        list_objective_u_flat = list_objective_u.flatten()
        list_objective_u_no_BI_flat = list_objective_u[:, self.T_BI :].flatten()

        print("starting plot of objective function")
        folder_path_inter = f"{self.path_img}/objective"
        folder_path = f"{folder_path_inter}/{model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        # * With Burn in - total
        plt.figure(figsize=(8, 6))
        plt.title(f"Total objective evolution during sampling")
        plt.plot(list_objective_flat, label="objective")

        for seed in range(self.N_MCMC):
            if seed == 0:
                plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--", label="T_BI")
            elif seed == 1:
                plt.axvline(seed * self.T_MC, c="k", ls="-", label="new MC")
                plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--")

            else:
                plt.axvline(seed * self.T_MC, c="k", ls="-")
                plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--")

        if list_objective.max() <= 0:
            plt.yscale("linear")
        elif list_objective.min() < 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")
            # plt.yscale("linear")
        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{folder_path}/sampling_objective_total_{model_name}_with_bi.PNG",
            bbox_inches="tight",
        )
        plt.close()

        # * With Burn in - theta
        plt.figure(figsize=(8, 6))
        plt.title(f"objective evolution of theta during sampling")
        plt.plot(list_objective_theta_flat, label="objective")

        for seed in range(self.N_MCMC):
            if seed == 0:
                plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--", label="T_BI")
            elif seed == 1:
                plt.axvline(seed * self.T_MC, c="k", ls="-", label="new MC")
                plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--")

            else:
                plt.axvline(seed * self.T_MC, c="k", ls="-")
                plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--")

        if list_objective.max() <= 0:
            plt.yscale("linear")
        elif list_objective.min() < 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")
            # plt.yscale("linear")
        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{folder_path}/sampling_objective_theta_{model_name}_with_bi.PNG",
            bbox_inches="tight",
        )
        plt.close()

        # * With Burn in - theta
        plt.figure(figsize=(8, 6))
        plt.title(f"objective evolution of u during sampling")
        plt.plot(list_objective_u_flat, label="objective")

        for seed in range(self.N_MCMC):
            if seed == 0:
                plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--", label="T_BI")
            elif seed == 1:
                plt.axvline(seed * self.T_MC, c="k", ls="-", label="new MC")
                plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--")

            else:
                plt.axvline(seed * self.T_MC, c="k", ls="-")
                plt.axvline(seed * self.T_MC + self.T_BI, c="k", ls="--")

        if list_objective.max() <= 0:
            plt.yscale("linear")
        elif list_objective.min() < 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")
            # plt.yscale("linear")
        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{folder_path}/sampling_objective_u_{model_name}_with_bi.PNG",
            bbox_inches="tight",
        )
        plt.close()

        # TODO: implement (?)
        # if self.Theta_true_scaled is not None:
        #     forward_map_evals = self.dict_posteriors[model_name][
        #         1
        #     ].likelihood.evaluate_all_forward_map(self.Theta_true_scaled, True)

        #     nll_utils = self.dict_posteriors[model_name][
        #         1
        #     ].likelihood.evaluate_all_nll_utils(forward_map_evals)

        #     objective_true_theta = self.dict_posteriors[model_name][1].neglog_pdf(
        #         self.Theta_true_scaled, forward_map_evals, nll_utils
        #     )

        #     plt.axhline(objective_true_theta, c="r", ls="--", label="obj Theta_true")
        #     plt.legend()
        #     plt.savefig(
        #         f"{folder_path}/sampling_objective_{model_name}_with_bi_and_true.PNG",
        #         bbox_inches="tight",
        #     )

        # * Without Burn in - total
        plt.figure(figsize=(8, 6))
        plt.title(f"Total objective evolution during sampling (no Burn-In)")
        plt.plot(list_objective_no_BI_flat, label="objective")

        for seed in range(1, self.N_MCMC):
            if seed == 1:
                plt.axvline(
                    seed * (self.T_MC - self.T_BI), c="k", ls="-", label="new MC"
                )
            else:
                plt.axvline(seed * (self.T_MC - self.T_BI), c="k", ls="-")

        if list_objective.max() < 0:
            plt.yscale("linear")
        elif list_objective.min() < 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")

        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{folder_path}/sampling_objective_total_{model_name}_no_bi.PNG",
            bbox_inches="tight",
        )
        plt.close()

        # * Without Burn in - theta
        plt.figure(figsize=(8, 6))
        plt.title(f"Objective evolution of theta during sampling (no Burn-In)")
        plt.plot(list_objective_theta_no_BI_flat, label="objective")

        for seed in range(1, self.N_MCMC):
            if seed == 1:
                plt.axvline(
                    seed * (self.T_MC - self.T_BI), c="k", ls="-", label="new MC"
                )
            else:
                plt.axvline(seed * (self.T_MC - self.T_BI), c="k", ls="-")

        if list_objective.max() < 0:
            plt.yscale("linear")
        elif list_objective.min() < 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")

        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{folder_path}/sampling_objective_theta_{model_name}_no_bi.PNG",
            bbox_inches="tight",
        )
        plt.close()

        # * Without Burn in - u
        plt.figure(figsize=(8, 6))
        plt.title(f"Objective evolution of u during sampling (no Burn-In)")
        plt.plot(list_objective_u_no_BI_flat, label="objective")

        for seed in range(1, self.N_MCMC):
            if seed == 1:
                plt.axvline(
                    seed * (self.T_MC - self.T_BI), c="k", ls="-", label="new MC"
                )
            else:
                plt.axvline(seed * (self.T_MC - self.T_BI), c="k", ls="-")

        if list_objective.max() < 0:
            plt.yscale("linear")
        elif list_objective.min() < 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")

        plt.grid()
        plt.legend()
        # plt.tight_layout()
        plt.savefig(
            f"{folder_path}/sampling_objective_u_{model_name}_no_bi.PNG",
            bbox_inches="tight",
        )
        plt.close()

        # TODO: implement (?)
        # if self.Theta_true_scaled is not None:
        #     forward_map_evals = self.dict_posteriors[
        #         model_name
        #     ].likelihood.evaluate_all_forward_map(self.Theta_true_scaled, True)
        #     nll_utils = self.dict_posteriors[
        #         model_name
        #     ].likelihood.evaluate_all_nll_utils(forward_map_evals)
        #     objective_true = self.dict_posteriors[model_name].neglog_pdf(
        #         self.Theta_true_scaled, forward_map_evals, nll_utils
        #     )
        #     plt.axhline(objective_true, c="r", ls="--", label="obj Theta_true")
        #     plt.legend()
        #     plt.savefig(
        #         f"{folder_path}/sampling_objective_{model_name}_no_bi_with_true.PNG",
        #         bbox_inches="tight",
        #     )
        plt.close()
        print("plot of objective function done")

        ## compute index of sampling MAP
        list_objective_no_BI = list_objective[:, self.T_BI :].flatten()
        idx_MAP_sampling = int(np.argmin(list_objective_no_BI))
        objective_MAP_sampling = np.min(list_objective_no_BI)
        return idx_MAP_sampling, objective_MAP_sampling

    def mcmc_results_analysis_MAP(
        self,
        model_name: str,
        list_mcmc_folders: List[str],
        idx_MAP_sampling: int,
        objective_MAP_sampling: float,
        freq_save_mcmc: int,
    ) -> None:
        # TODO: implement freq_save_mcmc > 1 behavior
        print("begin MAP analysis and plot")
        idx_chain_MAP_sampling = idx_MAP_sampling // self.T_MC
        idx_inchain_MAP_sampling = idx_MAP_sampling % self.T_MC

        assert 0 <= idx_chain_MAP_sampling < self.N_MCMC
        assert 0 <= idx_inchain_MAP_sampling < self.T_MC

        for seed, mc_path in enumerate(list_mcmc_folders):
            if seed == idx_chain_MAP_sampling:
                with h5py.File(mc_path, "r") as f:
                    Theta_MAP_lin = np.array(f["list_Theta"][idx_inchain_MAP_sampling])
                    U_MAP_lin = np.array(f["list_U"][idx_inchain_MAP_sampling])

        assert Theta_MAP_lin.shape == (self.N, self.D)
        assert U_MAP_lin.shape == (self.N, self.L)

        folder_path_inter = f"{self.path_img}/objective"
        folder_path = f"{folder_path_inter}/{model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        Theta_MAP_scaled = self.scaler.from_lin_to_scaled(Theta_MAP_lin)

        if self.Theta_true_scaled is not None:
            mse = su.compute_MSE(Theta_MAP_scaled, self.Theta_true_scaled)
            snr = su.compute_SNR(Theta_MAP_scaled, self.Theta_true_scaled)
        else:
            mse = None
            snr = None

        estimator_name = "MAP_sampling"
        self.save_estimator_performance(
            estimator_name, model_name, mse, snr, objective_MAP_sampling
        )
        if self.N > 1 and self.index_arr.size > 1:
            folder_path = f"{self.path_img}/estimators"
            folder_path_inter = f"{folder_path}/{model_name}"
            folder_path_MAP_sampling = f"{folder_path_inter}/MAP_sampling"
            for path_ in [folder_path, folder_path_inter, folder_path_MAP_sampling]:
                if not os.path.isdir(path_):
                    os.mkdir(path_)

            self.plot_estimator(
                Theta_MAP_lin,
                estimator_name,
                folder_path_MAP_sampling,
                model_name,
            )
            # self.plot_map_nll_of_estimator(Theta_MAP_scaled, estimator_name, model_name)

            self.plot_estimator(
                U_MAP_lin, "MAP_u", folder_path_MAP_sampling, model_name, is_theta=False
            )

        # save estimator
        path_overall_results = f"{self.path_data_csv_out_mcmc}/estimation_Theta_{model_name}_MAP_sampling.csv"

        df_MAP = pd.DataFrame()
        nn, dd = np.meshgrid(np.arange(self.N), np.arange(self.D))
        df_MAP["n"] = nn.astype(int).flatten()
        df_MAP["d"] = dd.astype(int).flatten()
        df_MAP = df_MAP.sort_values(by=["n", "d"])
        df_MAP["Theta_MAP_sampling"] = Theta_MAP_lin.flatten()
        df_MAP.to_csv(path_overall_results)

        print("MAP analysis and plot done.")
        return

    def mcmc_results_analysis_MC(
        self,
        model_name: str,
        list_mcmc_folders: List[str],
        plot_ESS: bool,
        plot_1D_chains: bool,
        plot_2D_chains: bool,
        plot_comparisons_yspace: bool,
        freq_save_mcmc: int,
    ) -> None:
        # TODO: implement freq_save_mcmc > 1 behavior

        global _one_pixel_mmse_ic_extraction

        folder_path_mc = f"{self.path_img}/mc"

        folder_path_1D_no_kappa = f"{folder_path_mc}/{model_name}_theta_1D"
        folder_path_1D_no_kappa_chain = f"{folder_path_1D_no_kappa}/chains"
        folder_path_1D_no_kappa_hist = f"{folder_path_1D_no_kappa}/hist"

        folder_path_1D_u = f"{folder_path_mc}/{model_name}_u_1D"
        folder_path_1D_u_chain = f"{folder_path_1D_u}/chains"
        folder_path_1D_u_hist = f"{folder_path_1D_u}/hist"

        folder_path_2D_no_kappa = f"{folder_path_mc}/{model_name}_theta_2D"
        folder_path_2D_no_kappa_chain = f"{folder_path_2D_no_kappa}/chains"
        folder_path_2D_no_kappa_hist = f"{folder_path_2D_no_kappa}/hist"

        folder_path_2D_u = f"{folder_path_mc}/{model_name}_u_2D"
        folder_path_2D_u_chain = f"{folder_path_2D_u}/chains"
        folder_path_2D_u_hist = f"{folder_path_2D_u}/hist"

        for path_ in [
            folder_path_mc,
            #
            folder_path_1D_no_kappa,
            folder_path_1D_no_kappa_chain,
            folder_path_1D_no_kappa_hist,
            #
            folder_path_1D_u,
            folder_path_1D_u_chain,
            folder_path_1D_u_hist,
            #
            folder_path_2D_no_kappa,
            folder_path_2D_no_kappa_chain,
            folder_path_2D_no_kappa_hist,
            #
            # folder_path_2D_u,
            # folder_path_2D_u_chain,
            # folder_path_2D_u_hist,
        ]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        def _one_pixel_mmse_ic_extraction(dict_input: dict):
            """for one pixel n, performs:
            - MMSE and credibility interval extraction
            - ESS computation
            - plot 1D histograms
            - plot 2D histograms
            """
            n = dict_input["n"]
            Theta_n_true = dict_input["Theta_n_true"]
            y_n = dict_input["y_n"]

            list_Theta_n_lin = np.zeros((self.N_MCMC, self.T_MC - self.T_BI, self.D))
            list_U_n_lin = np.zeros((self.N_MCMC, self.T_MC - self.T_BI, self.L))
            for seed, mc_path in enumerate(list_mcmc_folders):
                with h5py.File(mc_path, "r") as f:
                    list_Theta_n_lin[seed] = np.array(
                        f["list_Theta"][self.T_BI :, n, :]
                    )
                    list_U_n_lin[seed] = np.array(f["list_U"][self.T_BI :, n, :])

            list_Theta_n_lin_flatter = list_Theta_n_lin.reshape(
                (self.N_MCMC * (self.T_MC - self.T_BI), self.D)
            )
            list_U_n_lin_flatter = list_U_n_lin.reshape(
                (self.N_MCMC * (self.T_MC - self.T_BI), self.L)
            )

            ##* MMSE and IC estimators
            ##* Theta
            per_0p5 = np.percentile(list_Theta_n_lin_flatter, 0.5, axis=0)  # (D,)
            per_2p5 = np.percentile(list_Theta_n_lin_flatter, 2.5, axis=0)  # (D,)
            per_5 = np.percentile(list_Theta_n_lin_flatter, 5, axis=0)  # (D,)
            per_95 = np.percentile(list_Theta_n_lin_flatter, 95, axis=0)  # (D,)
            per_97p5 = np.percentile(list_Theta_n_lin_flatter, 97.5, axis=0)  # (D,)
            per_99p5 = np.percentile(list_Theta_n_lin_flatter, 99.5, axis=0)  # (D,)

            list_Theta_n_scaled_flatter = self.scaler.from_lin_to_scaled(
                list_Theta_n_lin_flatter
            )
            Theta_n_MMSE_scaled = np.mean(list_Theta_n_scaled_flatter, axis=0)  # (D,)
            Theta_n_MMSE_lin = self.scaler.from_scaled_to_lin(
                Theta_n_MMSE_scaled.reshape((1, self.D))
            ).flatten()  # (D,)

            assert per_0p5.shape == (
                self.D,
            ), f"per_0p5 has shape {per_0p5.shape} and should have shape {(self.D,)}"
            assert Theta_n_MMSE_lin.shape == (
                self.D,
            ), f"Theta_n_MMSE has shape {Theta_n_MMSE_lin.shape} and should have shape {(self.D,)}"

            path_overall_results = (
                f"{self.path_data_csv_out_mcmc}/estimation_Theta_{model_name}.csv"
            )

            df_estim = pd.DataFrame()
            df_estim["n"] = n * np.ones((self.D,), dtype=np.int32)
            df_estim["d"] = np.arange(self.D)
            df_estim["Theta_MMSE"] = Theta_n_MMSE_lin * 1
            df_estim["per_0p5"] = per_0p5 * 1
            df_estim["per_2p5"] = per_2p5 * 1
            df_estim["per_5"] = per_5 * 1
            df_estim["per_95"] = per_95 * 1
            df_estim["per_97p5"] = per_97p5 * 1
            df_estim["per_99p5"] = per_99p5 * 1

            # in order to avoid re-writing multiple times the header because of
            # paralell writing, force a delay to favor n = 0 to be writen first
            # with header
            if n == 0:
                df_estim.to_csv(
                    path_overall_results,
                    mode="w",
                    # header=not (os.path.exists(path_overall_results)),
                )
            else:
                while not (os.path.exists(path_overall_results)):
                    time.sleep(0.5)

                df_estim.to_csv(
                    path_overall_results,
                    mode="a",
                    header=not (os.path.exists(path_overall_results)),
                )

            ## index of first element st true val btw [MC first val, elt] or
            #  [elt, MC first val]
            if Theta_n_true is not None:
                first_elt_arr = -np.ones((self.N_MCMC, self.D))
                for seed in range(self.N_MCMC):
                    for d in range(self.D):
                        if list_Theta_n_lin[seed, 0, d] < Theta_n_true[d]:
                            (idx,) = np.where(
                                list_Theta_n_lin[seed, :, d] >= Theta_n_true[d]
                            )
                        else:
                            (idx,) = np.where(
                                list_Theta_n_lin[seed, :, d] <= Theta_n_true[d]
                            )

                        if idx.size > 0:
                            first_elt_arr[seed, d] = idx[0]

                list_dict = [
                    {
                        "seed": seed,
                        "n": n,
                        "d": d,
                        "first_elt_valid_mc": int(first_elt_arr[seed, d]),
                    }
                    for seed in range(self.N_MCMC)
                    for d in range(self.D)
                ]
                df_first_elt_valid_mc = pd.DataFrame.from_records(list_dict)

                path_file = (
                    f"{self.path_data_csv_out_mcmc}/first_elt_valid_mc_{model_name}.csv"
                )
                if n == 0:
                    df_first_elt_valid_mc.to_csv(
                        path_file, mode="w", header=not (os.path.exists(path_file))
                    )
                else:
                    while not (os.path.exists(path_file)):
                        time.sleep(0.5)

                    df_first_elt_valid_mc.to_csv(
                        path_file, mode="a", header=not (os.path.exists(path_file))
                    )

            ## ESS
            if plot_ESS:
                list_Theta_n_scaled = list_Theta_n_scaled_flatter.reshape(
                    (self.N_MCMC, self.T_MC - self.T_BI, self.D)
                )
                list_dict_output = []
                for d in range(self.D):
                    ess_ = ess.compute_ess(list_Theta_n_scaled[:, :, d])
                    list_dict_output.append(
                        {
                            "n": n,
                            "d": d,
                            "seed": "overall",
                            "model_name": model_name,
                            "ess": ess_,
                        }
                    )
                df_ess_nd = pd.DataFrame.from_records(list_dict_output)

                path_file = (
                    f"{self.path_data_csv_out_mcmc}/estimation_ESS_{model_name}.csv"
                )
                if n == 0:
                    df_ess_nd.to_csv(
                        path_file,
                        mode="w",  # , header=not (os.path.exists(path_file))
                    )

                else:
                    while not (os.path.exists(path_file)):
                        time.sleep(0.5)

                    df_ess_nd.to_csv(
                        path_file, mode="a", header=not (os.path.exists(path_file))
                    )

            ## 1D histograms
            if plot_1D_chains:
                for d in range(self.D):
                    true_val = Theta_n_true[d] if Theta_n_true is not None else None

                    plots.plot_1D_chain(
                        list_Theta_n_lin_flatter[:, d],
                        n,
                        d,
                        folder_path_1D_no_kappa_chain,
                        f"MC component {self.list_names[d]} of pixel {n}",
                        self.lower_bounds_lin,
                        self.upper_bounds_lin,
                        self.N_MCMC,
                        self.T_MC,
                        self.T_BI,
                        true_val,
                    )

                    plots.plot_1D_hist(
                        list_Theta_n_lin_flatter[:, d],
                        n,
                        d,
                        folder_path_1D_no_kappa_hist,
                        title=f"hist. of {self.list_names[d]} of pixel {n}",
                        lower_bounds_lin=self.lower_bounds_lin,
                        upper_bounds_lin=self.upper_bounds_lin,
                        seed=None,
                        estimator=Theta_n_MMSE_lin[d],
                        true_val=true_val,
                    )

                for ell in range(self.L):
                    plots.plot_1D_chain(
                        list_U_n_lin_flatter[:, ell],
                        n,
                        ell,
                        folder_path_1D_u_chain,
                        f"MC component {self.list_lines[ell]} of pixel {n}",
                        None,
                        None,
                        self.N_MCMC,
                        self.T_MC,
                        self.T_BI,
                        y_n[ell],
                    )

                    plots.plot_1D_hist(
                        list_U_n_lin_flatter[:, ell],
                        n,
                        ell,
                        folder_path_1D_u_hist,
                        title=f"hist. of {self.list_lines[ell]} of pixel {n}",
                        lower_bounds_lin=None,
                        upper_bounds_lin=None,
                        seed=None,
                        estimator=None,
                        true_val=y_n[ell],
                    )

            ## 2D histograms
            if plot_2D_chains and self.D > 1:
                for d1 in range(self.D):
                    for d2 in range(d1 + 1, self.D):
                        true_val = (
                            Theta_n_true[[d1, d2]] if Theta_n_true is not None else None
                        )

                        plots.plot_2D_hist(
                            list_Theta_n_lin_flatter[:, [d1, d2]],
                            n,
                            d1,
                            d2,
                            model_name,
                            folder_path_2D_no_kappa_hist,
                            self.list_names,
                            self.lower_bounds_lin,
                            self.upper_bounds_lin,
                            Theta_MMSE=Theta_n_MMSE_lin[[d1, d2]],
                            true_val=true_val,
                        )

            return

        ##
        if self.Theta_true_scaled is not None:
            Theta_true_lin = self.scaler.from_scaled_to_lin(self.Theta_true_scaled)
        else:
            Theta_true_lin = None

        list_params = [
            {
                "n": n,
                "Theta_n_true": Theta_true_lin[n]
                if Theta_true_lin is not None
                else None,
                "y_n": self.y[n],
            }
            for n in range(self.N)
        ]
        with ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=mp.get_context("fork")
        ) as p:
            list_results = list(
                tqdm(p.map(_one_pixel_mmse_ic_extraction, list_params), total=self.N)
            )
        # for params in tqdm(list_params):
        #     _one_pixel_mmse_ic_extraction(params)

        return

    def mcmc_results_analysis_MMSE_perf(
        self, model_name: str, df_estim: pd.DataFrame
    ) -> None:
        Theta_MMSE_lin = df_estim.loc[:, "Theta_MMSE"].values.reshape((self.N, self.D))
        Theta_MMSE_scaled = self.scaler.from_lin_to_scaled(Theta_MMSE_lin)

        print("starting estimators and IC plots")
        if self.Theta_true_scaled is not None:
            mse = su.compute_MSE(Theta_MMSE_scaled, self.Theta_true_scaled)
            snr = su.compute_SNR(Theta_MMSE_scaled, self.Theta_true_scaled)
        else:
            mse = None
            snr = None

        forward_map_evals = self.dict_posteriors[
            model_name
        ].likelihood.evaluate_all_forward_map(Theta_MMSE_scaled, True)
        nll_utils = self.dict_posteriors[model_name].likelihood.evaluate_all_nll_utils(
            forward_map_evals
        )
        objective = self.dict_posteriors[model_name].neglog_pdf(
            Theta_MMSE_scaled, forward_map_evals, nll_utils
        )

        self.save_estimator_performance("MMSE", model_name, mse, snr, objective)
        return

    def mcmc_results_analysis_plot_estimators(
        self, model_name: str, df_estim: pd.DataFrame
    ) -> None:

        if self.N > 1 and self.index_arr.size > 1:
            Theta_MMSE_lin = df_estim["Theta_MMSE"].values.reshape((self.N, self.D))
            per_0p5 = df_estim["per_0p5"].values.reshape((self.N, self.D))
            per_2p5 = df_estim["per_2p5"].values.reshape((self.N, self.D))
            per_5 = df_estim["per_5"].values.reshape((self.N, self.D))
            per_95 = df_estim["per_95"].values.reshape((self.N, self.D))
            per_97p5 = df_estim["per_97p5"].values.reshape((self.N, self.D))
            per_99p5 = df_estim["per_99p5"].values.reshape((self.N, self.D))

            Theta_MMSE_scaled = self.scaler.from_lin_to_scaled(Theta_MMSE_lin)

            if self.signal_2dim:
                folder_path = f"{self.path_img}/estimators"
                folder_path_inter = f"{folder_path}/{model_name}"
                folder_path_MMSE = f"{folder_path_inter}/MMSE"
                folder_path_CI90 = f"{folder_path_inter}/CI90"
                folder_path_CI95 = f"{folder_path_inter}/CI95"
                folder_path_CI99 = f"{folder_path_inter}/CI99"
                folder_path_MAP_sampling = f"{folder_path_inter}/MAP_sampling"

                for path_ in [
                    folder_path,
                    folder_path_inter,
                    folder_path_MMSE,
                    folder_path_CI90,
                    folder_path_CI95,
                    folder_path_CI99,
                    folder_path_MAP_sampling,
                ]:
                    if not os.path.isdir(path_):
                        os.mkdir(path_)

                self.plot_estimator(
                    Theta_MMSE_lin, "MMSE", folder_path_MMSE, model_name
                )
                self.plot_map_nll_of_estimator(Theta_MMSE_scaled, "MMSE", model_name)

                self.plot_estimator(per_0p5, "percentile 0.5%", folder_path_CI99)
                self.plot_estimator(per_99p5, "percentile 99.5%", folder_path_CI99)
                self.plot_estimator(per_2p5, "percentile 2.5%", folder_path_CI95)
                self.plot_estimator(per_97p5, "percentile 97.5%", folder_path_CI95)
                self.plot_estimator(per_5, "percentile 5%", folder_path_CI90)
                self.plot_estimator(per_95, "percentile 95%", folder_path_CI90)

                self.plot_estimator(
                    per_99p5 / per_0p5,
                    "99% CI size",
                    folder_path_CI99,
                    model_name,
                    True,
                )
                self.plot_estimator(
                    per_97p5 / per_2p5,
                    "95% CI size",
                    folder_path_CI95,
                    model_name,
                    True,
                )
                self.plot_estimator(
                    per_95 / per_5,
                    "90% CI size",
                    folder_path_CI90,
                    model_name,
                    True,
                )

            else:
                folder_path = f"{self.path_img}/estimators"
                folder_path_inter = f"{folder_path}/{model_name}"
                folder_path_MMSE = f"{folder_path_inter}/MMSE"
                folder_path_MAP_sampling = f"{folder_path_inter}/MAP_sampling"

                for path_ in [
                    folder_path,
                    folder_path_inter,
                    folder_path_MMSE,
                    folder_path_MAP_sampling,
                ]:
                    if not os.path.isdir(path_):
                        os.mkdir(path_)

                list_others_to_plot = [
                    {"per": per_2p5, "style": "k-", "label": "CI 95"},
                    {"per": per_97p5, "style": "k-", "label": None},
                ]
                self.plot_estimator(
                    Theta_MMSE_lin,
                    "MMSE",
                    folder_path_MMSE,
                    model_name,
                    list_others_to_plot=list_others_to_plot,
                )
                self.plot_map_nll_of_estimator(Theta_MMSE_scaled, "MMSE", model_name)

        print("estimators and IC plots done")
        return

    def mcmc_results_analysis_ess_plots(self, model_name: str) -> None:

        assert (
            self.N > 1
        ), "this function should only be called when N > 1 to avoid 1-pixel maps"
        print("starting ESS plots")
        path_file = f"{self.path_data_csv_out_mcmc}/estimation_ESS_{model_name}.csv"
        df_ess_model = pd.read_csv(path_file, index_col=["n", "d"])
        df_ess_model = df_ess_model.sort_index().reset_index(drop=False)

        # only one ESS per component
        assert (
            len(df_ess_model) == self.N * self.D
        ), f"has length {len(df_ess_model)}, should have length {self.N * self.D}"

        folder_path_inter = f"{self.path_img}/ess"
        folder_path = f"{folder_path_inter}/{model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        for d in range(self.D):
            if self.signal_2dim:
                df_ess_overall = df_ess_model[
                    (df_ess_model["seed"] == "overall") & (df_ess_model["d"] == d)
                ]

                plt.figure(figsize=(8, 6))
                plt.title(f"ESS per pixel for {self.list_names[d]}")

                ess_arr = df_ess_overall.loc[:, "ess"].values
                ess_arr_shaped = utils.reshape_vector_for_plots(
                    ess_arr, self.index_arr, self.N
                )
                plt.imshow(
                    ess_arr_shaped,
                    norm=colors.LogNorm(vmin=1.0),
                    cmap="jet",
                )
                plt.colorbar()
                # plt.tight_layout()
                plt.savefig(
                    f"{folder_path}/ESS_{model_name}_d{d}.PNG",
                    bbox_inches="tight",
                )
                plt.close()

            else:
                plt.figure(figsize=(8, 6))
                plt.title(f"ESS per pixel for {self.list_names[d]}")
                for seed in ["overall"]:  # list(range(self.N_MCMC)) +
                    df_ess_seed = df_ess_model[
                        (df_ess_model["seed"] == seed) & (df_ess_model["d"] == d)
                    ]
                    df_ess_seed = df_ess_seed.sort_values("n")

                    plt.semilogy(df_ess_seed["n"], df_ess_seed["ess"], "+-", label=seed)

                plt.ylim(bottom=1.0)
                plt.grid()
                plt.legend()
                # plt.tight_layout()
                plt.savefig(
                    f"{folder_path}/ESS_{model_name}_d{d}.PNG",
                    bbox_inches="tight",
                )
                plt.close()

        print("ESS plots done")
        return

    def mcmc_results_analysis_valid_mc(self, model_name: str) -> None:
        print("starting plot proportion of well reconstructed pixels")

        folder_path_inter = f"{self.path_img}/well_reconstructed"
        folder_path = f"{folder_path_inter}/{model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        path_file = f"{self.path_data_csv_out_mcmc}/first_elt_valid_mc_{model_name}.csv"
        df_valid_mc = pd.read_csv(path_file)

        assert (
            len(df_valid_mc) == self.N_MCMC * self.N * self.D
        ), f"has length {len(df_valid_mc)}, should have length {self.N_MCMC * self.N * self.D}"

        incr = 100 / self.N

        for seed in range(self.N_MCMC):
            df_seed = df_valid_mc[df_valid_mc["seed"] == seed]

            list_evolution = np.zeros((self.T_MC - self.T_BI, self.D))

            for d in range(self.D):
                idx_arr_d = df_seed.loc[df_seed["d"] == d, "first_elt_valid_mc"].values
                for idx in idx_arr_d:
                    if idx > 0:
                        list_evolution[int(idx) :, d] += incr

            plt.figure(figsize=(8, 6))
            plt.title("evolution of proportion of valid MC")
            for d, name in enumerate(self.list_names):
                plt.plot(list_evolution[:, d], label=name)
            plt.plot(np.mean(list_evolution, 1), "k--", label="overall")
            plt.grid()
            plt.legend()
            plt.savefig(
                f"{folder_path}/prop_well_reconstruct_pixels_all_{model_name}_seed{seed}.PNG",
                bbox_inches="tight",
            )
            plt.close()

            # max_idx = 100_000
            # plt.figure(figsize=(8, 6))
            # plt.title("evolution of proportion of valid MC")
            # for d, name in enumerate(self.list_names):
            #     plt.plot(list_evolution[:max_idx, d], label=name)
            # plt.plot(np.mean(list_evolution, 1)[:max_idx], "k--", label="overall")
            # plt.grid()
            # plt.legend()
            # plt.savefig(
            #     f"{folder_path}/prop_well_reconstruct_pixels_first{model_name}_seed{seed}.PNG",
            #     bbox_inches="tight",
            # )
            # plt.close()
        print("plot proportion of well reconstructed pixels done")

    def mcmc_results_analysis_ydistri_comp(
        self,
        model_name: str,
        df_estim: pd.DataFrame,
        list_mcmc_folders: List[str],
    ) -> None:
        global _plot_one_distribution_comparison
        N_samples_per_chain = (self.T_MC - self.T_BI) // 20
        N_samples = N_samples_per_chain * self.N_MCMC
        rng = np.random.default_rng(42)

        print("starting plot comparison of distributions of y and f(x)")

        folder_path_inter = f"{self.path_img}/distri_comp_yspace"
        folder_path = f"{folder_path_inter}/{model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        Theta_MMSE_lin = df_estim.loc[:, "Theta_MMSE"].values.reshape((self.N, self.D))
        Theta_MMSE_scaled = self.scaler.from_lin_to_scaled(Theta_MMSE_lin)
        f_Theta_MMSE = self.forward_map.evaluate(Theta_MMSE_scaled)

        Theta_MLE_lin, _ = self.read_MLE_from_csv_file(model_name)
        Theta_MLE_scaled = self.scaler.from_lin_to_scaled(Theta_MLE_lin)
        f_Theta_MLE = self.forward_map.evaluate(Theta_MLE_scaled)

        def add_label(violin, label, list_labels):
            color = violin["bodies"][0].get_facecolor().flatten()
            list_labels.append((mpatches.Patch(color=color), label))
            return list_labels

        def _plot_one_distribution_comparison(n):
            list_theta_n_lin = np.zeros((self.N_MCMC, N_samples_per_chain, self.D))
            for seed, mc_path in enumerate(list_mcmc_folders):
                list_t = list(
                    rng.choice(
                        a=np.arange(self.T_BI, self.T_MC),
                        size=N_samples_per_chain,
                        replace=False,
                    )
                )  # list
                list_t.sort()  # list

                with h5py.File(mc_path, "r") as f:
                    list_theta_n_lin[seed] = np.array(f["list_Theta"][list_t, n, :])

            list_theta_n_lin = (
                list_theta_n_lin.transpose((2, 0, 1)).reshape((self.D, N_samples)).T
            )  # (N_samples, D)

            list_theta_n_scaled = self.scaler.from_lin_to_scaled(list_theta_n_lin)
            list_f_Theta_n_lin = self.forward_map.evaluate(
                list_theta_n_scaled
            )  # (N_samples, L)

            list_labels = []

            plt.figure(figsize=(12, 8))
            if self.N > 1:
                plt.title(
                    r"comparison of $f(x)$ and $y$ distributions for pixel " + f"{n}"
                )
            else:
                plt.title(r"comparison of $f(x)$ and $y$ distributions")

            plt.xlabel("lines")
            plt.ylabel(r"$\log y$")

            n_std = 1

            l_y_add = None
            for ell in range(self.L):
                if self.y[n, ell] > self.omega[n, ell]:
                    list_theta = np.array([ell, ell, ell]) + 1
                    list_y = [
                        self.y[n, ell] - n_std * self.sigma_a[n, ell],
                        self.y[n, ell],
                        self.y[n, ell] + n_std * self.sigma_a[n, ell],
                    ]

                    (l_y_add,) = plt.plot(list_theta, list_y, f"C2_-")

            l_y_multi = None
            for ell in range(self.L):
                if self.y[n, ell] > self.omega[n, ell]:
                    list_theta = np.array([ell, ell, ell]) + 1.15
                    list_y = [
                        np.exp(np.log(self.y[n, ell]) - (n_std * self.sigma_m[n, ell])),
                        self.y[n, ell],
                        np.exp(np.log(self.y[n, ell]) + (n_std * self.sigma_m[n, ell])),
                    ]

                    (l_y_multi,) = plt.plot(list_theta, list_y, "C1_-")

            list_labels = []
            if l_y_add is not None:
                list_labels.append((l_y_add, r"$Y \pm \sigma_a$"))
            if l_y_multi is not None:
                list_labels.append((l_y_multi, r"$Y \pm \sigma_m Y$"))

            list_labels = add_label(
                plt.violinplot(
                    list_f_Theta_n_lin,
                    positions=np.arange(1, self.L + 1) + 0.45,
                    widths=0.45,
                    showmeans=True,
                    showextrema=True,
                ),
                label=r"$f(Theta)$",
                list_labels=list_labels,
            )

            l_mmse = plt.scatter(
                np.arange(1, self.L + 1) + 0.45,
                f_Theta_MMSE[n, :],
                marker="*",
                c="r",
                s=50,
            )
            l_mle = plt.scatter(
                np.arange(1, self.L + 1) + 0.45,
                f_Theta_MLE[n, :],
                marker="*",
                c="g",
                s=50,
            )
            list_labels += [
                (l_mmse, r"$f(\hat{x}_{MMSE})$"),
                (l_mle, r"$f(\hat{x}_{MLE})$"),
            ]

            l_omega = None
            for ell in range(self.L):
                if self.y[n, ell] <= self.omega[n, ell]:
                    (l_omega,) = plt.plot(
                        [1 + ell, 1 + ell + 0.8],
                        [self.omega[n, ell], self.omega[n, ell]],
                        "k--",
                    )

            if l_omega is not None:
                list_labels.append((l_omega, r"censor limit $\omega$"))

            plt.xticks(np.arange(1, self.L + 1), self.list_lines, rotation=90)
            plt.yscale("log")
            plt.ylim([list_f_Theta_n_lin.min() / 2, None])
            plt.grid()
            plt.legend(*zip(*list_labels))
            # plt.tight_layout()
            plt.savefig(
                f"{folder_path}/distribution_comparison_pix_{n}.PNG",
                bbox_inches="tight",
            )
            plt.close()

            return

        list_params = range(self.N)
        with ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=mp.get_context("fork")
        ) as p:
            _ = list(
                tqdm(
                    p.map(_plot_one_distribution_comparison, list_params),
                    total=self.N,
                )
            )
        print("plot comparison of distributions of y and f(x) done")
        return

    def mcmc_results_analysis(
        self,
        model_name: str,
        plot_ESS: bool,
        plot_1D_chains: bool,
        plot_2D_chains: bool,
        plot_comparisons_yspace: bool,
        freq_save_mcmc: int,
    ) -> None:
        list_mcmc_folders = [
            f"{x[0]}/mc_chains.hdf5"
            for x in os.walk(f"{self.path_raw}/{model_name}")
            if "mcmc_" in x[0]
        ]
        list_mcmc_folders.sort()

        # step 1 : plot kernel analysis (freq accept, log proba accept)
        self.mcmc_results_analysis_kernels(
            model_name, list_mcmc_folders, freq_save_mcmc
        )

        # step 2 : spatial regularization parameter
        self.mcmc_results_analysis_regu_param(
            model_name, list_mcmc_folders, freq_save_mcmc
        )

        # step 3 : plot objective (and return idx of min)
        idx_MAP_sampling, objective_MAP_sampling = self.mcmc_results_analysis_objective(
            model_name, list_mcmc_folders, freq_save_mcmc
        )

        # step 4 : import MAP sampling and save
        self.mcmc_results_analysis_MAP(
            model_name,
            list_mcmc_folders,
            idx_MAP_sampling,
            objective_MAP_sampling,
            freq_save_mcmc,
        )

        # step 5 : deal with whole MC for MMSE and histograms
        self.mcmc_results_analysis_MC(
            model_name,
            list_mcmc_folders,
            plot_ESS,
            plot_1D_chains,
            plot_2D_chains,
            plot_comparisons_yspace,
            freq_save_mcmc,
        )

        # step 6 : save global MMSE performance
        # (to do now, once the MMSE if computed for all pixels)
        path_overall_results = (
            f"{self.path_data_csv_out_mcmc}/estimation_Theta_{model_name}.csv"
        )

        df_estim = pd.read_csv(path_overall_results, index_col=["n", "d"])
        df_estim = df_estim.sort_index().reset_index(drop=False)

        assert (
            len(df_estim) == self.N * self.D
        ), f"has length {len(df_estim)}, should have length {self.N * self.D}"

        # TODO: correct to obtain performance of MMSE
        # self.mcmc_results_analysis_MMSE_perf(model_name, df_estim)

        # step 7 : plot maps of estimators
        self.mcmc_results_analysis_plot_estimators(model_name, df_estim)

        # step 8 : plot maps of ESS
        if plot_ESS and (self.N > 1):
            self.mcmc_results_analysis_ess_plots(model_name)

        # step 9 : plot how many components have their true value
        # between min and max of MC
        if self.Theta_true_scaled is not None:
            self.mcmc_results_analysis_valid_mc(model_name)

        # * step 10 : plot comparison of distributions of y and f(x)
        if plot_comparisons_yspace:
            self.mcmc_results_analysis_ydistri_comp(
                model_name,
                df_estim,
                list_mcmc_folders,
            )

        print()
        return

    # TODO (@Pierre) : possibly adapt all this part to the new configuration
    # (commented for now to focus on MCMC)
    # def optimization_MAP_results_analysis_objective(
    #     self,
    #     model_name: str,
    #     list_mcmc_folders: List[str],
    #     freq_save_map: int,
    # ):
    #     list_objective = np.zeros((self.N_MCMC, self.T_OPTI // freq_save_map))
    #     for seed, mc_path in enumerate(list_mcmc_folders):
    #         with h5py.File(mc_path, "r") as f:
    #             list_objective[seed] = np.array(f["list_objective"])

    #     print("starting plot of objective function")
    #     folder_path_inter = f"{self.path_img}/objective"
    #     folder_path = f"{folder_path_inter}/{model_name}"
    #     for path_ in [folder_path_inter, folder_path]:
    #         if not os.path.isdir(path_):
    #             os.mkdir(path_)

    #     plt.figure(figsize=(8, 6))
    #     plt.title(f"Objective evolution during optimization for MAP")
    #     for seed in range(self.N_MCMC):
    #         plt.plot(list_objective[seed, :])
    #     if list_objective.max() < 0:
    #         plt.yscale("linear")
    #     elif list_objective.min() < 0:
    #         plt.yscale("symlog")
    #     else:
    #         plt.yscale("log")
    #     plt.grid()
    #     # plt.tight_layout()
    #     plt.savefig(
    #         f"{folder_path}/optim_MAP_{model_name}_objective.PNG",
    #         bbox_inches="tight",
    #     )
    #     if self.Theta_true_scaled is not None:
    #         forward_map_evals = self.dict_posteriors[
    #             model_name
    #         ].likelihood.evaluate_all_forward_map(self.Theta_true_scaled, True)
    #         nll_utils = self.dict_posteriors[
    #             model_name
    #         ].likelihood.evaluate_all_nll_utils(forward_map_evals)
    #         objective_true = self.dict_posteriors[model_name].neglog_pdf(
    #             self.Theta_true_scaled, forward_map_evals, nll_utils
    #         )
    #         print(f"true objective {model_name}: {objective_true:,.2f}")

    #         plt.axhline(objective_true, c="r", ls="--", label="obj Theta_true")
    #         plt.legend()
    #         plt.savefig(
    #             f"{folder_path}/optim_MAP_{model_name}_objective_with_true.PNG",
    #             bbox_inches="tight",
    #         )
    #     plt.close()

    #     print("plot of objective function done")

    #     ## compute index of optimization MAP
    #     list_objective_no_BI = list_objective.flatten()
    #     idx_MAP_optim = int(np.argmin(list_objective_no_BI))
    #     objective_MAP_optim = np.min(list_objective_no_BI)
    #     return idx_MAP_optim, objective_MAP_optim

    # def optimization_MAP_results_analysis_MAP(
    #     self,
    #     model_name: str,
    #     list_mcmc_folders: List[str],
    #     idx_MAP_optim: int,
    #     objective_MAP_optim: float,
    #     freq_save_map: int,
    # ) -> None:

    #     print("begin MAP analysis and plot")
    #     idx_chain_MAP_optim = idx_MAP_optim // (self.T_OPTI // freq_save_map)
    #     idx_inchain_MAP_optim = idx_MAP_optim % (self.T_OPTI // freq_save_map)

    #     assert 0 <= idx_chain_MAP_optim < self.N_MCMC
    #     assert 0 <= idx_inchain_MAP_optim < (self.T_OPTI // freq_save_map)

    #     for seed, mc_path in enumerate(list_mcmc_folders):
    #         if seed == idx_chain_MAP_optim:
    #             with h5py.File(mc_path, "r") as f:
    #                 Theta_MAP_lin = np.array(f["list_Theta"][idx_inchain_MAP_optim])

    #     assert Theta_MAP_lin.shape == (self.N, self.D)

    #     Theta_MAP_scaled = self.scaler.from_lin_to_scaled(Theta_MAP_lin)

    #     if self.Theta_true_scaled is not None:
    #         mse = su.compute_MSE(Theta_MAP_scaled, self.Theta_true_scaled)
    #         snr = su.compute_SNR(Theta_MAP_scaled, self.Theta_true_scaled)
    #     else:
    #         mse = None
    #         snr = None

    #     # save estimator
    #     self.save_estimator_performance(
    #         "MAP_optim", model_name, mse, snr, objective_MAP_optim
    #     )

    #     if self.N > 1 and self.index_arr.size > 1:
    #         folder_path_inter = f"{self.path_img}/estimators"
    #         folder_path = f"{folder_path_inter}/{model_name}"
    #         folder_path_MAP_optim = f"{folder_path}/MAP_optim"
    #         for path_ in [folder_path_inter, folder_path, folder_path_MAP_optim]:
    #             if not os.path.isdir(path_):
    #                 os.mkdir(path_)

    #         self.plot_estimator(
    #             Theta_MAP_lin, "MAP_optim", folder_path_MAP_optim, model_name
    #         )
    #         self.plot_map_nll_of_estimator(Theta_MAP_scaled, "MAP_optim", model_name)

    #     # save estimator
    #     path_overall_results = f"{self.path_data_csv_out_optim_map}/estimation_Theta_{model_name}_MAP_optim.csv"

    #     df_MAP = pd.DataFrame()
    #     nn, dd = np.meshgrid(np.arange(self.N), np.arange(self.D))
    #     df_MAP["n"] = nn.astype(int).flatten()
    #     df_MAP["d"] = dd.astype(int).flatten()
    #     df_MAP = df_MAP.sort_values(by=["n", "d"])
    #     df_MAP["Theta_MAP_optim"] = Theta_MAP_lin.flatten()
    #     df_MAP.to_csv(path_overall_results)

    #     print("MAP analysis and plot done.\n")
    #     return

    # def optimization_MAP_results_analysis_kernels(
    #     self, model_name: str, list_mcmc_folders: List[str], freq_save_map: int
    # ) -> None:
    #     # plots
    #     # - freq accepted plots per kernel
    #     # - log proba accept per kernel
    #     print("starting plot of accepted frequencies")

    #     list_type = np.zeros((self.N_MCMC, self.T_OPTI // freq_save_map))
    #     list_accepted = np.zeros((self.N_MCMC, self.T_OPTI // freq_save_map))
    #     list_log_proba_accept = np.zeros((self.N_MCMC, self.T_OPTI // freq_save_map))

    #     for seed, mc_path in enumerate(list_mcmc_folders):
    #         with h5py.File(mc_path, "r") as f:
    #             list_type[seed] = np.array(f["list_type_t"])
    #             list_accepted[seed] = np.array(f["list_accepted_t"])
    #             list_log_proba_accept[seed] = np.array(f["list_log_proba_accept_t"])

    #     # mobile mean size
    #     k_mm_mtm = 20  # MTM
    #     k_mm_mala = 20  # P-MALA

    #     folder_path_inter = f"{self.path_img}/accepted_freq"
    #     folder_path_inter2 = f"{folder_path_inter}/optim_MAP"
    #     folder_path = f"{folder_path_inter2}/{model_name}"
    #     for path_ in [folder_path_inter, folder_path_inter2, folder_path]:
    #         if not os.path.isdir(path_):
    #             os.mkdir(path_)

    #     for seed in range(self.N_MCMC):
    #         idx_mtm = list_type[seed] == 0
    #         accepted_mtm = list_accepted[seed, idx_mtm]

    #         if accepted_mtm.size > k_mm_mtm:
    #             accepted_mtm_smooth = np.convolve(
    #                 accepted_mtm, np.ones(k_mm_mtm) / k_mm_mtm, mode="valid"
    #             )

    #             plt.figure(figsize=(8, 6))
    #             plt.title(f"MTM : {100 * np.nanmean(accepted_mtm):.2f} % accepted")
    #             plt.plot(accepted_mtm_smooth, label="mobile mean")
    #             # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
    #             plt.grid()
    #             plt.legend()
    #             plt.xticks(rotation=45)
    #             # plt.tight_layout()
    #             plt.savefig(
    #                 f"{folder_path}/freq_accept_{model_name}_seed{seed}_MTM.PNG",
    #                 bbox_inches="tight",
    #             )
    #             plt.close()

    #     for seed in range(self.N_MCMC):
    #         idx_pmala = list_type[seed] == 1
    #         accepted_pmala = list_accepted[seed, idx_pmala]

    #         if accepted_pmala.size > k_mm_mala:
    #             accepted_pmala_smooth = np.convolve(
    #                 accepted_pmala, np.ones(k_mm_mala) / k_mm_mala, mode="valid"
    #             )

    #             plt.figure(figsize=(8, 6))
    #             plt.title(f"PMALA : {100 * np.nanmean(accepted_pmala):.2f} % accepted")
    #             plt.plot(accepted_pmala_smooth, label="mobile mean")
    #             # plt.axvline(self.T_BI, c="k", ls="--", label="T_BI")
    #             plt.grid()
    #             plt.legend()
    #             plt.xticks(rotation=45)
    #             # plt.tight_layout()
    #             plt.savefig(
    #                 f"{folder_path}/freq_accept_{model_name}_seed{seed}_PMALA.PNG",
    #                 bbox_inches="tight",
    #             )
    #             plt.close()
    #     print("plots of accepted frequencies done")
    #     return

    # def optimization_MAP_results_analysis(
    #     self,
    #     model_name,
    #     freq_save_map: int,
    # ) -> None:
    #     list_mcmc_folders = [
    #         f"{x[0]}/mc_chains.hdf5"
    #         for x in os.walk(f"{self.path_raw}/{model_name}")
    #         if "optim_MAP_" in x[0]
    #     ]
    #     list_mcmc_folders.sort()

    #     # step 1: accept proba and log proba accept
    #     self.optimization_MAP_results_analysis_kernels(
    #         model_name, list_mcmc_folders, freq_save_map
    #     )

    #     # step 2: objective
    #     (
    #         idx_MAP_optim,
    #         objective_MAP_optim,
    #     ) = self.optimization_MAP_results_analysis_objective(
    #         model_name, list_mcmc_folders, freq_save_map
    #     )

    #     # step 3: MAP
    #     self.optimization_MAP_results_analysis_MAP(
    #         model_name,
    #         list_mcmc_folders,
    #         idx_MAP_optim,
    #         objective_MAP_optim,
    #         freq_save_map,
    #     )
    #     return

    # def read_MLE_from_csv_file(
    #     self, model_name: str
    # ) -> Tuple[np.ndarray, pd.DataFrame]:
    #     path_file = f"{self.path_data_csv_out_optim_mle}/results_MLE.csv"
    #     assert os.path.isfile(path_file), f"The MLE has not been computed yet."

    #     df_results_mle = pd.read_csv(path_file)

    #     df_mle_best = (
    #         df_results_mle[df_results_mle["model_name"] == model_name]
    #         .groupby("n")["objective"]
    #         .min()
    #     )
    #     df_mle_best = df_mle_best.reset_index()

    #     df_mle_final = pd.merge(
    #         df_results_mle, df_mle_best, on=["n", "objective"], how="inner"
    #     )
    #     df_mle_final = df_mle_final.sort_values("n")
    #     df_mle_final = df_mle_final.drop_duplicates(["n", "objective"])

    #     Theta_MLE = np.zeros((self.N, self.D))
    #     for d in range(self.D):
    #         Theta_MLE[:, d] = df_mle_final.loc[:, f"x_MLE_{d}_lin"].values

    #     return Theta_MLE, df_mle_final

    # def optimization_MLE_results_analysis(self, freq_save_mle: int) -> None:
    #     global _extract_mle_results, _plot_objectives

    #     def _extract_mle_results(dict_input: dict) -> dict:
    #         n = dict_input["n"]
    #         seed = dict_input["seed"]
    #         model_name = dict_input["model_name"]

    #         path_results = f"{self.path_raw}/{model_name}/opti_MLE_{seed}/pixel_{n}"

    #         with h5py.File(f"{path_results}/mc_chains.hdf5", "r") as f:
    #             list_Theta_lin = np.array(f["list_Theta"])  # (T, 1, D)
    #             list_objective = np.array(f["list_objective"])  # (T,)

    #         x_MLE_lin, objective = su.estimate_MAP_or_MLE(list_Theta_lin, list_objective)

    #         if self.Theta_true_scaled is not None:
    #             x_n_scaled_true = self.Theta_true_scaled[n, :].reshape((1, self.D))
    #             mse = su.compute_MSE(x_MLE_lin, x_n_scaled_true)
    #             snr = su.compute_SNR(x_MLE_lin, x_n_scaled_true)

    #         else:
    #             mse = None
    #             snr = None

    #         dict_output = {
    #             "n": n,
    #             "seed": seed,
    #             "model_name": model_name,
    #             "MSE": mse,
    #             "SNR": snr,
    #             "objective": objective,
    #         }
    #         for d in range(self.D):
    #             dict_output[f"x_MLE_{d}_lin"] = x_MLE_lin[0, d]
    #             if self.Theta_true_scaled is not None:
    #                 Theta_true = self.scaler.from_scaled_to_lin(self.Theta_true_scaled)
    #                 dict_output[f"x_true_{d}_lin"] = Theta_true[n, d]
    #         return dict_output

    #     print("starting gathering MLE results")
    #     list_params = [
    #         {"n": n, "seed": seed, "model_name": model_name}
    #         for n in range(self.N)
    #         for seed in range(self.N_MCMC)
    #         for model_name in list(self.dict_posteriors.keys())
    #     ]
    #     with ProcessPoolExecutor(
    #         max_workers=self.max_workers, mp_context=mp.get_context("fork")
    #     ) as p:
    #         list_results = list(
    #             tqdm(p.map(_extract_mle_results, list_params), total=len(list_params))
    #         )

    #     path_file = f"{self.path_data_csv_out_optim_mle}/results_MLE.csv"
    #     df_results_mle = pd.DataFrame(list_results)
    #     df_results_mle.to_csv(path_file)
    #     print("gathering MLE results done")

    #     # plot MLE for each model
    #     for model_name in list(self.dict_posteriors.keys()):
    #         Theta_MLE, df_mle_final = self.read_MLE_from_csv_file(model_name)
    #         Theta_MLE_scaled = self.scaler.from_lin_to_scaled(Theta_MLE)
    #         if self.Theta_true_scaled is not None:
    #             mse_whole = su.compute_MSE(Theta_MLE_scaled, self.Theta_true_scaled)
    #             snr_whole = su.compute_SNR(Theta_MLE_scaled, self.Theta_true_scaled)
    #         else:
    #             mse_whole = None
    #             snr_whole = None

    #         objective_whole = df_mle_final["objective"].sum()

    #         # save estimator
    #         folder_path_inter = f"{self.path_img}/estimators"
    #         folder_path = f"{folder_path_inter}/{model_name}"
    #         folder_path_MLE = f"{folder_path}/MLE"
    #         for path_ in [folder_path_inter, folder_path, folder_path_MLE]:
    #             if not os.path.isdir(path_):
    #                 os.mkdir(path_)

    #         self.save_estimator_performance(
    #             "MLE", model_name, mse_whole, snr_whole, objective_whole
    #         )
    #         if self.N > 1 and self.index_arr.size > 1:
    #             self.plot_estimator(Theta_MLE, "MLE", folder_path_MLE, model_name)
    #             self.plot_map_nll_of_estimator(Theta_MLE_scaled, "MLE", model_name)

    #     def _plot_objectives(dict_input: dict) -> bool:
    #         n = dict_input["n"]
    #         model_name = dict_input["model_name"]

    #         folder_path_inter = f"{self.path_img}/objective"
    #         folder_path_inter2 = f"{folder_path_inter}/{model_name}"
    #         folder_path = f"{folder_path_inter2}/MLE_objectives"

    #         list_objective_all_seeds = np.zeros(
    #             (self.N_MCMC, self.T_OPTI_MLE // freq_save_mle)
    #         )

    #         # read objectives
    #         for seed in range(self.N_MCMC):
    #             path_results = f"{self.path_raw}/{model_name}/opti_MLE_{seed}/pixel_{n}"
    #             with h5py.File(f"{path_results}/mc_chains.hdf5", "r") as f:
    #                 list_objective_seed = np.array(f["list_objective"])  # (T,)
    #                 assert list_objective_seed.shape == (
    #                     self.T_OPTI_MLE // freq_save_mle,
    #                 )

    #             list_objective_all_seeds[seed] += list_objective_seed

    #         # plot objectives
    #         plt.figure(figsize=(8, 6))
    #         plt.title(f"Objective evolution during optimization for pixel {n}")
    #         for seed in range(self.N_MCMC):
    #             plt.plot(
    #                 range(self.T_OPTI_MLE // freq_save_mle),
    #                 list_objective_all_seeds[seed, :],
    #             )
    #         if list_objective_all_seeds.max() <= 0:
    #             plt.yscale("linear")
    #         elif list_objective_all_seeds.min() <= 0:
    #             plt.yscale("symlog")
    #         else:
    #             plt.yscale("log")
    #         plt.grid()
    #         # plt.tight_layout()
    #         plt.savefig(
    #             f"{folder_path}/objective_pixel_{n}.PNG",
    #             bbox_inches="tight",
    #         )
    #         plt.close()

    #         return True

    #     print("starting plotting MLE objectives")

    #     # create folders
    #     for model_name in list(self.dict_posteriors.keys()):
    #         folder_path_inter = f"{self.path_img}/objective"
    #         folder_path_inter2 = f"{folder_path_inter}/{model_name}"
    #         folder_path = f"{folder_path_inter2}/MLE_objectives"
    #         for path_ in [folder_path_inter, folder_path_inter2, folder_path]:
    #             if not os.path.isdir(path_):
    #                 os.mkdir(path_)

    #     list_params = [
    #         {"n": n, "model_name": model_name}
    #         for n in range(self.N)
    #         for model_name in list(self.dict_posteriors.keys())
    #     ]
    #     with ProcessPoolExecutor(
    #         max_workers=self.max_workers, mp_context=mp.get_context("fork")
    #     ) as p:
    #         _ = list(tqdm(p.map(_plot_objectives, list_params), total=len(list_params)))
    #     # for params in tqdm(list_params):
    #     #     _plot_objectives(params)
    #     print("plotting MLE objectives done")

    #     if (self.Theta_true_scaled is not None) and (self.N > 1):
    #         print("starting plots of R-SNR in function of SNR")
    #         list_snr_obs = (20 * np.log10(self.y / self.sigma_a)).mean(1)  # (N,)

    #         plt.figure(figsize=(8, 6))
    #         plt.title(f"MLE MSE in function of additive SNR")

    #         for model_name in list(self.dict_posteriors.keys()):
    #             Theta_MLE_model, _ = self.read_MLE_from_csv_file(model_name)
    #             Theta_MLE_model_scaled = self.scaler.from_lin_to_scaled(Theta_MLE_model)
    #             list_mse = np.array(
    #                 [
    #                     su.compute_MSE(
    #                         Theta_MLE_model_scaled[n, :], self.Theta_true_scaled[n, :]
    #                     )
    #                     for n in range(self.N)
    #                 ]
    #             )  # (N,)
    #             plt.semilogy(list_snr_obs, list_mse, "+", label=model_name)

    #         plt.xlabel(r"pixelwise average observation-to-additive noise ratio")
    #         plt.ylabel(f"pixelwise MSE")
    #         plt.grid()
    #         plt.legend()
    #         # plt.tight_layout()
    #         plt.savefig(
    #             f"{self.path_img}/pixelwise_MLE_MSE.PNG",
    #             bbox_inches="tight",
    #         )
    #         plt.close()
    #         print("plots of R-SNR in function of SNR done")

    def main(
        self,
        run_mle: bool = True,
        run_map: bool = True,
        run_mcmc: bool = True,
        psgld_params_mcmc: Optional[List[PSGLDParams]] = None,
        psgld_params_map: Optional[List[PSGLDParams]] = None,
        psgld_params_mle: Optional[List[PSGLDParams]] = None,
        plot_ESS: bool = True,
        # sample_regu_weights: bool = True,
        # T_BI_reguweights: Optional[int] = None,
        regu_spatial_N0: Union[int, float] = np.infty,
        regu_spatial_scale: Optional[float] = 1.0,
        regu_spatial_vmin: Optional[float] = 1e-8,
        regu_spatial_vmax: Optional[float] = 1e8,
        #
        plot_1D_chains: bool = True,
        plot_2D_chains: bool = True,
        plot_comparisons_yspace: bool = True,
        start_map_from: Optional[str] = None,
        start_mcmc_from: Optional[str] = None,
        freq_save_mle: int = 1,
        freq_save_map: int = 1,
        freq_save_mcmc: int = 1,
    ):
        assert (run_mcmc and (psgld_params_mcmc is not None)) or (not run_mcmc)
        assert (run_map and (psgld_params_map is not None)) or (not run_map)
        assert (run_mle and (psgld_params_mle is not None)) or (not run_mle)
        assert start_map_from in [None, "MLE", "MAP"]
        assert start_mcmc_from in [None, "MLE", "MAP"]

        print("starting simulations")
        tps_init = time.time()

        # TODO (@Pierre): to be adapted to the new setting considered
        # if run_mle:
        #     # step 0 : delete any prior analysis
        #     utils.empty_folder(self.path_data_csv_out_optim_mle)

        #     self.run_optimization_MLE(psgld_params_mle, freq_save_mle)
        #     self.optimization_MLE_results_analysis(freq_save_mle)

        # if run_map:
        #     # step 0 : delete any prior analysis
        #     utils.empty_folder(self.path_data_csv_out_optim_map)

        #     # set regularization weights
        #     # if self.N > 1 and sample_regu_weights:
        #     #     for model_name, posterior in self.dict_posteriors.items():
        #     #         weights_mixing = self.estimate_best_regu_weights_from_mcmc(
        #     #             model_name
        #     #         )
        #     #         posterior.prior_spatial.weights = weights_mixing

        #     # run MAP estimation
        #     self.run_optimization_MAP(psgld_params_map, start_map_from, freq_save_map)

        #     for model_name in list(self.dict_posteriors.keys()):
        #         self.optimization_MAP_results_analysis(model_name, freq_save_map)

        if run_mcmc:
            # step 0 : delete any prior analysis
            utils.empty_folder(self.path_data_csv_out_mcmc)

            # TODO: uncomment (commented out for debugging)
            self.run_mcmc_simulations(
                psgld_params_mcmc,
                start_mcmc_from,
                # sample_regu_weights,
                # T_BI_reguweights,
                regu_spatial_N0,
                regu_spatial_scale,
                regu_spatial_vmin,
                regu_spatial_vmax,
                #
                freq_save_mcmc,
            )

            for model_name in list(self.dict_posteriors.keys()):
                self.mcmc_results_analysis(
                    model_name,
                    plot_ESS,
                    plot_1D_chains,
                    plot_2D_chains,
                    plot_comparisons_yspace,
                    freq_save_mcmc,
                )

        total_duration = time.time() - tps_init  # is seconds
        total_duration_str = time.strftime("%H:%M:%S", time.gmtime(total_duration))
        print(
            f"Simulation and analysis finished. Total duration : {total_duration_str} s"
        )
