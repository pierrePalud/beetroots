# """Sequatially :
# 1. Builds the piecewise constant time series toy case
# 2. launches multiple sampling (for both mixing model and simple gaussian)
# 3. launches multiple MAP estimation (for both mixing model and simple gaussian)
# 4. launches multiple MLE estimate (for both mixing model and simple gaussian)
# """
# import multiprocessing as mp
# import os
# import pickle
# import shutil
# import time
# from concurrent.futures import ProcessPoolExecutor

# import h5py
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib import colors
# from scipy.stats import lognorm, norm
# from tqdm.auto import tqdm

# # import beetroots.utils as utils
# from beetroots.modelling.forward_maps.basic_exp import BasicExpForwardMap
# from beetroots.modelling.likelihoods.approx_censored_add_mult import (
#     MixingModelsLikelihood,
# )
# from beetroots.modelling.likelihoods.gaussian_censored import CensoredGaussianLikelihood
# from beetroots.modelling.posterior import Posterior
# from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
# from beetroots.modelling.priors.tv_1D_prior import TVeps1DSpatialPrior
# from beetroots.sampler import mysampler, saver
# from beetroots.sampler.psgldparams import PSGLDParams
# from beetroots.simulations.abstract_simulation import Simulation
# from beetroots.simulations.analysis import utils as su
# from beetroots.space_transform.id_transform import IdScaler

# SMALL_SIZE = 16
# MEDIUM_SIZE = 20
# BIGGER_SIZE = 24

# plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
# plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
# plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
# plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
# plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


# class PiecewiseTSToyCase(Simulation):
#     def __init__(
#         self,
#         max_workers: int,
#         N_MCMC: int,
#         T_MC: int,
#         T_OPTI: int,
#         T_OPTI_MLE: int,
#         T_BI: int,
#         batch_size: int,
#         signal_2dim: bool,
#         list_gaussian_approx_params: list,
#         list_mixing_model_params: list,
#         N: int,
#     ):
#         super().__init__(
#             max_workers,
#             N_MCMC,
#             T_MC,
#             T_OPTI,
#             T_OPTI_MLE,
#             T_BI,
#             batch_size,
#             signal_2dim,
#             list_gaussian_approx_params,
#             list_mixing_model_params,
#         )
#         self.N = N
#         self.cloud_name = "toy"

#     def setup_forward_map(self):
#         # define scaler and apply it
#         self.scaler = IdScaler()
#         self.forward_map = BasicExpForwardMap(self.D, self.L, self.N)

#         with open(self.path_output + "/scaler.pickle", "wb") as file_:
#             pickle.dump(self.scaler, file_)

#     def setup_observation(self, list_cols_names):
#         syn_map = pd.DataFrame()
#         syn_map["x"] = np.round(np.linspace(0, 9, self.N), 1)
#         syn_map["y"] = np.exp(syn_map["x"])
#         syn_map["idx"] = np.arange(self.N)
#         syn_map["idx1"] = np.arange(self.N)
#         syn_map["idx2"] = 0
#         syn_map = syn_map.set_index(["idx1", "idx2"])

#         self.Theta_true_scaled = syn_map.iloc[:, : self.D].values
#         assert self.Theta_true_scaled.shape == (self.N, self.D)

#         syn_map.iloc[:, self.D : -1] = self.forward_map.evaluate(self.Theta_true_scaled)

#         self.list_lines = list(syn_map.columns)[self.D : -1]

#         # generate observation
#         rv_a = norm(loc=0, scale=self.sigma_a)  # note that sigma_a has (N,L) shape
#         rv_m = lognorm(s=self.sigma_m)  # sigma_m : float

#         y0 = self.forward_map.evaluate(self.Theta_true_scaled)
#         self.y = np.maximum(
#             self.omega * np.ones_like(y0),
#             rv_m.rvs(y0.shape, random_state=42) * y0 + rv_a.rvs(random_state=42),
#         )  # (N,L)

#         # save observations
#         syn_map_tocsv = syn_map.copy()
#         syn_map_tocsv.iloc[:, : self.D] = self.scaler.from_scaled_to_lin(
#             syn_map_tocsv.iloc[:, : self.D].values,
#         )
#         syn_map_tocsv.to_csv(
#             f"{self.path_data_csv_in}/true_params_and_emissions_maps.csv"
#         )

#         self.index_arr = utils.from_df_to_full_rectangle(syn_map_tocsv)

#         #%% generation of noisy observation
#         rv_a = norm(loc=0, scale=self.sigma_a)  # note that sigma_a has (N,L) shape
#         rv_m = lognorm(s=self.sigma_m)  # sigma_m : float

#         y0 = self.forward_map.evaluate(self.Theta_true_scaled)
#         self.y = np.maximum(
#             self.omega,
#             rv_m.rvs(y0.shape, random_state=42) * y0 + rv_a.rvs(random_state=42),
#         )  # (N,L)

#         #%% save observation
#         df_observation = syn_map.copy()
#         df_observation = df_observation.drop(["x"], 1)
#         df_observation.iloc[:, :-1] = self.y

#         df_observation.to_csv(f"{self.path_data_csv_in}/observation_maps.csv")
#         return syn_map

#     def setup_posteriors(self, syn_map):
#         prior_spatial = None

#         prior_indicator = SmoothIndicatorPrior(
#             self.D,
#             self.N,
#             self.indicator_margin_scale,
#             self.lower_bounds,
#             self.upper_bounds,
#         )
#         self.prior_indicator_1pix = SmoothIndicatorPrior(
#             self.D,
#             1,
#             self.indicator_margin_scale,
#             self.lower_bounds,
#             self.upper_bounds,
#         )

#         for (transition_loc, alpha_f) in self.list_mixing_model_params:
#             model_name = f"mixing_loc{transition_loc:.1f}_alphaf{alpha_f:.1f}"

#             likelihood_mixing = MixingModelsLikelihood(
#                 self.forward_map,
#                 self.D,
#                 self.L,
#                 self.N,
#                 self.y,
#                 self.sigma_a,
#                 self.sigma_m,
#                 self.omega,
#                 transition_loc,
#                 alpha_f,
#             )
#             posterior_mixing = Posterior(
#                 self.D,
#                 self.L,
#                 self.N,
#                 likelihood_mixing,
#                 prior_spatial,
#                 prior_indicator,
#             )
#             self.dict_posteriors[model_name] = posterior_mixing

#         for is_raw in self.list_gaussian_approx_params:
#             name = "raw" if is_raw else "transformed"
#             model_name = f"gaussian_approx_{name}"

#             if is_raw:
#                 m_a = np.zeros((self.N, self.L))
#                 s_a = self.sigma_a * 1
#             else:
#                 m_a = (np.exp(self.sigma_m**2 / 2) - 1) * self.y
#                 s_a = np.sqrt(
#                     self.y**2
#                     * np.exp(self.sigma_m**2)
#                     * (np.exp(self.sigma_m**2) - 1)
#                     + self.sigma_a**2
#                 )

#             likelihood_censor = CensoredGaussianLikelihood(
#                 self.forward_map,
#                 self.D,
#                 self.L,
#                 self.N,
#                 self.y,
#                 s_a,
#                 self.omega,
#                 bias=m_a,
#             )
#             posterior_censor = Posterior(
#                 self.D,
#                 self.L,
#                 self.N,
#                 likelihood_censor,
#                 prior_spatial,
#                 prior_indicator,
#             )
#             self.dict_posteriors[model_name] = posterior_censor

#     def save_setup_to_csv(self):

#         dict_ = {
#             "N": {"value": self.N, "descr": "number of pixels to reconstruct"},
#             "D": {"value": self.D, "descr": "Number of physical parameters"},
#             "L": {"value": self.L, "descr": "number of observed lines"},
#             "sigma_a": {
#                 "value": self.sigma_a.mean(),
#                 "descr": "std of addititive noise",
#             },
#             "sigma_m": {
#                 "value": self.sigma_m.mean(),
#                 "descr": "std of multiplicative noise (computed)",
#             },
#             "omega": {
#                 "value": self.omega.mean(),
#                 "descr": "lower limit of intensity detection",
#             },
#             "LB_theta": {
#                 "value": self.lower_bounds_lin[0],
#                 "descr": "lower bound for kappa (lin space)",
#             },
#             "UB_theta": {
#                 "value": self.upper_bounds_lin[0],
#                 "descr": "upper bound for kappa (lin space)",
#             },
#             "indicator_margin_scale": {
#                 "value": self.indicator_margin_scale,
#                 "descr": "scale parameter of the smoothed indicator prior",
#             },
#             "transition_loc": {
#                 "value": self.transition_loc,
#                 "descr": "location parameter of the mixing models parameter",
#             },
#             "alpha_f": {
#                 "value": self.alpha_f,
#                 "descr": "speed parameter of the mixing models parameter",
#             },
#             "poly_approx_degree": {
#                 "value": self.deg,
#                 "descr": "degree of the polynomial used to approximate the PDR code",
#             },
#         }
#         df_hyperparameters = pd.DataFrame.from_dict(dict_, orient="index")
#         df_hyperparameters.index.name = "name"
#         df_hyperparameters.to_csv(f"{self.path_data_csv_in}/hyperparameters.csv")

#     def setup(self):

#         # observational data
#         self.L = 1  # number of lines
#         # self.N = 50  # number of observed pixels
#         # self.N = 20_000  # number of observed pixels # 10_000
#         self.D = 1  # Number of physical parameters
#         self.D_no_kappa = 1  # number of params that are used in polynom
#         self.list_names = [r"$x$"]

#         # degree of polynomial approximation
#         self.deg = 1

#         # observational params
#         self.sigma_a = 1.0 * np.ones((self.N, self.L))
#         self.sigma_m = np.log(1.1) * np.ones((self.N, self.L))
#         self.omega = -10 * np.ones((self.N, self.L))  # never censored
#         # self.omega = 3 * self.sigma_a

#         # prior hyperparameters
#         self.indicator_margin_scale = 1e-2
#         self.lower_bounds = np.array([-1])
#         self.upper_bounds = np.array([10])
#         self.lower_bounds_lin = self.lower_bounds.copy()
#         self.upper_bounds_lin = self.upper_bounds.copy()

#         # models mixing strategy (for mixing model)
#         self.transition_loc = 2
#         self.alpha_f = 2

#         # run setup
#         self.setup_output_folders("pwc_ts_mle")
#         list_cols_names = self.setup_forward_map()
#         syn_map = self.setup_observation(list_cols_names)
#         self.setup_posteriors(syn_map)

#         for model_name in list(self.dict_posteriors.keys()):
#             folder_path = f"{self.path_raw}/{model_name}"
#             if not os.path.isdir(folder_path):
#                 os.mkdir(folder_path)

#         # save setup
#         self.save_setup_to_csv()

#         self.plot_observations()
#         self.plot_censored_lines_proportion()

#         if self.Theta_true_scaled is not None:
#             folder_path = f"{self.path_img}/estimators"
#             folder_path_inter = f"{folder_path}/true"

#             for path_ in [folder_path, folder_path_inter]:
#                 if not os.path.isdir(path_):
#                     os.mkdir(path_)

#             Theta_true = self.scaler.from_scaled_to_lin(self.Theta_true_scaled)
#             self.plot_estimator(Theta_true, "true", folder_path_inter)

#     def run_optimization_MLE(self, psgld_params) -> None:
#         global _run_one_simulation_one_pixel

#         psgld_params.save_to_file(self.path_data_csv_in, "algo_params_optim_MLE.csv")

#         for n in range(self.N):
#             for model_name in list(self.dict_posteriors.keys()):
#                 for seed in range(self.N_MCMC):
#                     folder_path_inter = f"{self.path_raw}/{model_name}/opti_MLE_{seed}"

#                     folder_path = f"{folder_path_inter}/pixel_{n}"
#                     for path_ in [folder_path_inter, folder_path]:
#                         if not os.path.isdir(path_):
#                             os.mkdir(path_)

#         for model_name in list(self.dict_posteriors.keys()):
#             folder_path_inter = f"{self.path_img}/objective"
#             folder_path_inter2 = f"{folder_path_inter}/{model_name}"
#             folder_path = f"{folder_path_inter2}/MLE_objectives"
#             for path_ in [folder_path_inter, folder_path_inter2, folder_path]:
#                 if not os.path.isdir(path_):
#                     os.mkdir(path_)

#         def _run_one_simulation_one_pixel(dict_input: dict) -> dict:
#             n = dict_input["n"]
#             seed = dict_input["seed"]
#             model_name = dict_input["model_name"]

#             idx_model = list(self.dict_posteriors.keys()).index(model_name)
#             params = (self.list_mixing_model_params + self.list_gaussian_approx_params)[
#                 idx_model
#             ]

#             folder_path_inter = f"{self.path_raw}/{model_name}/opti_MLE_{seed}"
#             folder_path = f"{folder_path_inter}/pixel_{n}"

#             saver_ = saver.Saver(
#                 1,
#                 self.D,
#                 self.L,
#                 folder_path,
#                 self.scaler,
#                 self.batch_size,
#                 freq_save=1,
#             )
#             sampler = sampler.AugmentedPSGLD(
#                 psgld_params,
#                 self.D,
#                 self.D_no_kappa,
#                 self.L,
#                 1,
#                 np.random.default_rng(seed),
#             )
#             y_pix = self.y[n, :].reshape((1, self.L))
#             omega_pix = self.omega[n, :].reshape((1, self.L))

#             if "mixing" in model_name:
#                 sigma_a_pix = self.sigma_a[n, :].reshape((1, self.L))
#                 sigma_m_pix = self.sigma_m[n, :].reshape((1, self.L))

#                 transition_loc = params[0]
#                 alpha_f = params[1]

#                 likelihood_1pix = MixingModelsLikelihood(
#                     self.forward_map,
#                     self.D,
#                     self.L,
#                     1,
#                     y_pix,
#                     sigma_a_pix,
#                     sigma_m_pix,
#                     omega_pix,
#                     transition_loc,
#                     alpha_f,
#                 )
#             elif "gaussian_approx" in model_name:
#                 is_raw = params * 1
#                 if is_raw:
#                     m_a_pix = 0
#                     s_a_pix = self.sigma_a[n, :].reshape((1, self.L))
#                 else:
#                     m_a = (np.exp(self.sigma_m**2 / 2) - 1) * self.y
#                     s_a = np.sqrt(
#                         self.y**2
#                         * np.exp(self.sigma_m**2)
#                         * (np.exp(self.sigma_m**2) - 1)
#                         + self.sigma_a**2
#                     )

#                     m_a_pix = m_a[n, :].reshape((1, self.L))
#                     s_a_pix = s_a[n, :].reshape((1, self.L))

#                 likelihood_1pix = CensoredGaussianLikelihood(
#                     self.forward_map,
#                     self.D,
#                     self.L,
#                     1,
#                     y_pix,
#                     s_a_pix,
#                     omega_pix,
#                     bias=m_a_pix,
#                 )
#             else:
#                 raise ValueError(f"invalid model name : {model_name}")

#             posterior_1pix = Posterior(
#                 self.D,
#                 self.L,
#                 1,
#                 likelihood_1pix,
#                 None,
#                 self.prior_indicator_1pix,
#             )

#             tps0 = time.time()
#             sampler.sample(
#                 posterior_1pix,
#                 saver=saver_,
#                 max_iter=self.T_OPTI_MLE,
#                 sample_regu_weights=False,
#                 disable_progress_bar=True,
#             )

#             # return input dict with duration information
#             total_duration = time.time() - tps0

#             with h5py.File(f"{folder_path}/mc_chains.hdf5", "r") as f:
#                 list_Theta_lin = np.array(f["list_Theta"])  # (T, 1, D)
#                 list_objective = np.array(f["list_objective"])  # (T,)

#             shutil.rmtree(folder_path)  # allows to save memory

#             x_MLE_lin, objective = su.estimate_MAP_or_MLE(list_Theta_lin, list_objective)

#             if self.Theta_true_scaled is not None:
#                 x_n_scaled_true = self.Theta_true_scaled[n, :].reshape((1, self.D))
#                 mse = su.compute_MSE(x_MLE_lin, x_n_scaled_true)
#                 snr = su.compute_SNR(x_MLE_lin, x_n_scaled_true)

#             else:
#                 mse = None
#                 snr = None

#             dict_output = {
#                 "n": n,
#                 "seed": seed,
#                 "model_name": model_name,
#                 "total_duration": total_duration,
#                 "MSE": mse,
#                 "SNR": snr,
#                 "objective": objective,
#             }
#             for d in range(self.D):
#                 dict_output[f"x_MLE_{d}_lin"] = x_MLE_lin[0, d]
#                 if self.Theta_true_scaled is not None:
#                     Theta_true = self.scaler.from_scaled_to_lin(self.Theta_true_scaled)
#                     dict_output[f"x_true_{d}_lin"] = Theta_true[n, d]

#             # plot objective
#             # plt.figure(figsize=(8, 6))
#             # plt.title(f"Objective evolution during optimization for pixel {n}")
#             # plt.plot(range(self.T_OPTI_MLE), list_objective)
#             # if list_objective.min() < 0:
#             #     plt.yscale("symlog")
#             # else:
#             #     plt.yscale("log")
#             # plt.grid()
#             # plt.tight_layout()
#             # plt.savefig(f"{folder_path}/objective_pixel_{n}.PNG")
#             # plt.close()

#             return dict_output

#         print("starting optimization MLE")
#         list_params = [
#             {"n": n, "seed": seed, "model_name": model_name}
#             for n in range(self.N)
#             for seed in range(self.N_MCMC)
#             for model_name in list(self.dict_posteriors.keys())
#         ]

#         with ProcessPoolExecutor(
#             max_workers=self.max_workers, mp_context=mp.get_context("fork")
#         ) as p:
#             list_results = list(
#                 tqdm(
#                     p.map(_run_one_simulation_one_pixel, list_params),
#                     total=len(list_params),
#                 )
#             )

#         df_results_mle = pd.DataFrame(list_results)
#         df_results_mle.to_csv(f"{self.path_data_csv_out}/results_MLE.csv")

#         print("optimization MLE done")

#     def optimization_MLE_results_analysis(self) -> None:
#         global _build_MLE

#         df_results_mle = pd.read_csv(
#             f"{self.path_data_csv_out}/results_MLE.csv", index_col=0
#         )

#         def _build_MLE(model_name: str):
#             df_mle_best = (
#                 df_results_mle[df_results_mle["model_name"] == model_name]
#                 .groupby("n")["objective"]
#                 .min()
#             )
#             df_mle_best = df_mle_best.reset_index()

#             df_mle_final = pd.merge(
#                 df_results_mle, df_mle_best, on=["n", "objective"], how="inner"
#             )
#             df_mle_final = df_mle_final.sort_values("n")
#             df_mle_final = df_mle_final.drop_duplicates(["n", "objective"])

#             Theta_MLE = np.zeros((self.N, self.D))
#             for d in range(self.D):
#                 Theta_MLE[:, d] = df_mle_final.loc[:, f"x_MLE_{d}_lin"].values

#             return Theta_MLE, df_mle_final

#         # plot MLE for each model
#         for model_name in list(self.dict_posteriors.keys()):

#             Theta_MLE, df_mle_final = _build_MLE(model_name)
#             if self.Theta_true_scaled is not None:
#                 Theta_MLE_scaled = self.scaler.from_lin_to_scaled(Theta_MLE)

#                 mse_whole = su.compute_MSE(Theta_MLE_scaled, self.Theta_true_scaled)
#                 snr_whole = su.compute_SNR(Theta_MLE_scaled, self.Theta_true_scaled)
#             else:
#                 mse_whole = None
#                 snr_whole = None

#             objective_whole = df_mle_final["objective"].sum()

#             folder_path_inter = f"{self.path_img}/objective"
#             folder_path_inter2 = f"{folder_path_inter}/{model_name}"
#             folder_path = f"{folder_path_inter2}/MLE_objectives"
#             for path_ in [folder_path_inter, folder_path_inter2, folder_path]:
#                 if not os.path.isdir(path_):
#                     os.mkdir(path_)

#             self.save_estimator_performance(
#                 "MLE", model_name, mse_whole, snr_whole, objective_whole
#             )
#             Theta_true = self.scaler.from_scaled_to_lin(self.Theta_true_scaled)
#             self.plot_estimator(Theta_MLE, "MLE", folder_path, model_name, Theta_true=Theta_true)

#         if self.Theta_true_scaled is not None:
#             print("starting plots of R-SNR in function of SNR")
#             list_snr_obs = (20 * np.log10(self.y / self.sigma_a)).mean(1)  # (N,)

#             plt.figure(figsize=(8, 6))
#             plt.title(f"MLE MSE in function of additive SNR")

#             for model_name in list(self.dict_posteriors.keys()):
#                 Theta_MLE_model, _ = _build_MLE(model_name)
#                 Theta_MLE_model_scaled = self.scaler.from_lin_to_scaled(Theta_MLE_model)
#                 list_mse = np.array(
#                     [
#                         su.compute_MSE(
#                             Theta_MLE_model_scaled[n, :], self.Theta_true_scaled[n, :]
#                         )
#                         for n in range(self.N)
#                     ]
#                 )  # (N,)
#                 plt.semilogy(list_snr_obs, list_mse, "+", label=model_name)

#             plt.xlabel(r"pixelwise average observation-to-additive noise ratio")
#             plt.ylabel(f"pixelwise MSE")
#             plt.grid()
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(f"{self.path_img}/pixelwise_MLE_MSE.PNG")
#             plt.close()
#             print("plots of R-SNR in function of SNR done")
