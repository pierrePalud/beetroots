# """Sequatially :
# 1. Builds the piecewise constant time series toy case
# 2. launches multiple sampling (for both mixing model and simple gaussian)
# 3. launches multiple MAP estimation (for both mixing model and simple gaussian)
# 4. launches multiple MLE estimate (for both mixing model and simple gaussian)
# """
# import os
# import pickle

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.stats import lognorm, norm

# # import beetroots.utils as utils
# from beetroots.modelling.forward_maps.basic_exp import BasicExpForwardMap
# from beetroots.modelling.likelihoods.approx_censored_add_mult import (
#     MixingModelsLikelihood,
# )
# from beetroots.modelling.likelihoods.gaussian_censored import CensoredGaussianLikelihood
# from beetroots.modelling.posterior import Posterior
# from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
# from beetroots.simulations.abstract_simulation import Simulation
# from beetroots.space_transform.id_transform import IdScaler


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
#         with_spatial_prior: bool = True,
#         grid_name: str = "",  # useless in this class
#         spatial_prior_name: str = "L1-gradient",
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
#             with_spatial_prior,
#             grid_name="",  # useless in this class
#             spatial_prior_name=spatial_prior_name,
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
#         syn_map["x"] = np.round(np.linspace(0, 9, self.N))  #! increase number of steps
#         syn_map["y"] = np.exp(syn_map["x"])
#         syn_map["idx"] = np.arange(self.N)
#         syn_map["idx1"] = np.arange(self.N)
#         syn_map["idx2"] = 0
#         syn_map = syn_map.set_index(["idx1", "idx2"])

#         self.Theta_true_scaled = syn_map.iloc[:, : self.D].values
#         assert self.Theta_true_scaled.shape == (self.N, self.D)

#         syn_map.iloc[:, self.D : -1] = self.forward_map.evaluate(self.Theta_true_scaled)

#         self.list_lines = list(syn_map.columns)[self.D : -1]

#         # save
#         syn_map_tocsv = syn_map.copy()
#         syn_map_tocsv.iloc[:, : self.D] = self.scaler.from_scaled_to_lin(
#             syn_map_tocsv.iloc[:, : self.D].values
#         )
#         Theta_true = syn_map_tocsv.iloc[:, : self.D].values

#         syn_map_tocsv.to_csv(
#             f"{self.path_data_csv_in}/true_params_and_emissions_maps.csv"
#         )

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
#         prior_spatial = self.get_spatial_prior_from_name(syn_map)

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
#         # self.N = 100  # number of observed pixels
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

#         # run setup
#         self.setup_output_folders(self.cloud_name)
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
