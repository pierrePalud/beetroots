# import os

# import h5py
# import numpy as np
# import pandas as pd
# import pytest

# from beetroots.modelling.forward_maps.basic_exp import BasicExpForwardMap
# from beetroots.modelling.likelihoods.gaussian import GaussianLikelihood
# from beetroots.modelling.posterior import Posterior
# from beetroots.modelling.priors.l22_prior import L22SpatialPrior
# from beetroots.sampler import mysampler, saver
# from beetroots.sampler.psgldparams import MySamplerParams
# from beetroots.space_transform.id_transform import IdScaler


# @pytest.fixture
# def setup():
#     D = 4
#     D_no_kappa = 3
#     N = 100
#     N_axis = int(pow(N, 0.5))
#     L = 4
#     return D, L, N, D_no_kappa


# @pytest.fixture
# def posterior(setup):
#     D, L, N, D_no_kappa = setup
#     sigma_a = 1
#     y = 10 * np.ones((N, L))

#     df = pd.DataFrame()
#     df["x"] = np.arange(N)
#     df["y"] = 0
#     df["idx"] = np.arange(N)
#     df = df.set_index(["x", "y"])

#     forward_map = BasicExpForwardMap(D, L, N)
#     likelihood = GaussianLikelihood(forward_map, D, L, N, y, sigma_a)
#     prior_spatial = L22SpatialPrior(D, N, df)
#     posterior = Posterior(D, L, N, likelihood, prior_spatial)
#     return posterior


# def test_reproducibility(setup, posterior, tmp_path):
#     D, L, N, D_no_kappa = setup

#     dir_name_1 = f"{tmp_path}/test1"
#     dir_name_2 = f"{tmp_path}/test2"
#     for path in [dir_name_1, dir_name_2]:
#         if not os.path.isdir(path):
#             os.mkdir(path)

#     scaler = IdScaler()
#     my_sampler_params = MySamplerParams(1.5e-3, 1e-5, 0.995, 0.05, 1.0, 4**D, True, True)

#     # first sampling
#     batch_size = 5

#     max_iter_1 = 20
#     T_bi = 10
#     max_iter_2 = 10
#     assert max_iter_1 == T_bi + max_iter_2  # assert on test setup

#     saver_1 = saver.Saver(N, D, L, dir_name_1, scaler, batch_size, 1)
#     saver_2 = saver.Saver(N, D, L, dir_name_2, scaler, batch_size, 1)

#     sampler_1 = mysampler.MySampler(my_sampler_params, D, D_no_kappa, L, N)
#     sampler_1.sample(
#         posterior,
#         saver=saver_1,
#         max_iter=max_iter_1,
#         sample_regu_weights=True,
#     )

#     with h5py.File(f"{dir_name_1}/mc_chains.hdf5", "r") as f:
#         list_Theta_lin_1 = np.array(f["list_Theta"][T_bi:])
#         list_v_1 = np.array(f["list_v"][T_bi:])
#         list_tau_1 = np.array(f["list_tau"][T_bi:])
#         list_objective_1 = np.array(f["list_objective"][T_bi:])
#         list_log_proba_accept_1 = np.array(f["list_log_proba_accept_t"][T_bi:])

#         Theta_0 = scaler.from_lin_to_scaled(np.array(f["list_Theta"][T_bi - 1]))
#         v0 = np.array(f["list_v"][T_bi - 1]).flatten()
#         rng_state_1 = np.array(f["list_rng_state"][T_bi - 1])
#         rng_inc_1 = np.array(f["list_rng_inc"][T_bi - 1])

#     assert list_Theta_lin_1.shape == (max_iter_2, N, D)
#     assert list_v_1.shape == (max_iter_2, N, D)
#     assert list_tau_1.shape == (max_iter_2, D)
#     assert list_objective_1.shape == (max_iter_2,)

#     sampler_2 = mysampler.MySampler(my_sampler_params, D, D_no_kappa, L, N)
#     sampler_2.set_rng_state(rng_state_1, rng_inc_1)
#     sampler_2.sample(
#         posterior,
#         saver=saver_2,
#         max_iter=max_iter_2,
#         Theta_0=Theta_0,
#         v0=v0,
#         sample_regu_weights=True,
#     )

#     with h5py.File(f"{dir_name_2}/mc_chains.hdf5", "r") as f:
#         list_Theta_lin_2 = np.array(f["list_Theta"])
#         list_v_2 = np.array(f["list_v"])
#         list_tau_2 = np.array(f["list_tau"])
#         list_objective_2 = np.array(f["list_objective"])
#         list_log_proba_accept_2 = np.array(f["list_log_proba_accept_t"])

#     # test correct shape
#     assert list_Theta_lin_2.shape == (max_iter_2, N, D)
#     assert list_v_2.shape == (max_iter_2, N, D)
#     assert list_tau_2.shape == (max_iter_2, D)
#     assert list_objective_2.shape == (max_iter_2,)

#     # test exact same values (reproduced exactly same sampling)
#     assert np.allclose(list_log_proba_accept_1, list_log_proba_accept_2)
#     assert np.allclose(list_Theta_lin_1, list_Theta_lin_2)
#     assert np.allclose(list_v_1, list_v_2)
#     assert np.allclose(list_tau_1, list_tau_2)
#     assert np.allclose(list_objective_1, list_objective_2)
