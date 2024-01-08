# import os
# from typing import Sequence, Union

# import numpy as np

# from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
# from beetroots.sampler.utils.psgldparams import PSGLDParams
# from beetroots.simulations.astro_toy_case_nn import SimulationAstroToyCase

# if __name__ == "__main__":
#     N_1_side = 27

#     spatial_prior_params = SpatialPriorParams(
#         name="L2-laplacian",
#         use_next_nearest_neighbours=False,
#         use_clustering=False,
#         n_clusters=None,
#         cluster_algo=None,
#     )

#     list_gaussian_approx_params = []
#     list_mixing_model_params = [
#         {
#             "path_transition_params": f"{os.path.dirname(os.path.abspath(__file__))}/../../../data/toycases/params_entropy_for_gaussian_only_L{L}.csv"
#         }
#     ]

#     list_lines_fit = [
#         "chp_j1__j0",
#         "co_v0_j10__v0_j9",
#         "co_v0_j2__v0_j1",
#         "hcop_j3__j2",
#         "cp_el2p_j3_2__el2p_j1_2",
#     ]

#     psgld_params_mcmc = PSGLDParams(
#         5e-4, 1e-5, 0.99, np.array([0.5, 0.5]), 20, True, True
#     )

#     simulation1 = SimulationAstroToyCase(
#         max_workers=20,
#         N_MCMC=1,  # 5
#         T_MC=1_000,  # 100_000
#         T_OPTI=1000,  # 15_000
#         T_OPTI_MLE=50,  # 500
#         T_BI=200,  # 20_000
#         batch_size=5,
#         signal_2dim=True,
#         list_gaussian_approx_params=list_gaussian_approx_params,
#         list_mixing_model_params=list_mixing_model_params,
#         list_lines_fit=list_lines_fit,
#         N=N_1_side * N_1_side,
#         forward_model_name="meudon_pdr_model_dense",
#         angle=0.0,
#         with_spatial_prior=False,  # True,
#         spatial_prior_params=None,  # spatial_prior_params,
#         # initial_spatial_weights=np.array([0.2, 0.2, 0.2, 0.2]),
#     )

#     simulation1.setup()
#     simulation1.main(
#         run_mle=False,
#         run_map=False,
#         run_mcmc=True,
#         psgld_params_mcmc=psgld_params_mcmc,
#         psgld_params_map=None,
#         psgld_params_mle=None,
#         #
#         regu_spatial_N0=np.infty,  #! set to +np.infty to not optimize \tau
#         regu_spatial_scale=2e1,
#         regu_spatial_vmin=1e-2,
#         regu_spatial_vmax=1e4,
#         #
#         plot_ESS=False,
#         plot_1D_chains=False,
#         plot_2D_chains=True,
#         plot_comparisons_yspace=False,
#         start_mcmc_from=None,
#         freq_save_mcmc=1,
#     )
