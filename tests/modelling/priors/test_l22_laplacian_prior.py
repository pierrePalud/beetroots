# import numpy as np
# import pandas as pd
# import pytest

# from beetroots.modelling.priors.l22_laplacian_prior import (
#     L22LaplacianSpatialPrior,
#     compute_laplacian,
#     compute_neglog_chromatic_from_laplacian2,
# )
# from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams


# @pytest.fixture(scope="module")
# def build_test_1():
#     D = 1
#     D_full = 2
#     N = 9
#     scalar = 3.0

#     x, y = np.meshgrid(list(range(3)), list(range(3)))
#     df = pd.DataFrame()
#     df["x"] = x.flatten()
#     df["y"] = y.flatten()
#     df["idx"] = np.arange(len(df))
#     df["vals_1"] = 0.0
#     assert len(df) == 9

#     df = df.set_index(["x", "y"])
#     df.loc[(1, 1), "vals_1"] = 1.0
#     df["vals_2"] = scalar * df["vals_1"]

#     spatial_prior_params = SpatialPriorParams(
#         name="L2-laplacian",
#         use_next_nearest_neighbors=False,
#         initial_regu_weights=np.ones((D,)),
#     )
#     spatial_prior_params_full = SpatialPriorParams(
#         name="L2-laplacian",
#         use_next_nearest_neighbors=False,
#         initial_regu_weights=np.ones((D_full,)),
#     )

#     prior = L22LaplacianSpatialPrior(
#         spatial_prior_params, "", D, N, df, list_idx_sampling=list(range(D))
#     )
#     prior_full = L22LaplacianSpatialPrior(
#         spatial_prior_params_full,
#         "",
#         D_full,
#         N,
#         df,
#         list_idx_sampling=list(range(D_full)),
#     )

#     Theta1 = df["vals_1"].values.reshape((N, D))
#     Theta2 = df["vals_2"].values.reshape((N, D))
#     Theta_full = df[["vals_1", "vals_2"]].values.reshape((N, D_full))
#     return Theta1, Theta2, Theta_full, prior, prior_full, scalar


# def test_compute_laplacian(build_test_1):
#     Theta1, Theta2, Theta_full, prior, prior_full, scalar = build_test_1
#     N, D = Theta1.shape
#     _, D_full = Theta_full.shape

#     laplacian_1 = compute_laplacian(Theta1, prior.list_edges, idx_pix=np.arange(N))
#     true_laplacian_1 = np.array(
#         [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0]
#     ).reshape((N, D))
#     assert laplacian_1.shape == (N, D)
#     assert np.allclose(true_laplacian_1, laplacian_1)

#     laplacian_2 = compute_laplacian(Theta2, prior.list_edges, idx_pix=np.arange(N))
#     true_laplacian_2 = scalar * true_laplacian_1
#     assert np.allclose(true_laplacian_2, laplacian_2)

#     laplacian_full = compute_laplacian(
#         Theta_full, prior_full.list_edges, idx_pix=np.arange(N)
#     )
#     true_laplacian_full = np.hstack([true_laplacian_1, true_laplacian_2])  # (N, D_full)
#     assert np.allclose(true_laplacian_full, laplacian_full)


# def test_neglog_pdf(build_test_1):
#     Theta1, Theta2, Theta_full, prior, prior_full, scalar = build_test_1
#     N, D = Theta1.shape
#     _, D_full = Theta_full.shape

#     neglogpdf_1_chromatic = prior.neglog_pdf(
#         Theta1,
#         idx_pix=np.arange(N),
#         with_weights=True,
#         full=False,
#         chromatic_gibbs=True,
#     )
#     neglogpdf_1_no_chromatic = prior.neglog_pdf(
#         Theta1,
#         idx_pix=np.arange(N),
#         with_weights=True,
#         full=False,
#         chromatic_gibbs=False,
#     )
#     true_val_1_chromatic = np.array([20 + 4 * 17 + 4 * 2])
#     true_val_1_no_chromatic = np.array([16 + 4 * 1])

#     assert neglogpdf_1_chromatic.shape == (D,)
#     assert np.allclose(true_val_1_chromatic, neglogpdf_1_chromatic)
#     assert np.allclose(true_val_1_no_chromatic, neglogpdf_1_no_chromatic)

#     neglogpdf_2 = prior.neglog_pdf(
#         Theta2,
#         idx_pix=np.arange(N),
#         with_weights=True,
#         full=False,
#         chromatic_gibbs=True,
#     )
#     true_val_2_chromatic = np.array([180 + 4 * 153 + 4 * 18])
#     assert np.allclose(true_val_2_chromatic, neglogpdf_2)

#     neglogpdf_full = prior_full.neglog_pdf(
#         Theta_full,
#         idx_pix=np.arange(N),
#         with_weights=True,
#         full=False,
#         chromatic_gibbs=True,
#     )
#     true_val_full = np.hstack([true_val_1_chromatic, true_val_2_chromatic])  # (D_full,)
#     assert neglogpdf_full.shape == (D_full,)
#     assert np.allclose(true_val_full, neglogpdf_full)


# def test_gradient_neglogpdf(build_test_1):
#     Theta1, Theta2, Theta_full, prior, prior_full, scalar = build_test_1
#     N, D = Theta1.shape
#     _, D_full = Theta_full.shape

#     true_grad_1 = np.array(
#         [4.0, -14.0, 4.0, -14.0, 40.0, -14.0, 4.0, -14.0, 4.0]
#     ).reshape((N, D))

#     grad_1 = prior.gradient_neglog_pdf(Theta1, idx_pix=np.arange(N))
#     assert grad_1.shape == (N, D)
#     assert np.allclose(true_grad_1, grad_1)

#     grad_2 = prior.gradient_neglog_pdf(Theta2, idx_pix=np.arange(N))
#     true_grad_2 = scalar * true_grad_1
#     assert np.allclose(true_grad_2, grad_2)

#     grad_full = prior_full.gradient_neglog_pdf(Theta_full, idx_pix=np.arange(N))
#     true_grad_full = np.hstack([true_grad_1, true_grad_2])
#     assert grad_full.shape == (N, D_full)
#     assert np.allclose(true_grad_full, grad_full)

#     grad_pix = prior.gradient_neglog_pdf(Theta1, idx_pix=np.array([4]))
#     assert grad_pix.shape == (1, D)
#     assert np.allclose(true_grad_1[4], grad_pix)


# def test_hess_diag_neglogpdf(build_test_1):
#     Theta1, Theta2, Theta_full, prior, prior_full, scalar = build_test_1
#     N, D = Theta1.shape
#     _, D_full = Theta_full.shape

#     true_hess_1 = np.array(
#         [12.0, 24.0, 12.0, 24.0, 40.0, 24.0, 12.0, 24.0, 12.0]
#     ).reshape((N, D))

#     hess_1 = prior.hess_diag_neglog_pdf(Theta1, idx_pix=np.arange(N))
#     assert hess_1.shape == (N, D)
#     assert np.allclose(true_hess_1, hess_1)

#     hess_2 = prior.hess_diag_neglog_pdf(Theta2, idx_pix=np.arange(N))
#     true_hess_2 = true_hess_1 * 1.0
#     assert np.allclose(true_hess_2, hess_2)

#     hess_full = prior_full.hess_diag_neglog_pdf(Theta_full, idx_pix=np.arange(N))
#     true_hess_full = np.hstack([true_hess_1, true_hess_2])
#     assert hess_full.shape == (N, D_full)
#     assert np.allclose(true_hess_full, hess_full)

#     hess_pix = prior.hess_diag_neglog_pdf(Theta1, idx_pix=np.array([4]))
#     assert hess_pix.shape == (1, D)
#     assert np.allclose(true_hess_1[4], hess_pix)
