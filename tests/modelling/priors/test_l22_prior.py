# import numpy as np
# import pandas as pd
# import pytest

# from beetroots.modelling.priors.l22_prior import L22SpatialPrior


# @pytest.fixture(scope="module")
# def build_first_test():
#     D = 1

#     df = pd.DataFrame()
#     x, y = np.meshgrid([0, 1, 2], [0, 1, 2])
#     df["X"] = x.flatten()
#     df["Y"] = y.flatten()
#     df = df.set_index(["X", "Y"])
#     df = df.sort_index()

#     df["vals"] = 0
#     df.at[(1, 1), "vals"] = 1
#     df["idx"] = np.arange(len(df))

#     N = len(df)
#     weights = 2 * np.ones((D,))
#     prior = L22SpatialPrior(D, N, df, weights)
#     x = df["vals"].values.reshape((N, D))
#     return x, prior


# @pytest.fixture(scope="module")
# def build_test_local():
#     D = 1
#     D_full = 2
#     N_side = 5
#     N = N_side ** 2
#     scalar = 3.0

#     x, y = np.meshgrid(list(range(N_side)), list(range(N_side)))
#     df = pd.DataFrame()
#     df["x"] = x.flatten()
#     df["y"] = y.flatten()
#     df["idx"] = np.arange(N)
#     df["vals_1"] = np.mod(df["idx"], 2.0)
#     assert len(df) == N

#     df = df.set_index(["x", "y"])
#     df["vals_2"] = scalar * df["vals_1"]

#     prior = L22SpatialPrior(D, N, df)
#     prior_full = L22SpatialPrior(D_full, N, df)

#     x1 = df["vals_1"].values.reshape((N, D))
#     x2 = df["vals_2"].values.reshape((N, D))
#     x_full = df[["vals_1", "vals_2"]].values.reshape((N, D_full))
#     return x1, x2, x_full, prior, prior_full, scalar


# @pytest.fixture(scope="module")
# def build_second_test():
#     D = 1

#     df = pd.DataFrame()
#     x, y = np.meshgrid([0, 1, 2], [0, 1, 2])
#     df["X"] = x.flatten()
#     df["Y"] = y.flatten()
#     df = df[(df["X"] + df["Y"]) % 2 == 0]
#     df = df.set_index(["X", "Y"])
#     df = df.sort_index()

#     df["vals"] = 0
#     df.at[(1, 1), "vals"] = 1
#     df["idx"] = np.arange(len(df))

#     N = len(df)

#     weights = 2 * np.ones((D,))
#     prior = L22SpatialPrior(D, N, df, weights)
#     x = df["vals"].values.reshape((N, D))
#     return x, prior


# @pytest.fixture(scope="module")
# def build_third_test():
#     D = 1

#     df = pd.DataFrame()
#     x, y = np.meshgrid([0, 1, 2], [0, 1, 2])
#     df["X"] = x.flatten()
#     df["Y"] = y.flatten()
#     df = df.set_index(["X", "Y"])
#     df = df.sort_index()

#     # drop only one pixel
#     df = df.drop([(1, 0)])

#     df["vals"] = 0
#     df.at[(1, 1), "vals"] = 1
#     df["idx"] = np.arange(len(df))

#     N = len(df)

#     weights = 2 * np.ones((D,))
#     prior = L22SpatialPrior(D, N, df, weights)
#     x = df["vals"].values.reshape((N, D))
#     return x, prior


# def test_init(build_first_test, build_second_test, build_third_test):
#     _, prior = build_first_test
#     assert prior.list_edges.shape == (12, 2)

#     _, prior = build_second_test
#     assert prior.list_edges.shape == (0,)

#     _, prior = build_third_test
#     assert prior.list_edges.shape == (9, 2)


# def test_neglog_pdf(build_first_test, build_second_test, build_third_test):
#     x, prior = build_first_test
#     assert np.allclose(prior.neglog_pdf(x), prior.weights * 4)

#     x, prior = build_second_test
#     assert np.allclose(prior.neglog_pdf(x), prior.weights * 0)

#     x, prior = build_third_test
#     assert np.allclose(prior.neglog_pdf(x), prior.weights * 3)


# def test_neglog_pdf_one_pix(build_test_local):
#     x1, x2, x_full, prior, prior_full, scalar = build_test_local
#     N, D = x1.shape
#     _, D_full = x_full.shape
#     N_side = int(np.sqrt(N))

#     for n in range(N):
#         is_pix_in_border_right = n % N_side == N_side - 1  # right
#         is_pix_in_border_down = n // N_side == N_side - 1  # bottom

#         true_1 = 1.0 ** 2 * (2 - is_pix_in_border_right - is_pix_in_border_down)
#         true_2 = scalar ** 2 * (2 - is_pix_in_border_right - is_pix_in_border_down)

#         neglog_pdf_n_1 = prior.neglog_pdf_one_pix(x1, n)
#         assert np.allclose(np.array([true_1]), neglog_pdf_n_1)

#         neglog_pdf_n_2 = prior.neglog_pdf_one_pix(x2, n)  # (D,)
#         assert np.allclose(np.array([true_2]), neglog_pdf_n_2)

#         neglog_pdf_n_full = prior_full.neglog_pdf_one_pix(x_full, n)  # (D_full,)
#         assert np.allclose(np.array([true_1, true_2]), neglog_pdf_n_full)


# def test_gradient_neglogpdf(build_first_test, build_second_test, build_third_test):
#     x, prior = build_first_test
#     manual_grad = np.array([0, -2, 0, -2, 8, -2, 0, -2, 0], dtype=np.float64)
#     manual_grad = manual_grad.reshape((prior.N, prior.D))
#     manual_grad = prior.weights[None, :] * manual_grad

#     assert np.allclose(prior.gradient_neglog_pdf(x), manual_grad)

#     x, prior = build_second_test
#     manual_grad = np.array([0, 0, 0, 0, 0], dtype=np.float64)
#     manual_grad = manual_grad.reshape((prior.N, prior.D))
#     manual_grad = prior.weights[None, :] * manual_grad

#     assert np.allclose(prior.gradient_neglog_pdf(x), manual_grad)

#     x, prior = build_third_test
#     manual_grad = np.array([0, -2, 0, 6, -2, 0, -2, 0], dtype=np.float64)
#     manual_grad = manual_grad.reshape((prior.N, prior.D))
#     manual_grad = prior.weights[None, :] * manual_grad

#     assert np.allclose(prior.gradient_neglog_pdf(x), manual_grad)


# def test_hess_diag_neglogpdf(build_first_test, build_second_test, build_third_test):
#     x, prior = build_first_test
#     manual_hess_diag = np.array([4, 6, 4, 6, 8, 6, 4, 6, 4], dtype=np.float64)
#     manual_hess_diag = manual_hess_diag.reshape((prior.N, prior.D))
#     manual_hess_diag = prior.weights[None, :] * manual_hess_diag

#     assert np.allclose(prior.hess_diag_neglog_pdf(x), manual_hess_diag)

#     x, prior = build_second_test
#     manual_hess_diag = np.array([0, 0, 0, 0, 0], dtype=np.float64)
#     manual_hess_diag = manual_hess_diag.reshape((prior.N, prior.D))
#     manual_hess_diag = prior.weights[None, :] * manual_hess_diag

#     assert np.allclose(prior.hess_diag_neglog_pdf(x), manual_hess_diag)

#     x, prior = build_third_test
#     manual_hess_diag = np.array([2, 6, 4, 6, 6, 2, 6, 4], dtype=np.float64)
#     manual_hess_diag = manual_hess_diag.reshape((prior.N, prior.D))
#     manual_hess_diag = prior.weights[None, :] * manual_hess_diag

#     assert np.allclose(prior.hess_diag_neglog_pdf(x), manual_hess_diag)
