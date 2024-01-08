# import numpy as np
# import pandas as pd
# import pytest

# from beetroots.modelling.priors.tv_1D_prior import TVeps1DSpatialPrior


# @pytest.fixture(scope="module")
# def build_first_test():
#     D = 1

#     df = pd.DataFrame()
#     df["X"] = [1, 2, 3]
#     df["Y"] = 1
#     df = df.set_index(["X", "Y"])
#     df = df.sort_index()

#     df["vals"] = 0
#     df.at[(2, 1), "vals"] = 1
#     df["idx"] = np.arange(len(df))

#     N = len(df)
#     weights = 2 * np.ones((D,))
#     eps = 1e-4
#     prior = TVeps1DSpatialPrior(D, N, df, weights, eps)
#     x = df["vals"].values.reshape((N, D))
#     return x, prior


# def test_init(build_first_test):
#     _, prior = build_first_test
#     assert prior.list_edges.shape == (2, 2)


# def test_neglog_pdf(build_first_test):
#     x, prior = build_first_test
#     assert prior.neglog_pdf(x) == prior.weights * 2 * np.sqrt(1 + prior.eps)


# def test_gradient_neglogpdf(build_first_test):
#     x, prior = build_first_test
#     manual_grad = np.array([-1, 2, -1], dtype=np.float64)
#     manual_grad *= 1 / np.sqrt(1 + prior.eps)
#     manual_grad = manual_grad.reshape((prior.N, prior.D))
#     manual_grad = prior.weights[None, :] * manual_grad

#     assert np.allclose(prior.gradient_neglog_pdf(x), manual_grad)


# def test_hess_diag_neglog_pdf(build_first_test):
#     x, prior = build_first_test
#     manual_hess_diag = np.array([1, 2, 1], dtype=np.float64)
#     manual_hess_diag *= (2 * prior.eps) / (1 ** 2 + prior.eps) ** (3 / 2)
#     manual_hess_diag = manual_hess_diag.reshape((prior.N, prior.D))
#     manual_hess_diag = prior.weights[None, :] * manual_hess_diag
#     print(prior.hess_diag_neglog_pdf(x))
#     print(manual_hess_diag)
#     assert np.allclose(prior.hess_diag_neglog_pdf(x), manual_hess_diag)
