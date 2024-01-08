# TODO: update and test

# import numpy as np
# import pytest
# from scipy.stats import norm as statsnorm

# from beetroots.modelling.forward_maps.basic_exp import BasicExpForwardMap
# from beetroots.modelling.likelihoods import utils
# from beetroots.modelling.likelihoods.approx_censored_add_mult import (
#     MixingModelsLikelihood,
# )


# @pytest.fixture(scope="module")
# def settings():
#     D = 1
#     L = 1
#     N = 21
#     return D, L, N


# @pytest.fixture(scope="module")
# def points(settings):
#     D, L, N = settings

#     x1 = 5 * np.ones((N, D))  # at init : uncensored, dominated by additive model
#     x1[: N // 3] = 0  # censored
#     x1[-N // 3 :] = 10  # uncensored, dominated by multiplicative noise

#     x2 = np.ones((N, D))
#     return x1, x2


# @pytest.fixture(scope="module")
# def my_likelihood(settings, points):
#     D, L, N = settings
#     x1, x2 = points
#     forward_map = BasicExpForwardMap(D, L, N)

#     sigma_a = 0.5
#     omega = 3 * sigma_a
#     sigma_m = 0.1

#     y = forward_map.evaluate(x1)  # fully uncensored
#     y = np.maximum(y, omega)

#     path_transition_params = f"./data/test/best_params.csv"

#     my_likelihood = MixingModelsLikelihood(
#         forward_map, D, L, N, y, sigma_a, sigma_m, omega, path_transition_params
#     )
#     return my_likelihood


# @pytest.fixture(scope="module")
# def forward_map_evals_Theta1(my_likelihood, points):
#     x1, _ = points
#     forward_map_evals_Theta1 = my_likelihood.evaluate_all_forward_map(x1, True)
#     return forward_map_evals_Theta1


# @pytest.fixture(scope="module")
# def forward_map_evals_Theta2(my_likelihood, points):
#     _, x2 = points
#     forward_map_evals_Theta2 = my_likelihood.evaluate_all_forward_map(x2, True)
#     return forward_map_evals_Theta2


# @pytest.fixture(scope="module")
# def nll_utils_Theta1(my_likelihood, forward_map_evals_Theta1):
#     nll_utils_Theta1 = my_likelihood.evaluate_all_nll_utils(forward_map_evals_Theta1)
#     return nll_utils_Theta1


# @pytest.fixture(scope="module")
# def nll_utils_Theta2(my_likelihood, forward_map_evals_Theta2):
#     nll_utils_Theta2 = my_likelihood.evaluate_all_nll_utils(forward_map_evals_Theta2)
#     return nll_utils_Theta2


# def test_init(settings, my_likelihood):
#     D, L, N = settings
#     assert my_likelihood.sigma_a.shape == (N, L)
#     assert my_likelihood.sigma_m.shape == (N, L)
#     assert my_likelihood.omega.shape == (N, L)


# def test_evaluate_all_forward_map(settings, my_likelihood, points):
#     D, L, N = settings
#     x1, x2 = points

#     forward_map_evals = my_likelihood.evaluate_all_forward_map(x1, True)

#     list_keys = list(forward_map_evals.keys())
#     list_keys_manual = [
#         "f_Theta",
#         "grad_f_Theta",
#         "hess_diag_f_Theta",
#         "log_f_Theta",
#         "grad_log_f_Theta",
#         "hess_diag_log_f_Theta",
#     ]
#     assert sorted(list_keys) == sorted(list_keys_manual)


# def test_evaluate_all_nll_utils(
#     settings,
#     my_likelihood,
#     points,
#     forward_map_evals_Theta1,
# ):
#     nll_utils = my_likelihood.evaluate_all_nll_utils(forward_map_evals_Theta1)
#     list_keys = list(nll_utils.keys())
#     list_keys_manual = [
#         "m_a",
#         "s_a",
#         "m_m",
#         "s_m",
#         "lambda_",
#         "grad_lambda_",
#         "hess_diag_lambda_",
#     ]
#     assert sorted(list_keys) == sorted(list_keys_manual)

#     nll_utils = my_likelihood.evaluate_all_nll_utils(
#         forward_map_evals_Theta1, idx=0
#     )  # as if x1 was a vector of N candidates for pixel n=0
#     assert sorted(list_keys) == sorted(list_keys_manual)


# def test_neglog_pdf(
#     settings,
#     my_likelihood,
#     forward_map_evals_Theta1,
#     forward_map_evals_Theta2,
#     nll_utils_Theta1,
#     nll_utils_Theta2,
# ):
#     D, L, N = settings

#     # from constant np.ndarray to float
#     omega = my_likelihood.omega.mean()

#     nll_Theta1 = my_likelihood.neglog_pdf(
#         forward_map_evals_Theta1, nll_utils_Theta1, pixelwise=False
#     )
#     # first third : censored (with constant),
#     # second third : uncensored (add.) at exact value,
#     # third third : uncensored (mult.)
#     # nll_Theta1_manual = (
#     #     -L
#     #     * (N // 3)
#     #     * statsnorm.logcdf(
#     #         (omega - forward_map_evals_Theta1["f_Theta"][0, 0] - nll_utils_Theta1["m_a"][0, 0])
#     #         / nll_utils_Theta1["s_a"][0, 0]
#     #     )
#     # )
#     # nll_Theta1_manual += (
#     #     L
#     #     * (N // 3)
#     #     * my_likelihood.lambda_[N // 2, 0]
#     #     * np.log(nll_utils_Theta1["s_a"][N // 2, 0])
#     # )  # add. : at exact point
#     # nll_Theta1_manual += (
#     #     L
#     #     * (N // 3)
#     #     * (1 - my_likelihood.lambda_)[-1, 0]
#     #     * (
#     #         (
#     #             np.log(nll_utils_Theta1["s_m"][-1, 0])
#     #             + 0.5
#     #             * (
#     #                 (
#     #                     my_likelihood.log_y[-1, 0]
#     #                     - forward_map_evals_Theta1["log_f_Theta"][-1, 0]
#     #                     - nll_utils_Theta1["m_m"][-1, 0]
#     #                 )
#     #                 / nll_utils_Theta1["s_m"][-1, 0]
#     #             )
#     #             ** 2
#     #         )
#     #         + np.log(nll_utils_Theta1["s_m"][-1, 0])
#     #     )
#     # )  # mult.

#     assert isinstance(nll_Theta1, float)
#     # print(nll_Theta1, nll_Theta1_manual)
#     # assert np.isclose(nll_Theta1, nll_Theta1_manual)

#     nll_Theta1 = my_likelihood.neglog_pdf(
#         forward_map_evals_Theta1, nll_utils_Theta1, pixelwise=True
#     )

#     # nll_Theta1_manual = np.zeros((N,))

#     # nll_Theta1_manual[: N // 3] = -statsnorm.logcdf(
#     #     (omega - 1 - nll_utils_Theta1["m_a"][0, 0]) / nll_utils_Theta1["s_a"][0, 0]
#     # )
#     # nll_Theta1_manual[N // 3 : -N // 3] = np.log(nll_utils_Theta1["s_a"][N // 2, 0])

#     # nll_Theta1_manual[-N // 3 :] = (1 - my_likelihood.lambda_)[-1, 0] * (
#     #     np.log(nll_utils_Theta1["s_m"][-1, 0])
#     #     + 0.5
#     #     * (
#     #         (
#     #             my_likelihood.log_y[-1, 0]
#     #             - forward_map_evals_Theta1["log_f_Theta"][-1, 0]
#     #             - nll_utils_Theta1["m_m"][-1, 0]
#     #         )
#     #         / nll_utils_Theta1["s_m"][-1, 0]
#     #     )
#     #     ** 2
#     # )

#     assert isinstance(nll_Theta1, np.ndarray)
#     assert nll_Theta1.shape == (N,)
#     # assert np.allclose(nll_Theta1, nll_Theta1_manual)

#     # nll_Theta2 = my_likelihood.neglog_pdf(
#     #     forward_map_evals_Theta2, nll_utils_Theta2, pixelwise=False
#     # )
#     # nll_Theta2_manual = -L * (N // 2) * statsnorm.logcdf((omega - np.e) / sigma)

#     # assert isinstance(nll_Theta2, float)
#     # assert np.isclose(nll_Theta2, nll_Theta2_manual)

#     # nll_Theta2 = my_likelihood.neglog_pdf(
#     #     forward_map_evals_Theta2, nll_utils_Theta2, pixelwise=True
#     # )
#     # nll_Theta2_manual = np.zeros((N,))
#     # nll_Theta2_manual[: N // 2] = -L * statsnorm.logcdf((omega - np.e) / sigma)
#     # assert isinstance(nll_Theta2, np.ndarray) and nll_Theta2.shape == (N,)
#     # assert np.allclose(nll_Theta2, nll_Theta2_manual)

#     # comparison of x1 and x2
#     nll_Theta1 = my_likelihood.neglog_pdf(
#         forward_map_evals_Theta1, nll_utils_Theta1, pixelwise=False
#     )
#     nll_Theta2 = my_likelihood.neglog_pdf(
#         forward_map_evals_Theta2, nll_utils_Theta2, pixelwise=False
#     )
#     assert nll_Theta1 < nll_Theta2


# def test_gradient_neglog_pdf(
#     settings,
#     my_likelihood,
#     forward_map_evals_Theta1,
#     forward_map_evals_Theta2,
#     nll_utils_Theta1,
#     nll_utils_Theta2,
# ):
#     D, L, N = settings

#     # from constant np.ndarray to float
#     omega = my_likelihood.omega.mean()

#     grad_nll_Theta1 = my_likelihood.gradient_neglog_pdf(forward_map_evals_Theta1, nll_utils_Theta1)

#     # grad_nll_Theta1_manual = np.zeros((N, D))
#     # grad_nll_Theta1_manual[: N // 3] = (
#     #     1
#     #     / nll_utils_Theta1["s_a"][0, 0]
#     #     * 1  # forward_map_evals_Theta1["grad_f_Theta"]
#     #     * utils.norm_pdf_cdf_ratio(
#     #         (omega - 1 - nll_utils_Theta1["m_a"][0, 0]) / nll_utils_Theta1["s_a"][0, 0]
#     #     )
#     # )
#     # grad_nll_Theta1_manual[-N // 3 :] = (
#     #     1
#     #     / nll_utils_Theta1["s_m"][0, 0] ** 2
#     #     * forward_map_evals_Theta1["grad_log_f_Theta"][0, 0]
#     #     * (
#     #         (
#     #             my_likelihood.log_y[0, 0]
#     #             - forward_map_evals_Theta1["log_f_Theta"][0, 0]
#     #             - nll_utils_Theta1["m_m"][0, 0]
#     #         )
#     #         / nll_utils_Theta1["s_m"][0, 0]
#     #     )
#     #     ** 2
#     # )

#     assert isinstance(grad_nll_Theta1, np.ndarray)
#     assert grad_nll_Theta1.shape == (N, D), grad_nll_Theta1
#     # assert np.allclose(grad_nll_Theta1, grad_nll_Theta1_manual)

#     # grad_nll_Theta2 = my_likelihood.gradient_neglog_pdf(forward_map_evals_Theta2, nll_utils_Theta2)
#     # grad_nll_Theta2_manual = np.zeros((N, D))
#     # grad_nll_Theta2_manual[: N // 2] = (
#     #     1
#     #     / sigma
#     #     * np.e  # forward_map_evals_Theta1["grad_f_Theta"]
#     #     * utils.norm_pdf_cdf_ratio((omega - np.e) / sigma)
#     # )

#     # assert isinstance(grad_nll_Theta2, np.ndarray) and grad_nll_Theta2.shape == (N, D)
#     # assert np.allclose(grad_nll_Theta2, grad_nll_Theta2_manual)


# def test_hess_diag_neglog_pdf(
#     settings,
#     my_likelihood,
#     forward_map_evals_Theta1,
#     forward_map_evals_Theta2,
#     nll_utils_Theta1,
#     nll_utils_Theta2,
# ):
#     D, L, N = settings

#     # from constant np.ndarray to float
#     omega = my_likelihood.omega.mean()

#     hess_diag_nll_Theta1 = my_likelihood.hess_diag_neglog_pdf(
#         forward_map_evals_Theta1, nll_utils_Theta1
#     )
#     # hess_diag_nll_Theta1_manual = np.zeros((N, D))
#     # hess_diag_nll_Theta1_manual[: N // 2] = (
#     #     1
#     #     / sigma
#     #     * utils.norm_pdf_cdf_ratio((omega - 1) / sigma)
#     #     * (
#     #         1
#     #         + 1
#     #         / sigma
#     #         * 1 ** 2
#     #         * (((omega - 1) / sigma) + utils.norm_pdf_cdf_ratio((omega - 1) / sigma))
#     #     )
#     # )
#     # hess_diag_nll_Theta1_manual[-N // 2 :] = 1 / sigma ** 2 * np.e ** 2
#     assert isinstance(hess_diag_nll_Theta1, np.ndarray)
#     assert hess_diag_nll_Theta1.shape == (N, D)
#     # assert np.allclose(hess_diag_nll_Theta1, hess_diag_nll_Theta1_manual)

#     # hess_diag_nll_Theta2 = my_likelihood.hess_diag_neglog_pdf(
#     #     forward_map_evals_Theta2, nll_utils_Theta2
#     # )

#     # hess_diag_nll_Theta2_manual = np.zeros((N, D))
#     # hess_diag_nll_Theta2_manual[: N // 2] = (
#     #     1
#     #     / sigma
#     #     * utils.norm_pdf_cdf_ratio((omega - np.e) / sigma)
#     #     * (
#     #         np.e
#     #         + 1
#     #         / sigma
#     #         * np.e ** 2
#     #         * (
#     #             ((omega - np.e) / sigma)
#     #             + utils.norm_pdf_cdf_ratio((omega - np.e) / sigma)
#     #         )
#     #     )
#     # )
#     # hess_diag_nll_Theta2_manual[-N // 2 :] = 1 / sigma ** 2 * np.e ** 2

#     # assert isinstance(hess_diag_nll_Theta2, np.ndarray) and hess_diag_nll_Theta2.shape == (N, D)
#     # assert np.allclose(hess_diag_nll_Theta2, hess_diag_nll_Theta2_manual)


# def test_neglog_pdf_candidates(settings, my_likelihood):
#     D, L, N = settings
#     N_candidates = 3 * N
#     candidates = np.linspace(-1, 10, N_candidates).reshape((N_candidates, D))

#     nll_candidates = my_likelihood.neglog_pdf_candidates(candidates, N - 1)

#     assert isinstance(nll_candidates, np.ndarray) and nll_candidates.shape == (
#         N_candidates,
#     )
#     # considering that the true value is 1, the best value will be the closest one, ie the last one
#     print(nll_candidates)
#     assert np.argmin(nll_candidates) == N_candidates - 1


# @pytest.fixture(scope="module")
# def transition_params_tests_lambda():
#     transition_loc = 2
#     k = 3
#     return transition_loc, k


# @pytest.fixture(scope="module")
# def nll_utils_tests_lambda(settings, transition_params_tests_lambda):
#     D, L, N = settings
#     forward_map = BasicExpForwardMap(D, L, N)

#     transition_loc, k = transition_params_tests_lambda

#     sigma_a = 2
#     omega = 0
#     sigma_m = np.log(1.1)

#     y = np.ones((N, L))  # whatever value, not used in this test

#     my_likelihood = MixingModelsLikelihood(
#         forward_map, D, L, N, y, sigma_a, sigma_m, omega, transition_loc, k
#     )

#     var_eps_m = np.exp(sigma_m**2) * (np.exp(sigma_m**2) - 1)
#     x_f0 = np.log(transition_loc * (sigma_a / np.sqrt(var_eps_m))) * np.ones((N, D))
#     forward_map_evals = my_likelihood.evaluate_all_forward_map(x_f0, True)
#     nll_utils_f0 = my_likelihood.evaluate_all_nll_utils(forward_map_evals)

#     x_fm1 = np.log(k * sigma_a) * np.ones((N, D))
#     forward_map_evals = my_likelihood.evaluate_all_forward_map(x_fm1, True)
#     nll_utils_fm1 = my_likelihood.evaluate_all_nll_utils(forward_map_evals)

#     x_fp1 = 2 * x_f0 - x_fm1
#     forward_map_evals = my_likelihood.evaluate_all_forward_map(x_fp1, True)
#     nll_utils_fp1 = my_likelihood.evaluate_all_nll_utils(forward_map_evals)
#     return nll_utils_f0, nll_utils_fm1, nll_utils_fp1


# def test_model_mixing_param(settings, nll_utils_tests_lambda):
#     D, L, N = settings
#     nll_utils_f0, nll_utils_fm1, nll_utils_fp1 = nll_utils_tests_lambda

#     assert nll_utils_f0["lambda_"].shape == (N, L)

#     assert np.allclose(nll_utils_f0["lambda_"], 0.5 * np.ones((N, L)))
#     assert np.allclose(nll_utils_fm1["lambda_"], np.ones((N, L)))
#     assert np.allclose(nll_utils_fp1["lambda_"], np.zeros((N, L)))


# def test_grad_model_mixing_param(
#     settings, transition_params_tests_lambda, nll_utils_tests_lambda
# ):
#     D, L, N = settings
#     transition_loc, k = transition_params_tests_lambda
#     _, nll_utils_fm1, nll_utils_fp1 = nll_utils_tests_lambda

#     assert nll_utils_fm1["grad_lambda_"].shape == (N, D, L)

#     assert np.allclose(nll_utils_fm1["grad_lambda_"], np.zeros((N, D, L)))
#     assert np.allclose(nll_utils_fp1["grad_lambda_"], np.zeros((N, D, L)))


# def test_hess_diag_model_mixing_param(settings, nll_utils_tests_lambda):
#     D, L, N = settings
#     # transition_loc, k = transition_params_tests_lambda
#     nll_utils_f0, nll_utils_fm1, nll_utils_fp1 = nll_utils_tests_lambda

#     assert nll_utils_f0["grad_lambda_"].shape == (N, D, L)

#     assert np.allclose(nll_utils_f0["hess_diag_lambda_"], np.zeros((N, D, L)))
#     assert np.allclose(nll_utils_fm1["hess_diag_lambda_"], np.zeros((N, D, L)))
#     assert np.allclose(nll_utils_fp1["hess_diag_lambda_"], np.zeros((N, D, L)))
