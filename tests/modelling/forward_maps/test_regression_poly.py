import numpy as np
import pandas as pd
import pytest

from beetroots.modelling.forward_maps.regression_poly import PolynomialApprox

# tolerance params
rtol = 1e-5
atol = 1e-8

D = 3
D_no_kappa = 2
L = 4
deg = 3


##% manual functions
def f0(Theta_0, Theta_scaled, x2):
    return Theta_0 + 2 * Theta_scaled**2 + x2**3


def f1(Theta_0, Theta_scaled, x2):
    return Theta_0 - Theta_scaled**2 + x2**2


def f2(Theta_0, Theta_scaled, x2):
    return Theta_0 + Theta_scaled - 3 * x2**2


def f3(Theta_0, Theta_scaled, x2):
    return Theta_0 + Theta_scaled + x2


def f(Theta):
    y = np.zeros((Theta.shape[0], L))
    y[:, 0] = f0(Theta[:, 0], Theta[:, 1], Theta[:, 2])
    y[:, 1] = f1(Theta[:, 0], Theta[:, 1], Theta[:, 2])
    y[:, 2] = f2(Theta[:, 0], Theta[:, 1], Theta[:, 2])
    y[:, 3] = f3(Theta[:, 0], Theta[:, 1], Theta[:, 2])
    y = np.exp(y)
    return y  # (N, L)


def grad_log_f(Theta):
    grad_ = np.zeros((Theta.shape[0], D, L))
    for i, x in enumerate(Theta):
        grad_[i] = np.array(
            [
                [1, 4 * Theta[1], 3 * Theta[2] ** 2],
                [1, -2 * Theta[1], 2 * Theta[2]],
                [1, 1, -6 * Theta[2]],
                [1, 1, 1],
            ]
        ).T
    return grad_  # (N, D, L)


def grad_f(Theta):
    grad_ = grad_log_f(Theta) * f(Theta)[:, None, :]
    return grad_  # (N, D, L)


def hess_diag_log_f(Theta):
    hess_diag = np.zeros((Theta.shape[0], D, L))
    for i, x in enumerate(Theta):
        hess_diag[i] = np.array(
            [
                [0.0, 4.0, 6 * Theta[2]],
                [0.0, -2, 2.0],
                [0.0, 0.0, -6.0],
                [0.0, 0.0, 0.0],
            ]
        ).T
    return hess_diag  # (N, D, L)


def hess_diag_f(Theta):
    hess_diag = (hess_diag_log_f(Theta) + grad_log_f(Theta) ** 2) * f(Theta)[:, None, :]
    return hess_diag  # (N, D, L)


# def hess_full_log_f(Theta):
#     hess_ = np.zeros((Theta.shape[0], D, D, L))
#     for i, x in enumerate(Theta):
#         hess_[i] = np.array(
#             [
#                 [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6 * Theta[2]]],
#                 [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 2.0]],
#                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -6.0]],
#                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#             ]
#         ).T
#     return hess_  # (N, D, D, L)


# def hess_full_f(Theta):
#     hess_ = (
#         hess_full_log_f(Theta) + grad_log_f(Theta)[:, None, :, :] * grad_log_f(Theta)[:, :, None, :]
#     ) * f(Theta)[:, None, None, :]
#     return hess_  # (N, D, D, L)


# * tests
# @pytest.fixture(scope="module")
# def pdr_code():
#     list_dicts = [
#         {"Theta_0": 0, "Theta_scaled": i / 100, "x2": j / 100}
#         for i in range(-100, 100)
#         for j in range(-100, 100)
#     ]
#     grid_reg = pd.DataFrame(list_dicts)
#     grid_reg["y0"] = grid_reg.apply(
#         lambda x: np.exp(f0(x["Theta_0"], Theta["Theta_scaled"], Theta["x2"])), axis=1
#     )
#     grid_reg["y1"] = grid_reg.apply(
#         lambda x: np.exp(f1(x["Theta_0"], Theta["Theta_scaled"], Theta["x2"])), axis=1
#     )
#     grid_reg["y2"] = grid_reg.apply(
#         lambda x: np.exp(f2(x["Theta_0"], Theta["Theta_scaled"], Theta["x2"])), axis=1
#     )
#     grid_reg["y3"] = grid_reg.apply(
#         lambda x: np.exp(f3(x["Theta_0"], Theta["Theta_scaled"], Theta["x2"])), axis=1
#     )

#     pdr_code = PolynomialApprox(grid_reg, D, D_no_kappa, L, deg)
#     return pdr_code


# # test data
# @pytest.fixture(scope="module")
# def test_data():
#     Theta_scaled = np.array(
#         [
#             [0, -1 / 2, 1 / 2],
#             [0, 0, 0],
#             [1, 0, 0],
#             [0, 1 / 2, -1 / 2],
#             [0, -3 / 4, 5 / 4],
#             [0, 5 / 4, 5 / 4],
#         ]
#     )
#     N_test = Theta_scaled.shape[0]

#     y = f(Theta_scaled)
#     log_y = np.log(y)
#     return Theta_scaled, N_test, y, log_y


# def test_evaluate(pdr_code, test_data):
#     Theta_scaled, N_test, y, log_y = test_data

#     y_pred = pdr_code.evaluate(Theta_scaled)
#     assert y_pred.shape == (N_test, L)
#     assert np.allclose(y, y_pred, rtol=rtol, atol=atol)


# def test_log_call(pdr_code, test_data):
#     Theta_scaled, N_test, y, log_y = test_data

#     log_y_pred = pdr_code.evaluate_log(Theta_scaled)
#     y_pred = pdr_code.evaluate(Theta_scaled)
#     print(log_y)
#     print(log_y_pred)
#     print(np.log(y_pred))

#     assert log_y_pred.shape == (N_test, L)
#     assert np.allclose(log_y, log_y_pred, rtol=rtol, atol=atol)
#     assert np.allclose(log_y, np.log(y_pred), rtol=rtol, atol=atol)


# def test_gradient(pdr_code, test_data):
#     Theta_scaled, N_test, y, log_y = test_data

#     grad_ = pdr_code.gradient(Theta_scaled)
#     grad_manual = grad_f(Theta_scaled)
#     print("code")
#     print(grad_)
#     print("manual")
#     print(grad_manual)
#     assert grad_.shape == (N_test, D, L)
#     assert np.allclose(grad_, grad_manual, rtol=rtol, atol=atol)


# def test_gradient_log(pdr_code, test_data):
#     Theta_scaled, N_test, y, log_y = test_data

#     grad_log = pdr_code.gradient_log(Theta_scaled)
#     grad_log_manual = grad_log_f(Theta_scaled)
#     print("code")
#     print(grad_log)
#     print("manual")
#     print(grad_log_manual)
#     assert grad_log.shape == (N_test, D, L)
#     assert np.allclose(grad_log, grad_log_manual, rtol=rtol, atol=atol)


# def test_hess_diag(pdr_code, test_data):
#     Theta_scaled, N_test, y, log_y = test_data

#     hess_diag = pdr_code.hess_diag(Theta_scaled)
#     hess_diag_manual = hess_diag_f(Theta_scaled)

#     print(hess_diag)
#     print(hess_diag_manual)
#     assert hess_diag.shape == (N_test, D, L)
#     assert np.allclose(hess_diag, hess_diag_manual, rtol=rtol, atol=atol)


# def test_hess_diag_log(pdr_code, test_data):
#     Theta_scaled, N_test, y, log_y = test_data

#     hess_diag = pdr_code.hess_diag_log(Theta_scaled)
#     hess_diag_manual = hess_diag_log_f(Theta_scaled)
#     assert hess_diag.shape == (N_test, D, L)
#     assert np.allclose(hess_diag, hess_diag_manual, rtol=rtol, atol=atol)


# # def test_hess_full(pdr_code, test_data):
# #     Theta_scaled, N_test, y, log_y = test_data

# #     hess_full = pdr_code.hess_full(Theta_scaled)
# #     hess_full_manual = hess_full_f(Theta_scaled)
# #     assert hess_full.shape == (N_test, D, D, L)
# #     assert np.allclose(hess_full, hess_full_manual, rtol=rtol, atol=atol)


# # def test_hess_full_log(pdr_code, test_data):
# #     Theta_scaled, N_test, y, log_y = test_data

# #     hess_full = pdr_code.hess_full_log(Theta_scaled)
# #     hess_full_manual = hess_full_log_f(Theta_scaled)
# #     print("hess full")
# #     print(hess_full)
# #     print("hess full manual")
# #     print(hess_full_manual)
# #     print("end")

# #     assert hess_full.shape == (N_test, D, D, L)
# #     assert np.allclose(hess_full, hess_full_manual, rtol=rtol, atol=atol)


# def test_compute_all(pdr_code, test_data):
#     Theta_scaled, N_test, y, log_y = test_data

#     # test that we have the right elements
#     forward_map_evals = pdr_code.compute_all(Theta_scaled, True, True, True)
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

#     forward_map_evals = pdr_code.compute_all(Theta_scaled, False, True, True)
#     list_keys = list(forward_map_evals.keys())
#     list_keys_manual = [
#         "log_f_Theta",
#         "grad_log_f_Theta",
#         "hess_diag_log_f_Theta",
#     ]
#     assert sorted(list_keys) == sorted(list_keys_manual)

#     forward_map_evals = pdr_code.compute_all(Theta_scaled, True, False, True)
#     list_keys = list(forward_map_evals.keys())
#     list_keys_manual = [
#         "f_Theta",
#         "grad_f_Theta",
#         "hess_diag_f_Theta",
#     ]
#     assert sorted(list_keys) == sorted(list_keys_manual)

#     forward_map_evals = pdr_code.compute_all(Theta_scaled, True, True, False)
#     list_keys = list(forward_map_evals.keys())
#     list_keys_manual = [
#         "f_Theta",
#         "log_f_Theta",
#     ]
#     assert sorted(list_keys) == sorted(list_keys_manual)

#     forward_map_evals = pdr_code.compute_all(Theta_scaled, True, False, False)
#     list_keys = list(forward_map_evals.keys())
#     list_keys_manual = [
#         "f_Theta",
#     ]
#     assert sorted(list_keys) == sorted(list_keys_manual)

#     forward_map_evals = pdr_code.compute_all(Theta_scaled, False, True, False)
#     list_keys = list(forward_map_evals.keys())
#     list_keys_manual = [
#         "log_f_Theta",
#     ]
#     assert sorted(list_keys) == sorted(list_keys_manual)

#     forward_map_evals = pdr_code.compute_all(Theta_scaled, True, True, True)

#     # test values
#     f_Theta_1 = pdr_code.evaluate(Theta_scaled)
#     log_f_Theta_1 = pdr_code.evaluate_log(Theta_scaled)

#     grad_f_Theta_1 = pdr_code.gradient(Theta_scaled)
#     grad_log_f_Theta_1 = pdr_code.gradient_log(Theta_scaled)

#     hess_diag_f_Theta_1 = pdr_code.hess_diag(Theta_scaled)
#     hess_diag_log_f_Theta_1 = pdr_code.hess_diag_log(Theta_scaled)

#     assert np.allclose(f_Theta_1, forward_map_evals["f_Theta"])
#     assert np.allclose(log_f_Theta_1, forward_map_evals["log_f_Theta"])

#     assert np.allclose(grad_f_Theta_1, forward_map_evals["grad_f_Theta"])
#     assert np.allclose(grad_log_f_Theta_1, forward_map_evals["grad_log_f_Theta"])

#     assert np.allclose(hess_diag_f_Theta_1, forward_map_evals["hess_diag_f_Theta"])
#     assert np.allclose(hess_diag_log_f_Theta_1, forward_map_evals["hess_diag_log_f_Theta"])
