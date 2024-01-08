import numpy as np
import pytest

from beetroots.modelling.forward_maps.basic_exp import BasicExpForwardMap


@pytest.fixture()
def settings():
    D = 4
    L = 4
    N = 16
    return D, L, N


@pytest.fixture
def points(settings):
    D, L, N = settings
    x1 = np.zeros((N, D))
    x2 = np.ones((N, D))
    return x1, x2


@pytest.fixture
def my_basic_map(settings):
    D, L, N = settings
    return BasicExpForwardMap(D, L, N)


def test_init():
    with pytest.raises(Exception) as e_info:
        assert BasicExpForwardMap(4, 1, 16)


def test_evaluate(settings, my_basic_map, points):
    D, L, N = settings
    x1, x2 = points

    f_Theta_1 = my_basic_map.evaluate(x1)
    assert np.allclose(f_Theta_1, np.ones((N, L)))

    f_Theta_2 = my_basic_map.evaluate(x2)
    assert np.allclose(f_Theta_2, np.e * np.ones((N, L)))


def test_evaluate_log(settings, my_basic_map, points):
    D, L, N = settings
    x1, x2 = points

    f_Theta_1 = my_basic_map.evaluate_log(x1)
    assert np.allclose(f_Theta_1, np.zeros((N, L)))

    f_Theta_2 = my_basic_map.evaluate_log(x2)
    assert np.allclose(f_Theta_2, np.ones((N, L)))


def test_gradient(settings, my_basic_map, points):
    D, L, N = settings
    x1, x2 = points

    grad_f_Theta_1 = my_basic_map.gradient(x1)
    assert np.allclose(grad_f_Theta_1, np.ones((N, D, L)))

    grad_f_Theta_2 = my_basic_map.gradient(x2)
    assert np.allclose(
        grad_f_Theta_2, my_basic_map.evaluate(x2)[:, None, :] * np.ones((N, D, L))
    )


def test_gradient_log(settings, my_basic_map, points):
    D, L, N = settings
    x1, x2 = points

    grad_f_Theta_1 = my_basic_map.gradient_log(x1)
    assert np.allclose(grad_f_Theta_1, np.ones((N, D, L)))

    grad_f_Theta_2 = my_basic_map.gradient_log(x2)
    assert np.allclose(grad_f_Theta_2, np.ones((N, D, L)))


def test_hess_diag(settings, my_basic_map, points):
    D, L, N = settings
    x1, x2 = points

    hess_diag_f_Theta_1 = my_basic_map.hess_diag(x1)
    assert np.allclose(hess_diag_f_Theta_1, np.ones((N, D, L)))

    hess_diag_f_Theta_2 = my_basic_map.hess_diag(x2)
    assert np.allclose(
        hess_diag_f_Theta_2, my_basic_map.evaluate(x2)[:, None, :] * np.ones((N, D, L))
    )


def test_hess_diag_log(settings, my_basic_map, points):
    D, L, N = settings
    x1, x2 = points

    hess_diag_f_Theta_1 = my_basic_map.hess_diag_log(x1)
    assert np.allclose(hess_diag_f_Theta_1, np.zeros((N, D, L)))

    hess_diag_f_Theta_2 = my_basic_map.hess_diag_log(x2)
    assert np.allclose(hess_diag_f_Theta_2, np.zeros((N, D, L)))


def test_compute_all(settings, my_basic_map, points):
    D, L, N = settings
    x1, _ = points

    # test that we have the right elements
    forward_map_evals = my_basic_map.compute_all(x1, True, True, True)
    list_keys = list(forward_map_evals.keys())
    list_keys_manual = [
        "f_Theta",
        "grad_f_Theta",
        "hess_diag_f_Theta",
        "log_f_Theta",
        "grad_log_f_Theta",
        "hess_diag_log_f_Theta",
    ]
    assert sorted(list_keys) == sorted(list_keys_manual)

    forward_map_evals = my_basic_map.compute_all(x1, False, True, True)
    list_keys = list(forward_map_evals.keys())
    list_keys_manual = [
        "log_f_Theta",
        "grad_log_f_Theta",
        "hess_diag_log_f_Theta",
    ]
    assert sorted(list_keys) == sorted(list_keys_manual)

    forward_map_evals = my_basic_map.compute_all(x1, True, False, True)
    list_keys = list(forward_map_evals.keys())
    list_keys_manual = [
        "f_Theta",
        "grad_f_Theta",
        "hess_diag_f_Theta",
    ]
    assert sorted(list_keys) == sorted(list_keys_manual)

    forward_map_evals = my_basic_map.compute_all(x1, True, True, False)
    list_keys = list(forward_map_evals.keys())
    list_keys_manual = [
        "f_Theta",
        "log_f_Theta",
    ]
    assert sorted(list_keys) == sorted(list_keys_manual)

    forward_map_evals = my_basic_map.compute_all(x1, True, False, False)
    list_keys = list(forward_map_evals.keys())
    list_keys_manual = [
        "f_Theta",
    ]
    assert sorted(list_keys) == sorted(list_keys_manual)

    forward_map_evals = my_basic_map.compute_all(x1, False, True, False)
    list_keys = list(forward_map_evals.keys())
    list_keys_manual = [
        "log_f_Theta",
    ]
    assert sorted(list_keys) == sorted(list_keys_manual)

    forward_map_evals = my_basic_map.compute_all(x1, True, True, True)

    # test values
    forward_map_evals = my_basic_map.compute_all(x1, True, True, True)

    f_Theta_1 = my_basic_map.evaluate(x1)
    log_f_Theta_1 = my_basic_map.evaluate_log(x1)

    grad_f_Theta_1 = my_basic_map.gradient(x1)
    grad_log_f_Theta_1 = my_basic_map.gradient_log(x1)

    hess_diag_f_Theta_1 = my_basic_map.hess_diag(x1)
    hess_diag_log_f_Theta_1 = my_basic_map.hess_diag_log(x1)

    assert np.allclose(f_Theta_1, forward_map_evals["f_Theta"])
    assert np.allclose(log_f_Theta_1, forward_map_evals["log_f_Theta"])

    assert np.allclose(grad_f_Theta_1, forward_map_evals["grad_f_Theta"])
    assert np.allclose(grad_log_f_Theta_1, forward_map_evals["grad_log_f_Theta"])

    assert np.allclose(hess_diag_f_Theta_1, forward_map_evals["hess_diag_f_Theta"])
    assert np.allclose(
        hess_diag_log_f_Theta_1, forward_map_evals["hess_diag_log_f_Theta"]
    )
