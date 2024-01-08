import numpy as np
import pytest

from beetroots.modelling.forward_maps.identity import BasicForwardMap


@pytest.fixture()
def settings():
    D = 4
    N = 16
    return D, N


@pytest.fixture
def points(settings):
    D, N = settings
    Theta_1 = np.zeros((N, D))
    Theta_2 = np.ones((N, D))
    return Theta_1, Theta_2


@pytest.fixture
def my_basic_map(settings):
    D, N = settings
    return BasicForwardMap(D, N)


def test_init():
    with pytest.raises(Exception) as e_info:
        assert BasicForwardMap(4, 4, 16)


def test_evaluate(settings, my_basic_map, points):
    D, N = settings
    Theta_1, Theta_2 = points

    f_Theta_1 = my_basic_map.evaluate(Theta_1)
    assert np.allclose(f_Theta_1, Theta_1)

    f_Theta_2 = my_basic_map.evaluate(Theta_2)
    assert np.allclose(f_Theta_2, Theta_2)


def test_gradient(settings, my_basic_map, points):
    D, N = settings
    Theta_1, Theta_2 = points

    grad_f_Theta_1 = my_basic_map.gradient(Theta_1)
    assert np.allclose(grad_f_Theta_1, np.ones((N, D, D)))

    grad_f_Theta_2 = my_basic_map.gradient(Theta_2)
    assert np.allclose(grad_f_Theta_2, np.ones((N, D, D)))


def test_hess_diag(settings, my_basic_map, points):
    D, N = settings
    Theta_1, Theta_2 = points

    grad_f_Theta_1 = my_basic_map.hess_diag(Theta_1)
    assert np.allclose(grad_f_Theta_1, np.zeros((N, D, D)))

    grad_f_Theta_2 = my_basic_map.hess_diag(Theta_2)
    assert np.allclose(grad_f_Theta_2, np.zeros((N, D, D)))


def test_compute_all(settings, my_basic_map, points):
    D, N = settings
    Theta_1, _ = points

    # test that we have the right elements
    forward_map_evals = my_basic_map.compute_all(Theta_1, True, False, True)
    list_keys = list(forward_map_evals.keys())
    list_keys_manual = [
        "f_Theta",
        "grad_f_Theta",
        "hess_diag_f_Theta",
    ]
    assert sorted(list_keys) == sorted(list_keys_manual)

    assert sorted(list_keys) == sorted(list_keys_manual)

    forward_map_evals = my_basic_map.compute_all(Theta_1, True, False, False)
    list_keys = list(forward_map_evals.keys())
    list_keys_manual = [
        "f_Theta",
    ]
    assert sorted(list_keys) == sorted(list_keys_manual)

    # test values
    forward_map_evals = my_basic_map.compute_all(Theta_1, True, True, True)

    f_Theta_1 = my_basic_map.evaluate(Theta_1)
    grad_f_Theta_1 = my_basic_map.gradient(Theta_1)
    hess_diag_f_Theta_1 = my_basic_map.hess_diag(Theta_1)

    assert np.allclose(f_Theta_1, forward_map_evals["f_Theta"])
    assert np.allclose(grad_f_Theta_1, forward_map_evals["grad_f_Theta"])
    assert np.allclose(hess_diag_f_Theta_1, forward_map_evals["hess_diag_f_Theta"])
