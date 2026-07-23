import pytest
from scipy.stats import norm as statsnorm

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed

from beetroots.modelling.forward_maps.basic_exp import BasicExpForwardMap
from beetroots.modelling.hierarchical.aux_var_u import AuxiliaryGivenTarget
from beetroots.modelling.likelihoods import utils

# --- AUXILIARY GIVEN TARGET ---


@pytest.fixture(scope="module")
def settings():
    D = 3
    L = 5
    N = 64
    return D, L, N


@pytest.fixture(scope="module")
def points_aux_given_target(settings):
    D, L, N = settings

    target1 = xp.zeros((N, D))
    target1[-N // 2 :] += 1

    target2 = xp.ones((N, D))
    return target1, target2


@pytest.fixture(scope="module")
def my_aux_distribution(settings, points_aux_given_target):
    D, L, N = settings
    forward_map = BasicExpForwardMap(D, L, N)

    sigma_m = 0.1
    differentiate_auxiliary = True

    my_aux_distribution = AuxiliaryGivenTarget(
        forward_map,
        D,
        L,
        N,
        differentiate_auxiliary,
        sigma_m,
    )
    return my_aux_distribution


@pytest.fixture(scope="module")
def forward_map_evals_target1(my_aux_distribution, points_aux_given_target):
    target1, _ = points_aux_given_target
    forward_map_evals_target1 = my_aux_distribution.evaluate_all_forward_map(
        target1, True
    )
    return forward_map_evals_target1


@pytest.fixture(scope="module")
def forward_map_evals_target2(my_aux_distribution, points_aux_given_target):
    _, target2 = points_aux_given_target
    forward_map_evals_target2 = my_aux_distribution.evaluate_all_forward_map(
        target2, True
    )
    return forward_map_evals_target2


@pytest.fixture(scope="module")
def nll_utils_target1(my_aux_distribution, forward_map_evals_target1):
    nll_utils_target1 = my_aux_distribution.evaluate_all_nll_utils(
        forward_map_evals_target1
    )
    return nll_utils_target1


@pytest.fixture(scope="module")
def nll_utils_target2(my_aux_distribution, forward_map_evals_target2):
    nll_utils_target2 = my_aux_distribution.evaluate_all_nll_utils(
        forward_map_evals_target2
    )
    return nll_utils_target2


def test_init(settings, my_aux_distribution):
    D, L, N = settings
    assert my_aux_distribution.sigma_m.shape == (N, L)


def test_evaluate_all_forward_map(
    settings, my_aux_distribution, points_aux_given_target
):
    D, L, N = settings
    target1, target2 = points_aux_given_target

    forward_map_evals = my_aux_distribution.evaluate_all_forward_map(target1, True)

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


def test_evaluate_all_nll_utils(
    settings,
    my_aux_distribution,
    points_aux_given_target,
    forward_map_evals_target1,
):
    nll_utils = my_aux_distribution.evaluate_all_nll_utils(forward_map_evals_target1)
    assert nll_utils == dict()

    nll_utils = my_aux_distribution.evaluate_all_nll_utils(
        forward_map_evals_target1, idx=0
    )  # as if x1 was a vector of N candidates for pixel n=0
    assert nll_utils == dict()


def test_neglog_pdf_aux_given_target(
    settings, my_aux_distribution, forward_map_evals_target1, nll_utils_target1
):
    D, L, N = settings
    nlpdf_utils = nll_utils_target1

    sigma_a, sigma_m, omega = (
        my_aux_distribution.sigma_a,
        my_aux_distribution.sigma_m,
        my_aux_distribution.omega,
    )
    neglog_function = my_aux_distribution.neglog_pdf_u(y, nlpdf_utils)
    # The x1 is censored on its first half and uncensored on its second half

    neglog_manual = -L * N // 2 * statsnorm.logcdf()

    expected_shape = (my_aux_distribution.N, my_aux_distribution.L)
    assert result.shape == expected_shape


# --- OBSERVATIONS GIVEN AUXILIARY ---


@pytest.fixture(scope="module")
def points_obs_given_target(settings):
    D, L, N = settings

    x1 = xp.zeros((N, D))
    x1[-N // 2 :] += 1

    x2 = xp.ones((N, D))
    return x1, x2


def test_neglog_pdf_obs_given_aux(
    settings, my_aux_distribution, forward_map_evals_Theta1, nll_utils_Theta1
):
    D, L, N = settings
    y = my_aux_distribution.y
    nlpdf_utils = nll_utils_Theta1

    sigma_a, sigma_m, omega = (
        my_aux_distribution.sigma_a,
        my_aux_distribution.sigma_m,
        my_aux_distribution.omega,
    )
    neglog_function = my_aux_distribution.neglog_pdf_u(y, nlpdf_utils)
    # The x1 is censored on its first half and uncensored on its second half

    neglog_manual = -L * N // 2 * statsnorm.logcdf()

    expected_shape = (my_aux_distribution.N, my_aux_distribution.L)
    assert result.shape == expected_shape
