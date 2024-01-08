import numpy as np
import pytest

from beetroots.modelling.forward_maps.basic_exp import BasicExpForwardMap
from beetroots.modelling.likelihoods.gaussian import GaussianLikelihood


@pytest.fixture(scope="module")
def settings():
    D = 1
    L = 1
    N = 16
    return D, L, N


@pytest.fixture(scope="module")
def points(settings):
    D, L, N = settings
    x1 = np.zeros((N, D))
    x2 = np.ones((N, D))
    return x1, x2


@pytest.fixture(scope="module")
def my_gaussian_likelihood(settings, points):
    D, L, N = settings
    _, x2 = points
    forward_map = BasicExpForwardMap(D, L, N)

    sigma = 1
    y = forward_map.evaluate(x2)

    my_gaussian_likelihood = GaussianLikelihood(forward_map, D, L, N, y, sigma)
    return my_gaussian_likelihood


@pytest.fixture(scope="module")
def forward_map_evals_Theta1(my_gaussian_likelihood, points):
    x1, _ = points
    forward_map_evals_Theta1 = my_gaussian_likelihood.evaluate_all_forward_map(x1, True)
    return forward_map_evals_Theta1


@pytest.fixture(scope="module")
def forward_map_evals_Theta2(my_gaussian_likelihood, points):
    _, x2 = points
    forward_map_evals_Theta2 = my_gaussian_likelihood.evaluate_all_forward_map(x2, True)
    return forward_map_evals_Theta2


@pytest.fixture(scope="module")
def nll_utils_Theta1(my_gaussian_likelihood, forward_map_evals_Theta1):
    nll_utils_Theta1 = my_gaussian_likelihood.evaluate_all_nll_utils(
        forward_map_evals_Theta1
    )
    return nll_utils_Theta1


@pytest.fixture(scope="module")
def nll_utils_Theta2(my_gaussian_likelihood, forward_map_evals_Theta2):
    nll_utils_Theta2 = my_gaussian_likelihood.evaluate_all_nll_utils(
        forward_map_evals_Theta2
    )
    return nll_utils_Theta2


def test_init(settings, my_gaussian_likelihood):
    D, L, N = settings
    assert my_gaussian_likelihood.sigma.shape == (N, L)


def test_evaluate_all_forward_map(settings, my_gaussian_likelihood, points):
    D, L, N = settings
    x1, x2 = points

    forward_map_evals = my_gaussian_likelihood.evaluate_all_forward_map(x1, True)
    list_keys = list(forward_map_evals.keys())
    list_keys_manual = ["f_Theta", "grad_f_Theta", "hess_diag_f_Theta"]
    assert sorted(list_keys) == sorted(list_keys_manual)

    forward_map_evals = my_gaussian_likelihood.evaluate_all_forward_map(x1, False)
    list_keys = list(forward_map_evals.keys())
    list_keys_manual = ["f_Theta"]
    assert sorted(list_keys) == sorted(list_keys_manual)


def test_evaluate_all_nll_utils(
    settings,
    my_gaussian_likelihood,
    points,
    forward_map_evals_Theta1,
):
    nll_utils = my_gaussian_likelihood.evaluate_all_nll_utils(forward_map_evals_Theta1)
    assert nll_utils == dict()

    nll_utils = my_gaussian_likelihood.evaluate_all_nll_utils(
        forward_map_evals_Theta1, idx=0
    )  # as if x1 was a vector of N candidates for pixel n=0
    assert nll_utils == dict()


def test_neglog_pdf(
    settings,
    my_gaussian_likelihood,
    forward_map_evals_Theta1,
    forward_map_evals_Theta2,
    nll_utils_Theta1,
    nll_utils_Theta2,
):
    D, L, N = settings

    # from constant np.ndarray to float
    sigma = my_gaussian_likelihood.sigma.mean()

    nll_Theta1 = my_gaussian_likelihood.neglog_pdf(
        forward_map_evals_Theta1, nll_utils_Theta1, pixelwise=False
    )
    nll_Theta1_manual = N * L * (np.e - 1) ** 2 / (2 * sigma**2)
    assert isinstance(nll_Theta1, float)
    assert np.isclose(nll_Theta1, nll_Theta1_manual)

    nll_Theta1 = my_gaussian_likelihood.neglog_pdf(
        forward_map_evals_Theta1, nll_utils_Theta1, pixelwise=True
    )
    nll_Theta1_manual = L * (np.e - 1) ** 2 / (2 * sigma**2) * np.ones((N,))
    assert isinstance(nll_Theta1, np.ndarray) and nll_Theta1.shape == (N,)
    assert np.allclose(nll_Theta1, nll_Theta1_manual)

    nll_Theta2 = my_gaussian_likelihood.neglog_pdf(
        forward_map_evals_Theta2, nll_utils_Theta2, pixelwise=False
    )
    nll_Theta2_manual = 0
    assert isinstance(nll_Theta2, float)
    assert np.isclose(nll_Theta2, nll_Theta2_manual)

    nll_Theta2 = my_gaussian_likelihood.neglog_pdf(
        forward_map_evals_Theta2, nll_utils_Theta2, pixelwise=True
    )
    nll_Theta2_manual = np.zeros((N,))
    assert isinstance(nll_Theta2, np.ndarray) and nll_Theta2.shape == (N,)
    assert np.allclose(nll_Theta2, nll_Theta2_manual)


def test_gradient_neglog_pdf(
    settings,
    my_gaussian_likelihood,
    forward_map_evals_Theta1,
    forward_map_evals_Theta2,
    nll_utils_Theta1,
    nll_utils_Theta2,
):
    D, L, N = settings

    # from constant np.ndarray to float
    sigma = my_gaussian_likelihood.sigma.mean()

    grad_nll_Theta1 = my_gaussian_likelihood.gradient_neglog_pdf(
        forward_map_evals_Theta1, nll_utils_Theta1
    )
    grad_nll_Theta1_manual = 1 * (1 - np.e) / sigma**2 * np.ones((N, D, L))
    assert isinstance(grad_nll_Theta1, np.ndarray) and grad_nll_Theta1.shape == (
        N,
        D,
        L,
    )
    assert np.allclose(grad_nll_Theta1, grad_nll_Theta1_manual)

    grad_nll_Theta2 = my_gaussian_likelihood.gradient_neglog_pdf(
        forward_map_evals_Theta2, nll_utils_Theta2
    )
    grad_nll_Theta2_manual = np.zeros((N, D, L))
    assert isinstance(grad_nll_Theta2, np.ndarray) and grad_nll_Theta2.shape == (
        N,
        D,
        L,
    )
    assert np.allclose(grad_nll_Theta2, grad_nll_Theta2_manual)


def test_hess_diag_neglog_pdf(
    settings,
    my_gaussian_likelihood,
    forward_map_evals_Theta1,
    forward_map_evals_Theta2,
    nll_utils_Theta1,
    nll_utils_Theta2,
):
    D, L, N = settings

    # from constant np.ndarray to float
    sigma = my_gaussian_likelihood.sigma.mean()

    hess_diag_nll_Theta1 = my_gaussian_likelihood.hess_diag_neglog_pdf(
        forward_map_evals_Theta1, nll_utils_Theta1
    )
    hess_diag_nll_Theta1_manual = (
        1 / sigma**2 * (1**2 + 1 * (1 - np.e)) * np.ones((N, D, L))
    )
    assert isinstance(
        hess_diag_nll_Theta1, np.ndarray
    ) and hess_diag_nll_Theta1.shape == (
        N,
        D,
        L,
    )
    assert np.allclose(hess_diag_nll_Theta1, hess_diag_nll_Theta1_manual)

    hess_diag_nll_Theta2 = my_gaussian_likelihood.hess_diag_neglog_pdf(
        forward_map_evals_Theta2, nll_utils_Theta2
    )
    print(hess_diag_nll_Theta2)
    hess_diag_nll_Theta2_manual = 1 / sigma**2 * np.e**2 * np.ones((N, D, L))
    assert isinstance(
        hess_diag_nll_Theta2, np.ndarray
    ) and hess_diag_nll_Theta2.shape == (
        N,
        D,
        L,
    )
    assert np.allclose(hess_diag_nll_Theta2, hess_diag_nll_Theta2_manual)


def test_neglog_pdf_candidates(settings, my_gaussian_likelihood):
    D, L, N = settings
    N_candidates = 3 * N
    candidates = np.linspace(-1, 1, N_candidates).reshape((N_candidates, D))

    nll_candidates = my_gaussian_likelihood.neglog_pdf_candidates(
        candidates, idx=np.array([0])
    )

    assert isinstance(nll_candidates, np.ndarray) and nll_candidates.shape == (
        N_candidates,
    )
    # considering that the true value is 1, the best value will be the closest one, ie the last one
    assert np.argmin(nll_candidates) == N_candidates - 1
