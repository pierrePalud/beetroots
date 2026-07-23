import numpy as np
import pytest

from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.kernels.gaussian_approx_aux import (
    GaussianApproximation,
    KernelGaussianApproximationCensored,
)


class MockTargetDistribution(TargetDistribution):
    def __init__(self):
        self.distribution_components = {"comp1": MockComponent()}

    def compute_all(
        self,
    ):
        pass

    def compute_all_for_saver():
        pass

    def grad_neglog_pdf():
        pass

    def hess_diag_neglog_pdf():
        pass

    def neglog_pdf():
        pass


class MockForwardMap:
    def compute_all(self, theta, **kwargs):
        return {"f_Var": theta, "log_f_Var": np.log(theta)}


class MockComponent:
    def __init__(self):
        self.forward_map_evals = {
            "f_Var": np.array([[1e-9]]),
            "log_f_Var": np.array([[np.log(1e-9)]]),
        }
        self.y = np.array([[1.5e-9]])
        self.censored = np.array([[False]])
        self.sigma_m = np.array([[np.log(1.1)]])
        self.sigma_m2 = self.sigma_m**2
        self.sigma_a = np.array([[1.35e-10]])
        self.sigma_a2 = self.sigma_a**2
        self.forward_map = MockForwardMap()


@pytest.fixture
def target_distribution():
    return MockTargetDistribution()


@pytest.fixture
def kernel_gaussian_approximation_censored(target_distribution):
    kernel = KernelGaussianApproximationCensored(
        key="comp1", target_distribution=target_distribution, key_forward_var="theta"
    )
    kernel.var_shape = (1, 1)
    return kernel


# def test_gaussian_approximation_init(gaussian_approximation):
#     assert gaussian_approximation.key == "comp1"
#     assert gaussian_approximation.joint is False
#     assert gaussian_approximation.acceptance_wise_type == "full"
#     assert gaussian_approximation.key_forward_var == "var"


def test_gaussian_approximation_extract_utils(
    kernel_gaussian_approximation_censored, target_distribution
):
    assert (
        kernel_gaussian_approximation_censored.distribution_with_forward_name == "comp1"
    )
    comp = target_distribution.distribution_components["comp1"]
    assert np.array_equal(comp.y, np.array([[1.5e-9]]))
    assert np.array_equal(comp.censored, np.array([[False]]))
    assert np.array_equal(comp.sigma_m, np.array([[np.log(1.1)]]))
    assert np.array_equal(comp.sigma_a, np.array([[1.35e-10]]))


def test_kernel_gaussian_approximation_censored_get_means_and_candidate(
    kernel_gaussian_approximation_censored, target_distribution
):
    current = {
        "comp1": {"var": np.array([1.40e-9])},
        "theta": {"var": np.array([[1e-9]])},
    }
    t = 0
    rng = np.random.default_rng()
    (
        candidate,
        mean_candidate_given_current,
        log_mean_candidate_given_current,
        mean_current_given_candidate,
        log_mean_current_given_candidate,
        weight_lower,
    ) = kernel_gaussian_approximation_censored.get_means_and_candidate(
        current, t, {"comp1": target_distribution}, rng
    )
    comp = target_distribution.distribution_components["comp1"]
    assert mean_candidate_given_current > np.minimum(comp.y, current["theta"]["var"])
    assert mean_candidate_given_current < np.maximum(comp.y, current["theta"]["var"])

    assert (
        kernel_gaussian_approximation_censored.kernel_additive_noise.sigma_a
        < comp.sigma_a
    )
    assert (
        kernel_gaussian_approximation_censored.kernel_additive_noise.sigma_a
        < comp.sigma_m
    )


# def test_kernel_gaussian_approximation_censored_neglog_proposal_candidate_given_current(
#     kernel_gaussian_approximation_censored,
# ):
#     current = {
#         "comp1": {"var": np.array([1.40e-9])},
#         "theta": {"var": np.array([[1e-9]])},
#     }
#     t = 0
#     rng = np.random.default_rng()
#     (
#         candidate,
#         mean_candidate_given_current,
#         log_mean_candidate_given_current,
#         mean_current_given_candidate,
#         log_mean_current_given_candidate,
#         weight_lower,
#     ) = kernel_gaussian_approximation_censored.get_means_and_candidate(
#         current, t, {"comp1": target_distribution}, rng
#     )

#     candidate = np.array([])
#     mean = np.array([1.0])
#     log_mean = np.array([0.0])
#     weight_lower = np.array([1.0])
#     result = (
#         kernel_gaussian_approximation_censored.neglog_proposal_candidate_given_current(
#             candidate, mean, log_mean, weight_lower
#         )
#     )
#     assert result is not None


# def test_kernel_gaussian_approximation_censored_neglog_proposal_current_given_candidate(
#     kernel_gaussian_approximation_censored,
# ):
#     current_var = np.array([1.0])
#     mean = np.array([1.0])
#     log_mean = np.array([0.0])
#     result = (
#         kernel_gaussian_approximation_censored.neglog_proposal_current_given_candidate(
#             current_var, mean, log_mean
#         )
#     )
#     assert result is not None


# def test_kernel_gaussian_approximation_censored_get_new_sample(
#     kernel_gaussian_approximation_censored, target_distribution
# ):
#     current = {"comp1": {"var": np.array([1.0])}}
#     t = 0
#     rng = np.random.default_rng()
#     result = kernel_gaussian_approximation_censored.get_new_sample(
#         current, t, {"comp1": target_distribution}, rng
#     )
#     assert result is not None


# def test_kernel_gaussian_approximation_censored_get_new_sample(
#     kernel_gaussian_approximation_censored, target_distribution
# ):
#     current = {"comp1": {"var": np.array([1.0])}}
#     t = 0
#     rng = np.random.default_rng()
#     result = kernel_gaussian_approximation_censored.get_new_sample(
#         current, t, {"comp1": target_distribution}, rng
#     )
#     assert result is not None
