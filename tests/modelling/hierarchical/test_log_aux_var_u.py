import pytest

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed

from beetroots.modelling.forward_maps.identity import BasicForwardMap
from beetroots.modelling.hierarchical.aux_var_u import AuxiliaryGivenTarget
from beetroots.modelling.hierarchical.log_aux_var_u import LogAuxiliaryGivenTarget


@pytest.fixture(scope="module")
def settings():
    D = 3
    L = 3
    N = 10
    return D, L, N


@pytest.fixture(scope="module")
def inputs(settings):
    D, L, N = settings

    targets = xp.random.rand(N, D) + 1
    aux = xp.random.rand(N, L) + 1
    observations = xp.random.rand(N, L) + 1  # We shift by one to ensure positivity.

    return targets, aux, observations


@pytest.fixture(scope="module")
def my_aux_distribution(settings):
    D, L, N = settings
    forward_map = BasicForwardMap(D, N)

    sigma_m = 0.1
    var_names_dict_log = {"log_aux": "log_aux", "target": "target"}
    var_names_dict_id = {"aux": "aux", "target": "target"}

    my_aux_distribution = AuxiliaryGivenTarget(
        forward_map=forward_map,
        D=D,
        L=L,
        N=N,
        var_names_dict=var_names_dict_id,
        sigma_m=sigma_m,
    )

    my_log_aux_distribution = LogAuxiliaryGivenTarget(
        forward_map=forward_map,
        D=D,
        L=L,
        N=N,
        var_names_dict=var_names_dict_log,
        sigma_m=sigma_m,
    )

    return my_aux_distribution, my_log_aux_distribution


@pytest.fixture(scope="module")
def evaluate_nlpdf_utils_target(my_aux_distribution, inputs):
    aux_dist, log_aux_dist = my_aux_distribution
    targets, aux, observations = inputs

    current_aux = {
        "aux": {"var": aux},
        "target": {"var": targets},
    }
    current_log_aux = {
        "log_aux": {"var": xp.log(aux)},
        "target": {"var": targets},
    }

    aux_dist.evaluate_all_nlpdf_utils(
        current_aux,
        idx_pix=None,
        compute_derivatives=True,
        compute_derivatives_2nd_order=True,
        mtm=False,
        deriv_var_name="target",
    )

    log_aux_dist.evaluate_all_nlpdf_utils(
        current_log_aux,
        idx_pix=None,
        compute_derivatives=True,
        compute_derivatives_2nd_order=True,
        mtm=False,
        deriv_var_name="target",
    )

    return aux_dist, log_aux_dist


@pytest.fixture(scope="module")
def evaluate_nlpdf_utils_aux(my_aux_distribution, inputs):
    aux_dist, log_aux_dist = my_aux_distribution
    targets, aux, observations = inputs

    current_aux = {
        "aux": {"var": aux},
        "target": {"var": targets},
    }
    current_log_aux = {
        "log_aux": {"var": xp.log(aux)},
        "target": {"var": targets},
    }

    aux_dist.evaluate_all_nlpdf_utils(
        current_aux,
        idx_pix=None,
        compute_derivatives=True,
        compute_derivatives_2nd_order=True,
        mtm=False,
        deriv_var_name="aux",
    )

    log_aux_dist.evaluate_all_nlpdf_utils(
        current_log_aux,
        idx_pix=None,
        compute_derivatives=True,
        compute_derivatives_2nd_order=True,
        mtm=False,
        deriv_var_name="log_aux",
    )

    return aux_dist, log_aux_dist


def test_pdf_equality_aux_log_aux_jacobian(evaluate_nlpdf_utils_target, inputs):
    targets, aux, observations = inputs
    aux_dist, log_aux_dist = evaluate_nlpdf_utils_target

    aux_pdf = aux_dist.neglog_pdf(full=True) - xp.log(aux)
    log_aux_pdf = log_aux_dist.neglog_pdf(full=True)

    assert xp.allclose(aux_pdf, log_aux_pdf)


def test_gradient_equality_aux_log_aux_wrt_target(evaluate_nlpdf_utils_target):
    aux_dist, log_aux_dist = evaluate_nlpdf_utils_target

    aux_grad = aux_dist.gradient_neglog_pdf(deriv_var_name="target")
    log_aux_grad = log_aux_dist.gradient_neglog_pdf(deriv_var_name="target")

    assert xp.allclose(aux_grad, log_aux_grad)


def test_hess_equality_aux_log_aux_wrt_target(evaluate_nlpdf_utils_target):
    aux_dist, log_aux_dist = evaluate_nlpdf_utils_target

    aux_hess = aux_dist.hess_diag_neglog_pdf(deriv_var_name="target")
    log_aux_hess = log_aux_dist.hess_diag_neglog_pdf(deriv_var_name="target")

    assert xp.allclose(aux_hess, log_aux_hess)


def test_gradient_equality_aux_log_aux_wrt_aux_log_aux(
    evaluate_nlpdf_utils_aux, inputs
):
    targets, aux, observations = inputs
    aux_dist, log_aux_dist = evaluate_nlpdf_utils_aux

    aux_grad = aux_dist.gradient_neglog_pdf(deriv_var_name="aux")
    log_aux_grad = (
        log_aux_dist.gradient_neglog_pdf(deriv_var_name="log_aux") / aux + 1 / aux
    )

    assert xp.allclose(aux_grad, log_aux_grad)
    assert xp.allclose(aux_grad, log_aux_grad)
