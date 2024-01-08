import numpy as np
import pandas as pd
import pytest

from beetroots.modelling.priors.smooth_indicator_prior import (
    SmoothIndicatorPrior,
    penalty_one_pix,
)


@pytest.fixture(scope="module")
def prior():
    D = 3
    N = 5

    indicator_margin_scale = 1e-2
    lower_bounds = np.array([-1.0, -2.0, -3.0])
    upper_bounds = np.array([1.0, 2.0, 3.0])

    prior = SmoothIndicatorPrior(
        D,
        N,
        indicator_margin_scale,
        lower_bounds,
        upper_bounds,
        list_idx_sampling=list(range(D)),
    )
    return prior


def test_penalty_one_pix():
    D = 3

    indicator_margin_scale = 1e-1
    lower_bounds = np.array([0, 0, 0])
    upper_bounds = np.array([1.0, 1.0, 1.0])

    Theta = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, -1, 0.5],
            [0.5, 2, 0.5],
        ]
    )
    penalty = penalty_one_pix(Theta, lower_bounds, upper_bounds, indicator_margin_scale)
    assert np.allclose(penalty, np.array([0, 0, 0, 1e4, 1e4]))


def test_neglog_pdf(prior):
    # 1 point inside "correct region"
    Theta = np.zeros((prior.N, prior.D))
    assert np.allclose(prior.neglog_pdf(Theta), np.zeros((prior.D,)))

    # one value only outside (lower)
    Theta = np.zeros((prior.N, prior.D))
    d = 0
    Theta[0, d] = prior.lower_bounds[d] - prior.indicator_margin_scale

    manual = np.zeros((prior.D,))
    manual[d] += 1

    assert np.allclose(prior.neglog_pdf(Theta), manual)

    # one value only outside (lower)
    Theta = np.zeros((prior.N, prior.D))
    d = 0
    Theta[0, d] = prior.upper_bounds[d] + prior.indicator_margin_scale

    manual = np.zeros((prior.D,))
    manual[d] += 1

    assert np.allclose(prior.neglog_pdf(Theta), manual)


def test_gradient_neglog_pdf(prior):
    Theta = np.zeros((prior.N, prior.D))
    assert np.allclose(prior.gradient_neglog_pdf(Theta), np.zeros((prior.N, prior.D)))

    # one value only outside (lower)
    Theta = np.zeros((prior.N, prior.D))
    n, d = 0, 0
    Theta[n, d] = prior.lower_bounds[d] - prior.indicator_margin_scale

    manual_grad = np.zeros((prior.N, prior.D))
    manual_grad[n, d] = -4 / prior.indicator_margin_scale

    assert np.allclose(prior.gradient_neglog_pdf(Theta), manual_grad)

    # one value only outside (lower)
    Theta = np.zeros((prior.N, prior.D))
    n, d = 0, 0
    Theta[n, d] = prior.upper_bounds[d] + prior.indicator_margin_scale

    manual_grad = np.zeros((prior.N, prior.D))
    manual_grad[n, d] = 4 / prior.indicator_margin_scale

    assert np.allclose(prior.gradient_neglog_pdf(Theta), manual_grad)


def test_hess_diag_neglog_pdf(prior):
    x = np.zeros((prior.N, prior.D))
    assert np.allclose(prior.hess_diag_neglog_pdf(x), np.zeros((prior.N, prior.D)))

    # one value only outside (lower)
    Theta = np.zeros((prior.N, prior.D))
    n, d = 0, 0
    Theta[n, d] = prior.lower_bounds[d] - prior.indicator_margin_scale

    manual_hess_diag = np.zeros((prior.N, prior.D))
    manual_hess_diag[n, d] = 12 / prior.indicator_margin_scale**2

    assert np.allclose(prior.hess_diag_neglog_pdf(Theta), manual_hess_diag)

    # one value only outside (lower)
    Theta = np.zeros((prior.N, prior.D))
    n, d = 0, 0
    Theta[n, d] = prior.upper_bounds[d] + prior.indicator_margin_scale

    manual_hess_diag = np.zeros((prior.N, prior.D))
    manual_hess_diag[n, d] = 12 / prior.indicator_margin_scale**2

    assert np.allclose(prior.hess_diag_neglog_pdf(Theta), manual_hess_diag)
