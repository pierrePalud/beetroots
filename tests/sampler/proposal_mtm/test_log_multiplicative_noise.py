import numpy as np
import pytest

from beetroots.sampler.kernels.log_multiplicative_noise import (
    LogMultiplicativeNoise,
)


@pytest.fixture
def setup_data():
    s_m = np.ones((10, 5))
    return s_m


def test_init(setup_data):
    s_m = setup_data
    proposal = LogMultiplicativeNoise(s_m)
    assert np.array_equal(proposal.s_m, s_m)


def test_sample(setup_data):
    s_m = setup_data
    proposal = LogMultiplicativeNoise(s_m)
    mean = np.zeros((10, 1, 5))
    idx_pix = np.arange(10)
    k_mtm = 1000
    rng = np.random.default_rng()
    samples = proposal.sample(mean, idx_pix, k_mtm, rng)
    assert samples.shape == (10, k_mtm, 5)
    assert np.abs(np.mean(samples - (mean - (s_m[:, None] ** 2) / 2))) < 1e-1


def test_neglog_pdf(setup_data):
    s_m = setup_data
    proposal = LogMultiplicativeNoise(s_m)
    mean = np.zeros((10, 1, 5))
    idx_pix = np.arange(10)
    candidates = np.random.rand(10, 3, 5)
    neglog_pdf = proposal.neglog_pdf(candidates, mean, idx_pix)
    assert neglog_pdf.shape == (10, 3)
    assert np.allclose(
        neglog_pdf,
        0.5
        * np.sum(
            (candidates - mean + (s_m[:, None] ** 2) / 2) ** 2 / s_m[:, None] ** 2,
            axis=-1,
        ),
    )
