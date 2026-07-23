import numpy as np
import pytest

from beetroots.sampler.kernels.additive_noise import AdditiveNoise


@pytest.fixture
def setup_additive_noise():
    N, D = 10, 2
    s_a = np.ones((N, D))
    s_m = np.log(1.1 * np.ones((N, D)))
    omega = 0.1 + np.zeros((N, D))
    return AdditiveNoise(s_a=s_a, s_m=s_m, omega=omega), N, D


def test_sample(setup_additive_noise):
    additive_noise, N, D = setup_additive_noise
    mean = np.zeros((N, 1, D))
    idx_pix = np.arange(N)
    k_mtm = 100
    rng = np.random.default_rng(seed=42)
    factor_s_a = 1.0

    samples = additive_noise.sample(mean, idx_pix, k_mtm, rng, factor_s_a)

    assert samples.shape == (N, k_mtm, D)
    assert np.abs(np.mean(samples - mean)) < 1e-1
    assert 0.9 < np.var(samples) and np.var(samples) < 1.1


def test_sample_with_exponential_transform(setup_additive_noise):
    additive_noise, N, D = setup_additive_noise
    additive_noise.exponential_transform = True
    mean = np.ones((N, 1, D))
    idx_pix = np.arange(N)
    k_mtm = 100
    rng = np.random.default_rng(seed=42)
    factor_s_a = 1.0

    samples = additive_noise.sample(mean, idx_pix, k_mtm, rng, factor_s_a)

    assert samples.shape == (N, k_mtm, D)
    assert 0.9 < np.abs(np.mean(samples)) and np.abs(np.mean(samples)) < 1.1


def test_neglog_pdf(setup_additive_noise):
    additive_noise, N, D = setup_additive_noise
    additive_noise.exponential_transform = False
    k_mtm = 5
    candidates = np.ones((N, k_mtm, D))
    mean = np.zeros((N, 1, D))
    idx_pix = np.arange(N)
    factor_s_a = 1.0
    s_a = additive_noise.s_a

    neglog_pdf = additive_noise.neglog_pdf(candidates, mean, idx_pix, factor_s_a)

    assert neglog_pdf.shape == (N, k_mtm)
    assert np.allclose(
        neglog_pdf,
        0.5
        * np.sum(
            ((np.exp(candidates) - np.exp(mean)) / s_a[:, None]) ** 2,
            axis=-1,
        ),
    )
