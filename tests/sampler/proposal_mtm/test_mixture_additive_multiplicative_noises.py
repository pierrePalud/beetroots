import numpy as np
import pytest

from beetroots.sampler.kernels.additive_noise import AdditiveNoise
from beetroots.sampler.kernels.log_multiplicative_noise import (
    LogMultiplicativeNoise,
)
from beetroots.sampler.kernels.mixture_additive_multiplicative_noises import (
    MixtureAdditiveMultiplicative,
)


@pytest.fixture
def setup_mixture():
    N, k_mtm, D = 10, 5, 3
    s_m = np.random.rand(N, D)
    s_a = np.random.rand(N, D)
    omega = 0.1 + np.zeros((N, D))

    return (
        AdditiveNoise(s_a, s_m=s_m, omega=omega),
        LogMultiplicativeNoise(s_m),
        MixtureAdditiveMultiplicative(s_m=s_m, s_a=s_a, omega=omega),
        N,
        k_mtm,
        D,
    )


def test_sample(setup_mixture):
    additive_noise, log_multiplicative_noise, mixture, N, k_mtm, D = setup_mixture
    mean = np.random.rand(N, 1, D)
    idx_pix = np.arange(N)

    rng = np.random.default_rng(seed=42)
    samples_mixture = mixture.sample(mean=mean, idx_pix=idx_pix, k_mtm=k_mtm, rng=rng)

    samples_by_hand = np.zeros((N, k_mtm, D))
    rng = np.random.default_rng(seed=42)
    is_mult = rng.random(size=N) < 0.5
    samples_by_hand[is_mult] += log_multiplicative_noise.sample(
        mean=mean[is_mult], idx_pix=idx_pix[is_mult], k_mtm=k_mtm, rng=rng
    )
    samples_by_hand[~is_mult] += additive_noise.sample(
        mean=mean[~is_mult], idx_pix=idx_pix[~is_mult], k_mtm=k_mtm, rng=rng
    )

    assert samples_mixture.shape == (N, k_mtm, D)
    assert np.all(samples_mixture == samples_by_hand)


def test_neglog_pdf(setup_mixture):
    additive_noise, log_multiplicative_noise, mixture, N, k_mtm, D = setup_mixture
    idx_pix = np.arange(N)
    candidates = np.random.rand(N, k_mtm, D)
    mean = np.random.rand(N, 1, D)

    rng = np.random.default_rng(seed=42)
    is_mult = rng.random(size=N) < 0.5
    mixture.is_mult = is_mult

    neglog_pdf_values = mixture.neglog_pdf(
        candidates=candidates, mean=mean, idx_pix=idx_pix
    )

    neglog_pdf_values_by_hand = np.zeros((N, k_mtm))
    neglog_pdf_values_by_hand[is_mult] += log_multiplicative_noise.neglog_pdf(
        candidates=candidates[is_mult],
        mean=mean[is_mult],
        idx_pix=idx_pix[is_mult],
    )
    neglog_pdf_values_by_hand[~is_mult] += additive_noise.neglog_pdf(
        candidates=candidates[~is_mult],
        mean=mean[~is_mult],
        idx_pix=idx_pix[~is_mult],
    )

    assert neglog_pdf_values.shape == (N, k_mtm)
    assert np.all(neglog_pdf_values == neglog_pdf_values_by_hand)
