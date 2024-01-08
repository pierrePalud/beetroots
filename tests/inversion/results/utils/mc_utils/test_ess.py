import numpy as np

from beetroots.inversion.results.utils.mc_utils import ess


def test_compute_ess():
    rng = np.random.default_rng(42)
    n = 10_000

    # independent samples
    x = rng.standard_normal(size=(1, n))
    print(ess.compute_ess(x))
    assert 0.95 * n <= ess.compute_ess(x) <= 1.05 * n

    # fully correlated samples
    Theta_0 = rng.standard_normal()
    x = Theta_0 * np.ones((1, n))
    assert np.isclose(ess.compute_ess(x), n / (2 * n + 1), rtol=1e-3)
