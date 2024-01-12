"""implementation of ESS computation, from :cite:t:`gelmanBayesianDataAnalysis2015a`

Original implementation : https://github.com/jwalton3141/jwalton3141.github.io/blob/master/assets/posts/ESS/rwmh.py

Corresponding post : https://jwalton.info/Efficient-effective-sample-size-python/
"""

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def compute_ess(Theta: np.ndarray) -> float:
    r"""Computes the effective sample size of estimand of interest for one physical parameter :math:`\theta_{nd}`. Vectorized implementation.

    Parameters
    ----------
    Theta : np.ndarray of shape (m_chains, n_iters)
        values of the ``n_iters`` iterates for each of the ``m_chains`` Markov chains associated to one physical parameter :math:`\theta_{nd}`

    Returns
    -------
    float
        ESS for physical parameter :math:`\theta_{nd}`
    """

    # if len(Theta.shape) == 1:
    #     Theta = Theta.reshape((1, -1))
    # assert (
    #     len(Theta.shape) == 2
    # ), f"Theta has shape {Theta.shape}, should be a 2D array of shape (m_chains, n_iters)"

    m_chains, n_iters = Theta.shape

    variogram = lambda t: ((Theta[:, t:] - Theta[:, : (n_iters - t)]) ** 2).sum() / (
        m_chains * (n_iters - t)
    )

    post_var = gelman_rubin(Theta)

    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = np.sum(rho[t - 1 : t + 1]) < 0

        t += 1

    return m_chains * n_iters / (1 + 2 * rho[1:t].sum())


@nb.jit(nopython=True)
def gelman_rubin(Theta: np.ndarray) -> float:
    r"""Estimates the marginal posterior variance. Vectorized implementation.

    Parameters
    ----------
    Theta : np.ndarray of shape (m_chains, n_iters)
        values of the ``n_iters`` iterates for each of the ``m_chains`` Markov chains associated to one physical parameter :math:`\theta_{nd}`

    Returns
    -------
    float
        marginal posterior variance
    """
    m_chains, n_iters = Theta.shape

    mean_arr = np.sum(Theta, axis=1) / n_iters
    if m_chains > 1:
        # Calculate between-chain variance
        B_over_n = ((mean_arr - np.mean(Theta)) ** 2).sum() / (m_chains - 1)
    else:
        B_over_n = 0.0

    # Calculate within-chain variances
    Theta_mean_same_dims = np.zeros_like(Theta)
    for chain in range(m_chains):
        Theta_mean_same_dims[chain, :] = mean_arr[chain]

    W = ((Theta - Theta_mean_same_dims) ** 2).sum() / (m_chains * (n_iters - 1))

    # (over) estimate of variance
    s2 = W * (n_iters - 1) / n_iters + B_over_n

    return s2
