from typing import Tuple

import numba as nb
import numpy as np
from scipy.special import gamma, ndtr

from beetroots.modelling.priors import smooth_indicator_prior

GAMMA_1_4 = gamma(1 / 4)


@nb.njit()
def sample_generalized_gaussian(
    alpha: float,
    a_alpha: float,
    size: Tuple[int, int],
    seed: int,
) -> np.ndarray:
    r"""sample from a generalized gaussian distribution of pdf:

    .. math::
        :label: eq:pdf_generalized_gaussian

        ..math::
        p(\theta) = \frac{2}{\delta \Gamma(1/4)} \exp \left\{- \left( \frac{\theta}{\delta}\right)^4\right\}

    with here :math:`\delta = 1/A(\alpha)`.
    See Nardon and Pianca, 2006 for more details.

    Parameters
    ----------
    alpha : float
        _description_
    a_alpha : float
        _description_
    size : Tuple
        number of samples to draw

    Returns
    -------
    np.ndarray
        array of samples
    """
    np.random.seed(seed)
    z = np.random.gamma(shape=1 / alpha, scale=(1 / a_alpha) ** alpha, size=size)
    x = np.random.binomial(n=1, p=0.5, size=size) * 2 - 1
    return x * z ** (1 / alpha)


@nb.njit()
def sample_smooth_indicator(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    indicator_margin_scale: float,
    size: Tuple[int, int],
    seed: int,
):
    np.random.seed(seed)
    k_mtm, D = size

    Z_unif = 2 * (upper_bounds - lower_bounds) / (indicator_margin_scale * GAMMA_1_4)
    Z_tot = 1 + Z_unif
    w_unif = Z_unif / Z_tot  # (D,)

    z1 = np.zeros((k_mtm, D))
    x_unif = np.zeros((k_mtm, D))
    for d in range(D):
        z1[:, d] = np.random.binomial(1, w_unif[d], size=k_mtm)
        x_unif[:, d] = np.random.uniform(lower_bounds[d], upper_bounds[d], size=k_mtm)

    x_gg = sample_generalized_gaussian(4.0, 1 / indicator_margin_scale, size, seed)
    x_gg = np.where(
        x_gg < 0,
        x_gg + np.expand_dims(lower_bounds, 0),
        x_gg + np.expand_dims(upper_bounds, 0),
    )
    return np.where(z1 == 1, x_unif, x_gg)


@nb.njit()
def get_neighboring_pixels(
    current_Theta: np.ndarray, list_edges: np.ndarray, idx_pix: int
) -> np.ndarray:
    list_edges_n = list_edges[
        (list_edges[:, 0] == idx_pix) | (list_edges[:, 1] == idx_pix)
    ]
    list_neighbors_idx = list_edges_n.flatten()
    list_neighbors_idx = list_neighbors_idx[list_neighbors_idx != idx_pix]
    return current_Theta[list_neighbors_idx, :]  # (n_neighbors, D)


@nb.njit()
def sample_conditional_spatial_and_indicator_prior(
    current_Theta: np.ndarray,
    spatial_list_edges: np.ndarray,
    spatial_weights: np.ndarray,
    indicator_lower_bounds: np.ndarray,
    indicator_upper_bounds: np.ndarray,
    indicator_indicator_margin_scale: float,
    idx_pix: np.ndarray,
    k_mtm: int,
    seed: int,
):
    np.random.seed(seed)
    (N, D) = current_Theta.shape
    n_pix = idx_pix.size * 1

    samples = np.zeros((n_pix, k_mtm, D))
    i = 0
    for idx_1_pix in idx_pix:
        # * sample from around neighbors
        neighbors = get_neighboring_pixels(current_Theta, spatial_list_edges, idx_1_pix)
        N_neighbors = neighbors.shape[0]

        if N_neighbors > 0:
            x = np.zeros((k_mtm, D))

            for k in range(k_mtm):
                # select which combination f neighbors is going to be used
                arr_use_neighbors = np.random.binomial(n=1, p=0.5, size=N_neighbors)
                while np.max(arr_use_neighbors) == 0:
                    arr_use_neighbors = np.random.binomial(n=1, p=0.5, size=N_neighbors)

                used_neighbors = neighbors[arr_use_neighbors == 1]
                N_used_neighbors = used_neighbors.shape[0]

                # sigma_mtm_eff = 1 / (2 * np.sqrt(N_neighbors * spatial_weights))  # (D,)
                sigma_mtm_eff = 1 / (
                    2 * np.sqrt(N_used_neighbors * spatial_weights)
                )  # (D,)

                # initialize array of candidates

                mean = np.zeros((1, D))
                for d in range(D):
                    mean[0, d] = np.mean(used_neighbors[:, d])
                    # mean[d] = np.mean(neighbors[:, d])

                repeat = True
                n_repeats = 0
                n_repeats_tot = 0
                while repeat:
                    # * step 1 : generate candidates from spatial prior only
                    x_cand = mean + sigma_mtm_eff * np.random.standard_normal(
                        size=(1, D)
                    )

                    # * step 2 : accept or reject with combination of spatial
                    # * and indicator priors
                    p_Theta = np.exp(
                        -smooth_indicator_prior.penalty_one_pix(
                            x_cand,
                            indicator_lower_bounds,
                            indicator_upper_bounds,
                            indicator_indicator_margin_scale,
                        )
                    )  # (1,)
                    u = np.random.uniform(0, 1)
                    if u <= p_Theta[0]:
                        Theta[k] += x_cand[0]
                        repeat = False

                    n_repeats += 1
                    n_repeats_tot += 1

                    if n_repeats >= 10:
                        # select which combination f neighbors is going to be used
                        arr_use_neighbors = np.random.binomial(
                            n=1, p=0.5, size=N_neighbors
                        )
                        while np.max(arr_use_neighbors) == 0:
                            arr_use_neighbors = np.random.binomial(
                                n=1, p=0.5, size=N_neighbors
                            )

                        used_neighbors = neighbors[arr_use_neighbors == 1]
                        N_used_neighbors = used_neighbors.shape[0]

                        # sigma_mtm_eff = 1 / (2 * np.sqrt(N_neighbors * spatial_weights))  # (D,)
                        sigma_mtm_eff = 1 / (
                            2 * np.sqrt(N_used_neighbors * spatial_weights)
                        )  # (D,)

                        # initialize array of candidates

                        mean = np.zeros((1, D))
                        for d in range(D):
                            mean[0, d] = np.mean(used_neighbors[:, d])
                            # mean[d] = np.mean(neighbors[:, d])

                        n_repeats = 0

                    assert n_repeats_tot < 1_000

            samples[i, :, :] = x * 1  # (k_mtm, D)

        else:
            samples[i, :, :] = sample_smooth_indicator(
                indicator_lower_bounds,
                indicator_upper_bounds,
                indicator_indicator_margin_scale,
                size=(k_mtm, D),
                seed=seed,
            )
        i += 1

    return samples  # .reshape((n_pix * k_mtm, D))


@nb.njit()
def compute_sum_subsets_norms(dists: np.ndarray) -> np.ndarray:
    """_summary_

    note: summing for all subset of neighbors, including the empty set, excluding the full set of neighbors

    Parameters
    ----------
    dists : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    N_neighbors, k_mtm_p1, D = dists.shape
    sums_ = np.zeros((2**N_neighbors - 1, k_mtm_p1, D))

    if N_neighbors == 1:
        # sums_ : (1, k_mtm + 1, D)
        return sums_

    if N_neighbors == 2:
        # sums_ : (3, k_mtm + 1, D)
        sums_[1, :] = dists[0, :] * 1
        sums_[2, :] = dists[1, :] * 1
        return sums_

    if N_neighbors == 3:
        # sums_ : (7, k_mtm + 1, D)
        # 1st neighbor
        sums_[1, :] = dists[0, :] * 1
        sums_[2, :] = sums_[1, :] + dists[1, :]
        sums_[3, :] = sums_[1, :] + dists[2, :]
        # 2nd neighbor
        sums_[4, :] = dists[1, :] * 1
        sums_[5, :] = sums_[4, :] + dists[2, :]
        # 3rd neighbor
        sums_[6, :] = dists[2, :] * 1
        return sums_

    if N_neighbors == 4:
        # sums_ : (15, k_mtm + 1, D)
        # 1st neighbor
        sums_[1, :] = dists[0, :] * 1
        sums_[2, :] = sums_[1, :] + dists[1, :]
        sums_[3, :] = sums_[2, :] + dists[2, :]
        sums_[4, :] = sums_[2, :] + dists[3, :]
        sums_[5, :] = sums_[1, :] + dists[2, :]
        sums_[6, :] = sums_[5, :] + dists[3, :]
        sums_[7, :] = sums_[1, :] + dists[3, :]
        # 2nd neighbor
        sums_[8, :] = dists[1, :] * 1
        sums_[9, :] = sums_[8, :] + dists[2, :]
        sums_[10, :] = sums_[9, :] + dists[3, :]
        sums_[11, :] = sums_[8, :] + dists[3, :]
        # 3rd neighbor
        sums_[12, :] = dists[2, :] * 1
        sums_[13, :] = sums_[12, :] + dists[3, :]
        # 4th neighbor
        sums_[14, :] = dists[3, :] * 1
        return sums_

    else:
        assert np.min(sums_) > 0
        return sums_


@nb.njit(fastmath=True)
def numba_logsumexp_stable(x):
    x_max = np.max(x)
    res = 0.0
    for j in range(x.size):
        res += np.exp(x[j] - x_max)
    res = np.log(res) + x_max
    return res


@nb.njit()
def compute_nlratio_prior_proposal(
    new_Theta: np.ndarray,
    spatial_list_edges: np.ndarray,
    spatial_weights: np.ndarray,
    idx_pix: np.ndarray,
    candidates_pix: np.ndarray,
) -> np.ndarray:
    n_pix, k_mtm_p1, D = candidates_pix.shape

    nl_ratio = np.zeros((n_pix, k_mtm_p1))
    for i in range(n_pix):
        idx_1_pix = idx_pix[i]

        # * step 1: get neighbors
        neighbors = get_neighboring_pixels(
            new_Theta, spatial_list_edges, idx_1_pix
        )  # (N_neighbors, D)
        N_neighbors = neighbors.shape[0]

        # * step 2: compute distances between candidate and its neighbors
        dists = (
            np.expand_dims(candidates_pix[i], 0) - np.expand_dims(neighbors, 1)
        ) ** 2  # (N_neighbors, k_mtm_p1, D)
        assert dists.shape == (N_neighbors, k_mtm_p1, D)

        # * step 3: compute list of sums of subsets
        sums_norms = compute_sum_subsets_norms(
            dists
        )  # (2 ** N_neighbors - 1, k_mtm_p1, D)
        assert sums_norms.shape == (2**N_neighbors - 1, k_mtm_p1, D)

        # * step 4: LogSumExp
        inter_ = 2 * np.expand_dims(np.expand_dims(spatial_weights, 0), 0) * sums_norms
        # (2 ** N_neighbors - 1, k_mtm_p1, D)
        for k in range(k_mtm_p1):
            for d in range(D):
                nl_ratio[i, k] += numba_logsumexp_stable(inter_[:, k, d])

    return nl_ratio


@nb.njit()
def correct_add_proposal(
    z_t_add: np.ndarray, current_u: np.ndarray, sigma_a: np.ndarray, random_seed: int
) -> np.ndarray:
    """Corrects for the 0 truncation of the proposal distribution for u, ie ensures that the candidates from the additive proposal are positive.

    Parameters
    ----------
    z_t_add : np.ndarray
        random component
    current_u : np.ndarray
        total number of physical parameters to reconstruct
    sigma_a : np.ndarray
        standard deviation
    random_seed : int
        random seed to use for reproducible results

    Results
    -------
    z_t_add : np.ndarray
        corrected random components
    """
    np.random.seed(random_seed)
    N, L = z_t_add.shape
    for n in range(N):
        for ell in range(L):
            while current_u[n, ell] + z_t_add[n, ell] < 0:
                z_t_add[n, ell] = sigma_a[n, ell] * np.random.normal()
    return z_t_add
