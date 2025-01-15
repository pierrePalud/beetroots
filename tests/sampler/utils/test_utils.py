import math

import numpy as np
import pandas as pd
import pytest

from beetroots.modelling.priors.l22_discrete_grad_prior import (
    L22DiscreteGradSpatialPrior,
)
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.sampler.utils.utils import (
    compute_nlpdf_spatial_proposal,
    compute_sum_subsets_mean,
)


@pytest.fixture(scope="module")
def build_test_1():
    D = 1
    k_mtm = 2
    N = 9
    scalar = 3.0

    x, y = np.meshgrid(list(range(3)), list(range(3)))
    df = pd.DataFrame()
    df["x"] = x.flatten()
    df["y"] = y.flatten()
    df["idx"] = np.arange(len(df))
    df["vals_1"] = 0.0
    assert len(df) == 9

    df = df.set_index(["x", "y"])
    df.loc[(1, 1), "vals_1"] = 1.0
    df["vals_2"] = scalar * df["vals_1"]
    spatial_prior_params = SpatialPriorParams(
        name="L2-laplacian",
        use_next_nearest_neighbors=False,
        initial_regu_weights=np.ones((D,)),
    )

    prior = L22DiscreteGradSpatialPrior(
        spatial_prior_params, "", D, N, df, list_idx_sampling=list(range(D))
    )

    Theta1 = df["vals_1"].values.reshape((N, D))
    Theta2 = df["vals_2"].values.reshape((N, D))
    Theta_full = df[["vals_1", "vals_2"]].values.reshape((N, k_mtm, D))
    return Theta1, Theta2, Theta_full, scalar, prior


def test_compute_nlpdf_spatial_proposal(build_test_1):
    Theta1, Theta2, Theta_full, scalar, prior = build_test_1
    N, k_mtm, D = Theta_full.shape
    idx_pix = np.array([4])

    _, N_neighbors = compute_sum_subsets_mean(np.zeros((4, k_mtm, D)))

    true_logpdf = np.zeros(Theta_full.shape[1])
    # Theta_full = np.ones((N, k_mtm, D))

    true_logpdf += (
        math.comb(4, 4) * 4 * np.exp(-(4**2) * Theta_full[idx_pix[0], :, 0] ** 2)
    )  # mode with 4 neighbors
    true_logpdf += (
        math.comb(4, 1) * 1 * np.exp(-(1**2) * Theta_full[idx_pix[0], :, 0] ** 2)
    )  # modes with 1 neighbor
    true_logpdf += (
        math.comb(4, 2) * 2 * np.exp(-(2**2) * Theta_full[idx_pix[0], :, 0] ** 2)
    )  # modes with 2 neighbors
    true_logpdf += (
        math.comb(4, 3) * 3 * np.exp(-(3**2) * Theta_full[idx_pix[0], :, 0] ** 2)
    )  # modes with 3 neighbors
    true_logpdf = np.log(true_logpdf)

    logpdf = compute_nlpdf_spatial_proposal(
        Theta_full, prior.list_edges, np.ones(1), idx_pix
    )

    assert np.allclose(true_logpdf[None, :], logpdf)
