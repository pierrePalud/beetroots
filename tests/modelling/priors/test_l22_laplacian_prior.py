import numpy as np
import pandas as pd
import pytest

from beetroots.modelling.priors.l22_laplacian_prior import (
    L22LaplacianSpatialPrior,
    compute_laplacian,
)
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams


@pytest.fixture(scope="module")
def build_test_1():
    D = 1
    D_full = 2
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
        use_next_nearest_neighbours=False,
        initial_regu_weights=np.ones((D,)),
        use_clustering=False,
        n_clusters=None,  # None,
        cluster_algo=None,  # None,
    )
    spatial_prior_params_full = SpatialPriorParams(
        name="L2-laplacian",
        use_next_nearest_neighbours=False,
        initial_regu_weights=np.ones((D_full,)),
        use_clustering=False,
        n_clusters=None,  # None,
        cluster_algo=None,  # None,
    )

    prior = L22LaplacianSpatialPrior(
        spatial_prior_params, "", D, N, df, list_idx_sampling=list(range(D))
    )
    prior_full = L22LaplacianSpatialPrior(
        spatial_prior_params_full,
        "",
        D_full,
        N,
        df,
        list_idx_sampling=list(range(D_full)),
    )

    Theta1 = df["vals_1"].values.reshape((N, D))
    Theta2 = df["vals_2"].values.reshape((N, D))
    Theta_full = df[["vals_1", "vals_2"]].values.reshape((N, D_full))
    return Theta1, Theta2, Theta_full, prior, prior_full, scalar


@pytest.fixture(scope="module")
def build_test_local():
    D = 1
    D_full = 2
    N_side = 5
    N = N_side**2
    scalar = 3.0

    x, y = np.meshgrid(list(range(N_side)), list(range(N_side)))
    df = pd.DataFrame()
    df["x"] = x.flatten()
    df["y"] = y.flatten()
    df["idx"] = np.arange(N)
    df["vals_1"] = np.mod(df["idx"], 2.0)
    assert len(df) == N

    df = df.set_index(["x", "y"])
    df["vals_2"] = scalar * df["vals_1"]

    spatial_prior_params = SpatialPriorParams(
        name="L2-laplacian",
        use_next_nearest_neighbours=False,
        initial_regu_weights=np.ones((D,)),
        use_clustering=False,
        n_clusters=None,  # None,
        cluster_algo=None,  # None,
    )
    spatial_prior_params_full = SpatialPriorParams(
        name="L2-laplacian",
        use_next_nearest_neighbours=False,
        initial_regu_weights=np.ones((D_full,)),
        use_clustering=False,
        n_clusters=None,  # None,
        cluster_algo=None,  # None,
    )

    prior = L22LaplacianSpatialPrior(
        spatial_prior_params, "", D, N, df, list_idx_sampling=list(range(D))
    )
    prior_full = L22LaplacianSpatialPrior(
        spatial_prior_params_full,
        "",
        D_full,
        N,
        df,
        list_idx_sampling=list(range(D_full)),
    )

    Theta1 = df["vals_1"].values.reshape((N, D))
    Theta2 = df["vals_2"].values.reshape((N, D))
    Theta_full = df[["vals_1", "vals_2"]].values.reshape((N, D_full))
    return Theta1, Theta2, Theta_full, prior, prior_full, scalar


# def test_init(build_first_test, build_second_test, build_third_test):
#     _, prior = build_first_test
#     assert prior.list_edges.shape == (12, 2)

#     _, prior = build_second_test
#     assert prior.list_edges.shape == (0,)

#     _, prior = build_third_test
#     assert prior.list_edges.shape == (9, 2)


def test_compute_laplacian(build_test_1):
    Theta1, Theta2, Theta_full, prior, prior_full, scalar = build_test_1
    N, D = Theta1.shape
    _, D_full = Theta_full.shape

    laplacian_1 = compute_laplacian(Theta1, prior.list_edges)
    true_laplacian_1 = np.array([0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]).reshape(
        (N, D)
    )
    assert laplacian_1.shape == (N, D)
    assert np.allclose(true_laplacian_1, laplacian_1)

    laplacian_2 = compute_laplacian(Theta2, prior.list_edges)
    true_laplacian_2 = scalar * true_laplacian_1
    assert np.allclose(true_laplacian_2, laplacian_2)

    laplacian_full = compute_laplacian(Theta_full, prior_full.list_edges)
    true_laplacian_full = np.hstack([true_laplacian_1, true_laplacian_2])  # (N, D_full)
    assert np.allclose(true_laplacian_full, laplacian_full)


def test_neglog_pdf_one_pix(build_test_local):
    Theta1, Theta2, Theta_full, prior, prior_full, scalar = build_test_local
    N, D = Theta1.shape
    _, D_full = Theta_full.shape

    laplacian_1 = compute_laplacian(Theta1, prior.list_edges)
    laplacian_2 = compute_laplacian(Theta2, prior.list_edges)
    laplacian_full = compute_laplacian(Theta_full, prior_full.list_edges)

    for n in range(N):
        print(Theta1, Theta1[[n]])
        neglog_pdf_n_1 = prior.neglog_pdf_one_pix(
            Theta=Theta1,
            idx_pix=np.array([n]),
            list_pixel_candidates=Theta1[[n]].reshape((1, 1, D)),
        )
        assert np.allclose(1.0 * laplacian_1[n] ** 2, neglog_pdf_n_1)

        neglog_pdf_n_2 = prior.neglog_pdf_one_pix(
            Theta=Theta2,
            idx_pix=np.array([n]),
            list_pixel_candidates=Theta2[[n]].reshape((1, 1, D)),
        )  # (D,)
        assert np.allclose(1.0 * laplacian_2[n] ** 2, neglog_pdf_n_2)

        neglog_pdf_n_full = prior_full.neglog_pdf_one_pix(
            Theta=Theta_full,
            idx_pix=np.array([n]),
            list_pixel_candidates=Theta_full[[n]].reshape((1, 1, D_full)),
        )  # (D_full,)
        assert np.allclose(np.sum(1.0 * laplacian_full[n] ** 2), neglog_pdf_n_full)


def test_neglog_pdf(build_test_1):
    Theta1, Theta2, Theta_full, prior, prior_full, scalar = build_test_1
    N, D = Theta1.shape
    _, D_full = Theta_full.shape

    laplacian_1 = compute_laplacian(Theta1, prior.list_edges)
    true_laplacian_1 = np.array([0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]).reshape(
        (N, D)
    )
    assert laplacian_1.shape == (N, D)
    assert np.allclose(true_laplacian_1, laplacian_1)

    laplacian_2 = compute_laplacian(Theta2, prior.list_edges)
    true_laplacian_2 = scalar * true_laplacian_1
    assert np.allclose(true_laplacian_2, laplacian_2)

    laplacian_full = compute_laplacian(Theta_full, prior_full.list_edges)
    true_laplacian_full = np.hstack([true_laplacian_1, true_laplacian_2])  # (N, D_full)
    assert np.allclose(true_laplacian_full, laplacian_full)

    neglogpdf_1 = prior.neglog_pdf(Theta1)
    true_val_1 = np.array([20.0])
    assert neglogpdf_1.shape == (D,)
    assert np.allclose(true_val_1, neglogpdf_1)

    neglogpdf_2 = prior.neglog_pdf(Theta2)
    true_val_2 = scalar**2 * true_val_1
    assert np.allclose(true_val_2, neglogpdf_2)

    neglogpdf_full = prior_full.neglog_pdf(Theta_full)
    true_val_full = np.hstack([true_val_1, true_val_2])  # (D_full,)
    assert neglogpdf_full.shape == (D_full,)
    assert np.allclose(true_val_full, neglogpdf_full)


def test_gradient_neglogpdf(build_test_1):
    Theta1, Theta2, Theta_full, prior, prior_full, scalar = build_test_1
    N, D = Theta1.shape
    _, D_full = Theta_full.shape

    grad_1 = prior.gradient_neglog_pdf(Theta1)
    true_grad_1 = np.array(
        [4.0, -14.0, 4.0, -14.0, 40.0, -14.0, 4.0, -14.0, 4.0]
    ).reshape((N, D))
    assert grad_1.shape == (N, D)
    assert np.allclose(true_grad_1, grad_1)

    grad_2 = prior.gradient_neglog_pdf(Theta2)
    true_grad_2 = scalar * true_grad_1
    assert np.allclose(true_grad_2, grad_2)

    grad_full = prior_full.gradient_neglog_pdf(Theta_full)
    true_grad_full = np.hstack([true_grad_1, true_grad_2])
    assert grad_full.shape == (N, D_full)
    assert np.allclose(true_grad_full, grad_full)


def test_hess_diag_neglogpdf(build_test_1):
    Theta1, Theta2, Theta_full, prior, prior_full, scalar = build_test_1
    N, D = Theta1.shape
    _, D_full = Theta_full.shape

    hess_diag_1 = prior.hess_diag_neglog_pdf(Theta1)
    true_hess_diag_1 = np.array(
        [12.0, 24.0, 12.0, 24.0, 40.0, 24.0, 12.0, 24.0, 12.0]
    ).reshape((N, D))
    assert hess_diag_1.shape == (N, D)
    assert np.allclose(true_hess_diag_1, hess_diag_1)

    hess_diag_2 = prior.hess_diag_neglog_pdf(Theta2)
    true_hess_diag_2 = true_hess_diag_1 * 1
    assert np.allclose(true_hess_diag_2, hess_diag_2)

    hess_diag_full = prior_full.hess_diag_neglog_pdf(Theta_full)
    true_hess_diag_full = np.hstack([true_hess_diag_1, true_hess_diag_2])
    assert hess_diag_full.shape == (N, D_full)
    assert np.allclose(true_hess_diag_full, hess_diag_full)
