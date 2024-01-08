"""Implementation of a Gaussian mixture model likelihood
"""

from typing import Optional, Union

import numba as nb
import numpy as np

from beetroots.modelling.forward_maps.abstract_base import ForwardMap
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood


@nb.jit(nopython=True, cache=True)
def u_i(Theta: np.ndarray, mu_i: np.ndarray, cov_i_inv: np.ndarray) -> np.ndarray:
    u_ = np.exp(-0.5 * np.dot(Theta - mu_i, np.dot(cov_i_inv, Theta - mu_i)))
    u_ /= np.linalg.det(cov_i_inv) ** 0.5
    return u_


@nb.jit(nopython=True, cache=True)
def grad_u_i(Theta: np.ndarray, mu_i: np.ndarray, cov_i_inv: np.ndarray) -> np.ndarray:
    return -np.dot(cov_i_inv, Theta - mu_i) * u_i(Theta, mu_i, cov_i_inv)  # (N, D)


def hess_u_i(Theta: np.ndarray, mu_i: np.ndarray, cov_i_inv: np.ndarray) -> np.ndarray:
    result_1 = -(Theta - mu_i) * grad_u_i(Theta, mu_i, cov_i_inv)
    result_2 = -u_i(Theta, mu_i, cov_i_inv) * np.diag(cov_i_inv)
    result_1 = result_1.reshape((1, 2))
    result_2 = result_2.reshape((1, 2))
    assert result_1.shape == (1, 2), result_1.shape
    assert result_2.shape == (1, 2), result_2.shape
    return result_1 + result_2


class GaussianMixtureLikelihood(Likelihood):
    """Class implementing a likelihood a Gaussian Mixture model"""

    __slots__ = (
        "forward_map",
        "D",
        "L",
        "N",
        "y",
        "n_means",
        "list_means",
        "list_cov_inv",
    )

    def __init__(
        self,
        forward_map: ForwardMap,
        D: int,
        list_means: np.ndarray,
        list_cov: np.ndarray,
    ) -> None:
        """Constructor of the MixingGaussianLikelihood object.

        Parameters
        ----------
        forward_map : ForwardMap instance
            forward map
        D : int
            number of disinct physical parameters in input space.
        L : int
            number of distinct observed physical parameters.
        N : int
            number of pixels in each physical dimension
        y : np.ndarray of shape (N, L)
            mean of the gaussian distribution
        sigma : float or np.ndarray of shape (N, L)
            variance of the Gaussian distribution

        Raises
        ------
        ValueError
            y must have the shape (N, L)
        """
        L = D * 1  # make sure that theta space and y space are equal
        N = 1  # force only "one pixel"
        y = np.zeros((N, L))
        super().__init__(forward_map, D, L, N, y)

        assert isinstance(list_means, np.ndarray)
        assert list_means.shape[1] == self.D
        self.n_means = list_means.shape[0]
        self.list_means = list_means

        self.list_cov_inv = np.linalg.inv(list_cov)

    def neglog_pdf(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        if idx is None:
            N_pix = self.N * 1
        else:
            N_pix = forward_map_evals["f_Theta"].shape[0]

        nlpdf = -np.log(
            np.sum(
                [
                    [
                        u_i(f_Theta, mu, self.list_cov_inv[i])
                        for i, mu in enumerate(self.list_means)
                    ]
                    for f_Theta in forward_map_evals["f_Theta"]
                ],
                axis=1,
            )
        )
        msg = f"should be ({N_pix},), is {nlpdf.shape}"
        assert nlpdf.shape == (N_pix,), msg

        if pixelwise:
            return nlpdf  # (N_pix,)
        if full:
            return 1 / self.D * np.ones((N_pix, self.D)) * nlpdf[:, None]  # (N_pix, L)

        return np.sum(nlpdf)  # float

    def gradient_neglog_pdf(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        u = np.sum(
            [
                [
                    u_i(f_Theta, mu, self.list_cov_inv[i])
                    for i, mu in enumerate(self.list_means)
                ]
                for f_Theta in forward_map_evals["f_Theta"]
            ],
            # axis=1,
        )  # float

        sum_grad_u_i = np.sum(
            [
                [
                    grad_u_i(f_Theta, mu, self.list_cov_inv[i])
                    for i, mu in enumerate(self.list_means)
                ]
                for f_Theta in forward_map_evals["f_Theta"]
            ],
            axis=1,
        )  # (N_pix, D) = (1, 2)

        grad_ = -1 / u * sum_grad_u_i  # (N_pix, D) = (1, 2)
        # print(u.shape, sum_grad_u_i.shape, grad_.shape)

        assert grad_.shape == (
            self.N,
            self.D,
        ), f"has shape {grad_.shape}, should have ({self.N}, {self.D})"
        return grad_  # (N, D)

    def hess_diag_neglog_pdf(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        u = np.sum(
            [
                [
                    u_i(f_Theta, mu, self.list_cov_inv[i])
                    for i, mu in enumerate(self.list_means)
                ]
                for f_Theta in forward_map_evals["f_Theta"]
            ],
            # axis=1,
        )  # float

        sum_grad_u_i = np.sum(
            [
                [
                    grad_u_i(f_Theta, mu, self.list_cov_inv[i])
                    for i, mu in enumerate(self.list_means)
                ]
                for f_Theta in forward_map_evals["f_Theta"]
            ],
            axis=1,
        )  # (N_pix, D) = (1, 2)

        sum_hess_diag_u_i = np.sum(
            [
                [
                    hess_u_i(f_Theta, mu, self.list_cov_inv[i])
                    for i, mu in enumerate(self.list_means)
                ]
                for f_Theta in forward_map_evals["f_Theta"]
            ],
            axis=(1, 2),
        )  # (N_pix, D) = (1, 2)

        hess_diag = (
            1
            / u**2
            * (u * np.log(sum_hess_diag_u_i) - sum_grad_u_i**2 * np.log(sum_grad_u_i))
        )
        # print(sum_grad_u_i.shape, sum_hess_diag_u_i.shape, hess_diag.shape)
        assert hess_diag.shape == (1, 2)
        return hess_diag  # (N, D)

    def evaluate_all_nll_utils(
        self,
        forward_map_evals: dict,
        idx: Optional[np.ndarray] = None,
        compute_derivatives: bool = True,
    ) -> dict:
        nll_utils = {}
        return nll_utils

    def sample_observation_model(
        self,
        forward_map_evals: dict,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        # to be disregarded, as model checking does not make sense
        # in this example
        return forward_map_evals["f_Theta"]

    def gradient_variable_neglog_pdf(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
    ):
        raise NotImplementedError("")

    def hess_diag_variable_neglog_pdf(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
    ):
        raise NotImplementedError("")
