"""Implementation of Gaussian likelihood with censorship (with a lower limit)
"""

from typing import Optional, Union

import numpy as np
from scipy.stats import norm as statsnorm

from beetroots.modelling.likelihoods import utils
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood


class CensoredGaussianLikelihood(Likelihood):
    r"""Class implementing a Gaussian likelihood model with lower censorship"""

    __slots__ = (
        "forward_map",
        "D",
        "L",
        "N",
        "y",
        "sigma",
        "omega",
        "bias",
    )

    def __init__(
        self,
        forward_map,
        D: int,
        L: int,
        N: int,
        y: np.ndarray,
        sigma: Union[float, np.ndarray],
        omega: Union[float, np.ndarray],
        bias: Union[float, np.ndarray] = 0.0,
    ) -> None:
        """Constructor of the GaussianLikelihood object.

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
        bias : float or np.ndarray of shape (N, L)
            variance of the Gaussian distribution
        sigma : float or np.ndarray of shape (N, L)
            variance of the Gaussian distribution
        omega : float or np.ndarray of shape (N, L)
            censorship threshold

        Raises
        ------
        ValueError
            y must have the shape (N, L)
        """
        super().__init__(forward_map, D, L, N, y)

        # ! trigger an error is the mean y contains less than N elements
        if not (y.shape == (N, L)):
            raise ValueError(
                "y must have the shape (N, L) = ({}, {}) elements".format(
                    self.N, self.L
                )
            )
        if isinstance(sigma, (float, int)):
            self.sigma = sigma * np.ones((N, L))
        else:
            assert sigma.shape == (N, L)
            self.sigma = sigma

        if isinstance(omega, (float, int)):
            self.omega = omega * np.ones((N, L))
        else:
            assert omega.shape == (N, L)
            self.omega = omega

        if isinstance(bias, (float, int)):
            self.bias = bias * np.ones((N, L))
        else:
            assert bias.shape == (N, L)
            self.bias = bias

    def neglog_pdf(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        r"""[summary]

        .. math::
            p(y_{n,\ell} \vert \theta_n) \propto \exp \left\{- [y_{n,\ell} = \omega] \Phi( \frac{\omega - f_{\ell}(\theta_n)}{\sigma^2} \right) - [y_{n,\ell} > \omega] \frac{\omega - f_{\ell}(\theta_n)}{\sigma^2} \right\}
        """
        if idx is None:
            N_pix = self.N * 1
            y = self.y * 1
            sigma = self.sigma * 1
            omega = self.omega * 1
            bias = self.bias * 1

        else:
            n_pix = idx.size
            k_mtm = forward_map_evals["f_Theta"].shape[0] // n_pix
            N_pix = forward_map_evals["f_Theta"].shape[0]
            assert n_pix * k_mtm == N_pix

            y = np.zeros((n_pix, k_mtm, self.L))
            sigma = np.zeros((n_pix, k_mtm, self.L))
            omega = np.zeros((n_pix, k_mtm, self.L))
            bias = np.zeros((n_pix, k_mtm, self.L))

            for i_pix in range(n_pix):
                y[i_pix, :, :] = self.y[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                sigma[i_pix, :, :] = self.sigma[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                omega[i_pix, :, :] = self.omega[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                bias[i_pix, :, :] = self.bias[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            y = y.reshape((N_pix, self.L))
            sigma = sigma.reshape((N_pix, self.L))
            omega = omega.reshape((N_pix, self.L))
            bias = bias.reshape((N_pix, self.L))

        nlpdf = np.where(
            y <= omega,
            self.neglog_pdf_ac(forward_map_evals, nll_utils, y, sigma, omega, bias),
            self.neglog_pdf_au(forward_map_evals, nll_utils, y, sigma, omega, bias),
        )  # (N_pix, L)

        if full:
            return nlpdf  # (N_pix, L)

        if pixelwise:
            return np.sum(nlpdf, axis=1)  # (N_pix,)

        return np.sum(nlpdf)

    def neglog_pdf_ac(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        y: np.ndarray,
        sigma: np.ndarray,
        omega: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        return -statsnorm.logcdf((omega - forward_map_evals["f_Theta"] - bias) / sigma)

    def neglog_pdf_au(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        y: np.ndarray,
        sigma: np.ndarray,
        omega: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        return (forward_map_evals["f_Theta"] + bias - y) ** 2 / (2 * sigma**2)

    def gradient_neglog_pdf(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        """[summary]

        [extended_summary]

        Parameters
        ----------
        x : np.ndarray of shape (N, D)
            [description]
        f_Theta : np.ndarray of shape (N, L), optional
            image of x via forward map, by default None
        grad_f_Theta : np.ndarray of shape (N, D, L), optional
            [description], by default None

        Returns
        -------
        np.ndarray of shape (N, D)
            [description]
        """
        # if f_Theta is None:
        #     f_Theta = self.forward_map.evaluate(x)  # (N, L)
        # if grad_f_Theta is None:
        #     grad_f_Theta = self.forward_map.gradient(x)  # (N, D, L)

        grad_ = np.where(
            (self.y == self.omega)[:, None, :],
            self.gradient_neglog_pdf_ac(forward_map_evals, nll_utils),
            self.gradient_neglog_pdf_au(forward_map_evals, nll_utils),
        )  # (N, D, L)

        # ! issue: do not sum over L if L = D (i.e. identity forward_map)
        if not self.D == self.L:
            grad_ = np.sum(grad_, axis=2)  # (N, D)

        return grad_

    def gradient_neglog_pdf_ac(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        grad_ = (
            forward_map_evals["grad_f_Theta"]
            * (
                utils.norm_pdf_cdf_ratio(
                    (self.omega - forward_map_evals["f_Theta"] - self.bias) / self.sigma
                )
                / self.sigma
            )[:, None, :]
        )
        return grad_  # (N, D, L)

    def gradient_neglog_pdf_au(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        grad_ = (
            forward_map_evals["grad_f_Theta"]
            * ((forward_map_evals["f_Theta"] + self.bias - self.y) / self.sigma**2)[
                :, None, :
            ]
        )
        return grad_  # (N, D, L)

    def hess_diag_neglog_pdf(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        r"""[summary]

        [extended_summary]

        Parameters
        ----------
        x : np.ndarray of shape (N, D)
            [description]
        f_Theta : np.ndarray of shape (N, L), optional
            [description], by default None
        grad_f_Theta : np.ndarray of shape (N, D, L), optional
            [description], by default None
        hess_diag_f_Theta : np.ndarray of shape (N, D, L), optional
            [description], by default None

        Returns
        -------
        np.ndarray of shape (N, D, L)
            [description]
        """
        hess_diag = np.where(
            (self.y == self.omega)[:, None, :],
            self.hess_diag_neglog_pdf_ac(forward_map_evals, nll_utils),
            self.hess_diag_neglog_pdf_au(forward_map_evals, nll_utils),
        )  # (N, D, L)

        # ! issue: do not sum over L if L = D (i.e. identity forward_map)
        if not self.D == self.L:
            hess_diag = np.sum(hess_diag, axis=2)  # (N, D)

        return hess_diag

    def hess_diag_neglog_pdf_ac(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        hess_diag = (
            utils.norm_pdf_cdf_ratio(
                (self.omega - forward_map_evals["f_Theta"]) / self.sigma
            )
            / self.sigma
        )[:, None, :] * (
            forward_map_evals["hess_diag_f_Theta"]
            + forward_map_evals["grad_f_Theta"] ** 2
            * (
                (
                    (self.omega - forward_map_evals["f_Theta"] - self.bias) / self.sigma
                    + utils.norm_pdf_cdf_ratio(
                        (self.omega - forward_map_evals["f_Theta"] - self.bias)
                        / self.sigma
                    )
                )
                / self.sigma
            )[:, None, :]
        )
        return hess_diag  # (N, D, L)

    def hess_diag_neglog_pdf_au(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        return (1 / self.sigma**2)[:, None, :] * (
            forward_map_evals["grad_f_Theta"] ** 2
            + forward_map_evals["hess_diag_f_Theta"]
            * (forward_map_evals["f_Theta"] + self.bias - self.y)[:, None, :]
        )  # (N, D, L)

    def evaluate_all_nll_utils(
        self,
        forward_map_evals: dict,
        idx: Optional[np.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        nll_utils = {}
        return nll_utils

    def sample_observation_model(
        self,
        forward_map_evals: dict,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        return np.maximum(
            self.omega,
            forward_map_evals["f_Theta"] + rng.normal(loc=0.0, scale=self.sigma),
        )

    def evaluate_all_forward_map(
        self,
        Theta: np.ndarray,
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        assert len(Theta.shape) == 2 and Theta.shape[1] == self.D
        forward_map_evals = self.forward_map.compute_all(
            Theta,
            compute_lin=True,
            compute_log=False,
            compute_derivatives=compute_derivatives,
            compute_derivatives_2nd_order=compute_derivatives_2nd_order,
        )
        return forward_map_evals
