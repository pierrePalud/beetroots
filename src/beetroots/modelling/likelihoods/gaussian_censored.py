"""Implementation of Gaussian likelihood with censorship (with a lower limit)"""

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
        pixelwise: bool = False,
        full: bool = False,
        idx_pix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        r"""[summary]

        .. math::
            p(y_{n,\ell} \vert \theta_n) \propto \exp \left\{- [y_{n,\ell} = \omega] \Phi( \frac{\omega - f_{\ell}(\theta_n)}{\sigma^2} \right) - [y_{n,\ell} > \omega] \frac{\omega - f_{\ell}(\theta_n)}{\sigma^2} \right\}
        """
        if idx_pix is None:
            N_pix = self.N * 1
            y = self.y * 1
            sigma = self.sigma * 1
            omega = self.omega * 1
            bias = self.bias * 1

        else:
            n_pix = idx_pix.size
            k_mtm = self.forward_map_evals["f_Var"].shape[0] // n_pix
            N_pix = self.forward_map_evals["f_Var"].shape[0]
            assert n_pix * k_mtm == N_pix

            y = np.zeros((n_pix, k_mtm, self.L))
            sigma = np.zeros((n_pix, k_mtm, self.L))
            omega = np.zeros((n_pix, k_mtm, self.L))
            bias = np.zeros((n_pix, k_mtm, self.L))

            for i_pix in range(n_pix):
                y[i_pix, :, :] = self.y[idx_pix[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                sigma[i_pix, :, :] = self.sigma[idx_pix[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                omega[i_pix, :, :] = self.omega[idx_pix[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                bias[i_pix, :, :] = self.bias[idx_pix[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            y = y.reshape((N_pix, self.L))
            sigma = sigma.reshape((N_pix, self.L))
            omega = omega.reshape((N_pix, self.L))
            bias = bias.reshape((N_pix, self.L))

        nlpdf = np.where(
            y <= omega,
            self.neglog_pdf_ac(
                self.forward_map_evals, self.nlpdf_utils, y, sigma, omega, bias
            ),
            self.neglog_pdf_au(
                self.forward_map_evals, self.nlpdf_utils, y, sigma, omega, bias
            ),
        )  # (N_pix, L)

        if full:
            return nlpdf  # (N_pix, L)

        if pixelwise:
            return np.sum(nlpdf, axis=1)  # (N_pix,)

        return np.sum(nlpdf)

    def neglog_pdf_ac(
        self,
        y: np.ndarray,
        sigma: np.ndarray,
        omega: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        return -statsnorm.logcdf(
            (omega - self.forward_map_evals["f_Var"] - bias) / sigma
        )

    def neglog_pdf_au(
        self,
        y: np.ndarray,
        sigma: np.ndarray,
        omega: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        return (self.forward_map_evals["f_Var"] + bias - y) ** 2 / (2 * sigma**2)

    def gradient_neglog_pdf(
        self,
    ) -> np.ndarray:
        """[summary]

        [extended_summary]

        Parameters
        ----------
        x : np.ndarray of shape (N, D)
            [description]
        f_Var : np.ndarray of shape (N, L), optional
            image of x via forward map, by default None
        grad_f_Var : np.ndarray of shape (N, D, L), optional
            [description], by default None

        Returns
        -------
        np.ndarray of shape (N, D)
            [description]
        """
        # if f_Var is None:
        #     f_Var = self.forward_map.evaluate(x)  # (N, L)
        # if grad_f_Var is None:
        #     grad_f_Var = self.forward_map.gradient(x)  # (N, D, L)

        grad_ = np.where(
            (self.y == self.omega)[:, None, :],
            self.gradient_neglog_pdf_ac(self.forward_map_evals, self.nlpdf_utils),
            self.gradient_neglog_pdf_au(self.forward_map_evals, self.nlpdf_utils),
        )  # (N, D, L)

        # ! issue: do not sum over L if L = D (i.e. identity forward_map)
        if not self.D == self.L:
            grad_ = np.sum(grad_, axis=2)  # (N, D)

        return grad_

    def gradient_neglog_pdf_ac(
        self,
    ) -> np.ndarray:
        grad_ = (
            self.forward_map_evals["grad_f_Var"]
            * (
                utils.norm_pdf_cdf_ratio(
                    (self.omega - self.forward_map_evals["f_Var"] - self.bias)
                    / self.sigma
                )
                / self.sigma
            )[:, None, :]
        )
        return grad_  # (N, D, L)

    def gradient_neglog_pdf_au(
        self,
    ) -> np.ndarray:
        grad_ = (
            self.forward_map_evals["grad_f_Var"]
            * (
                (self.forward_map_evals["f_Var"] + self.bias - self.y) / self.sigma**2
            )[:, None, :]
        )
        return grad_  # (N, D, L)

    def hess_diag_neglog_pdf(
        self,
    ) -> np.ndarray:
        r"""[summary]

        [extended_summary]

        Parameters
        ----------
        x : np.ndarray of shape (N, D)
            [description]
        f_Var : np.ndarray of shape (N, L), optional
            [description], by default None
        grad_f_Var : np.ndarray of shape (N, D, L), optional
            [description], by default None
        hess_diag_f_Var : np.ndarray of shape (N, D, L), optional
            [description], by default None

        Returns
        -------
        np.ndarray of shape (N, D, L)
            [description]
        """
        hess_diag = np.where(
            (self.y == self.omega)[:, None, :],
            self.hess_diag_neglog_pdf_ac(self.forward_map_evals, self.nlpdf_utils),
            self.hess_diag_neglog_pdf_au(self.forward_map_evals, self.nlpdf_utils),
        )  # (N, D, L)

        # ! issue: do not sum over L if L = D (i.e. identity forward_map)
        if not self.D == self.L:
            hess_diag = np.sum(hess_diag, axis=2)  # (N, D)

        return hess_diag

    def hess_diag_neglog_pdf_ac(
        self,
    ) -> np.ndarray:
        hess_diag = (
            utils.norm_pdf_cdf_ratio(
                (self.omega - self.forward_map_evals["f_Var"]) / self.sigma
            )
            / self.sigma
        )[:, None, :] * (
            self.forward_map_evals["hess_diag_f_Var"]
            + self.forward_map_evals["grad_f_Var"] ** 2
            * (
                (
                    (self.omega - self.forward_map_evals["f_Var"] - self.bias)
                    / self.sigma
                    + utils.norm_pdf_cdf_ratio(
                        (self.omega - self.forward_map_evals["f_Var"] - self.bias)
                        / self.sigma
                    )
                )
                / self.sigma
            )[:, None, :]
        )
        return hess_diag  # (N, D, L)

    def hess_diag_neglog_pdf_au(
        self,
    ) -> np.ndarray:
        return (1 / self.sigma**2)[:, None, :] * (
            self.forward_map_evals["grad_f_Var"] ** 2
            + self.forward_map_evals["hess_diag_f_Var"]
            * (self.forward_map_evals["f_Var"] + self.bias - self.y)[:, None, :]
        )  # (N, D, L)

    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[np.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
    ) -> None:
        self.nlpdf_utils = {}

    def sample_observation_model(
        self,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        return np.maximum(
            self.omega,
            self.forward_map_evals["f_Var"] + rng.normal(loc=0.0, scale=self.sigma),
        )

    def evaluate_all_forward_map(
        self,
        Var: np.ndarray,
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        assert len(Var.shape) == 2 and Var.shape[1] == self.D
        self.forward_map_evals = self.forward_map.compute_all(
            Var,
            compute_lin=True,
            compute_log=False,
            compute_derivatives=compute_derivatives,
            compute_derivatives_2nd_order=compute_derivatives_2nd_order,
        )
        return self.forward_map_evals
