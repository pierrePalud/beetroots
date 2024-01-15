"""Implementation of LogNormal likelihood with censorship (with a lower limit)
"""

from typing import Optional, Union

import numpy as np
from scipy.stats import norm as statsnorm

from beetroots.modelling.likelihoods import utils
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood


class CensoredLogNormalLikelihood(Likelihood):
    r"""Class implementing a LogNormal likelihood model with lower censorship"""

    __slots__ = (
        "forward_map",
        "D",
        "L",
        "N",
        "y",
        "logy",
        "sigma",
        "omega",
        "log_omega",
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
    ) -> None:
        """Constructor of the LogNormalLikelihood object.

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
            mean of the LogNormal distribution
        bias : float or np.ndarray of shape (N, L)
            variance of the LogNormal distribution
        sigma : float or np.ndarray of shape (N, L)
            variance of the LogNormal distribution
        omega : float or np.ndarray of shape (N, L)
            censorship threshold

        Raises
        ------
        ValueError
            y must have the shape (N, L)
        """
        super().__init__(forward_map, D, L, N, y)
        self.logy = np.log(self.y)

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
        self.log_omega = np.log(self.omega)

    def _update_observations(self, y):
        """Update the parameters on which the distribution is defined (if
        updated within the solver).

        Parameters
        ----------
        y : np.ndarray of shape (N, L)
            parameter of the log-normal distribution

        Raises
        ------
        ValueError
            y must have the shape (N, L)
        """
        # ! trigger an error is the mean y contains less than N elements
        if not y.shape == (self.N, self.L):
            raise ValueError(
                "y must have the shape (N, L) = ({}, {}) elements".format(
                    self.N, self.L
                )
            )
        self.y = y
        self.logy = np.log(self.y)

    def neglog_pdf(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        pixelwise: bool = False,
        idx: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        r"""[summary]

        .. math::
            p(y_{n,\ell} \vert x) \propto \exp \left\{- [y_{n,\ell} = \omega] \Phi( \frac{\omega - f_{\ell}(x_n)}{\sigma^2} \right) - [y_{n,\ell} > \omega] \frac{\omega - f_{\ell}(x_n)}{\sigma^2} \right\}
        """
        if idx is None:
            N_pix = self.N * 1
            logy = self.logy * 1
            sigma = self.sigma * 1
            log_omega = self.omega * 1

        else:
            n_pix = idx.size
            k_mtm = forward_map_evals["f_Theta"].shape[0] // n_pix
            N_pix = forward_map_evals["f_Theta"].shape[0]

            logy = np.zeros((n_pix, k_mtm, self.L))
            sigma = np.zeros((n_pix, k_mtm, self.L))
            log_omega = np.zeros((n_pix, k_mtm, self.L))

            for i_pix in range(n_pix):
                logy[i_pix, :, :] = self.logy[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                sigma[i_pix, :, :] = self.sigma[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                log_omega[i_pix, :, :] = self.log_omega[idx[i_pix], :][
                    None, :
                ] * np.ones((k_mtm, self.L))

            logy = logy.reshape((N_pix, self.L))
            sigma = sigma.reshape((N_pix, self.L))
            log_omega = log_omega.reshape((N_pix, self.L))

        nlpdf = np.where(
            logy == log_omega,
            self.neglog_pdf_ac(
                forward_map_evals,
                nll_utils,
                logy,
                sigma,
                log_omega,
            ),
            self.neglog_pdf_au(
                forward_map_evals,
                nll_utils,
                logy,
                sigma,
                log_omega,
            ),
        )  # (N_pix, L)

        if pixelwise:
            return np.sum(nlpdf, axis=1)  # (N_pix,)

        return np.sum(nlpdf)

    def neglog_pdf_ac(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        log_y: np.ndarray,
        log_sigma: np.ndarray,
        omega: np.ndarray,
    ) -> np.ndarray:
        return -statsnorm.logcdf((log_omega - forward_map_evals["f_Theta"]) / sigma)

    def neglog_pdf_au(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        log_y: np.ndarray,
        sigma: np.ndarray,
        log_omega: np.ndarray,
    ) -> np.ndarray:
        return log_y + (forward_map_evals["f_Theta"] - log_y) ** 2 / (2 * sigma**2)

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
                    (self.log_omega - forward_map_evals["f_Theta"]) / self.sigma
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
            * ((forward_map_evals["f_Theta"] - self.logy) / self.sigma**2)[:, None, :]
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
                (self.log_omega - forward_map_evals["f_Theta"]) / self.sigma
            )
            / self.sigma
        )[:, None, :] * (
            forward_map_evals["hess_diag_f_Theta"]
            + forward_map_evals["grad_f_Theta"] ** 2
            * (
                (
                    (self.log_omega - forward_map_evals["f_Theta"]) / self.sigma
                    + utils.norm_pdf_cdf_ratio(
                        (self.log_omega - forward_map_evals["f_Theta"]) / self.sigma
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
            * (forward_map_evals["f_Theta"] - self.logy)[:, None, :]
        )  # (N, D, L)

    def evaluate_all_forward_map(
        self,
        Theta: np.ndarray,
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
    ) -> dict:
        assert len(Theta.shape) == 2 and Theta.shape[1] == self.D
        forward_map_evals = self.forward_map.compute_all(
            Theta, True, False, compute_derivatives, compute_derivatives_2nd_order
        )
        return forward_map_evals

    def evaluate_all_nll_utils(
        self,
        forward_map_evals: dict,
        idx: Optional[np.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        nll_utils = {}
        return nll_utils
