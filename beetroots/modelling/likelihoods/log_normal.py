"""Implementation of log-normal likelihood
"""

from typing import Optional, Union

import numpy as np

from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood


class LogNormalLikelihood(Likelihood):
    """Class implementing a log-normal likelihood model."""

    __slots__ = (
        "forward_map",
        "D",
        "L",
        "N",
        "y",
        "logy",
        "sigma",
    )

    def __init__(
        self,
        forward_map,
        D: int,
        L: int,
        N: int,
        y: np.ndarray,
        sigma: Union[float, np.ndarray],
    ) -> None:
        """Constructor of the LogNormalLikelihood object.

        Parameters
        ----------
        forward_map : ForwardMap instance
            forward map, involved in the mean of the distribution.
        D : int
            number of disinct physical parameters in input space.
        L : int
            number of distinct observed physical parameters.
        N : int
            number of pixels in each physical dimension
        y : np.ndarray of shape (N, L)
            parameter of the log-normal distribution
        sigma : float or np.ndarray of shape (N, L)
            variance of the log-normal distribution

        Raises
        ------
        ValueError
            y must have the shape (N, L)

        Note
        ----
        * Derivatives and Hessians are taken with respect of the mean of the distribution.
        * y provided in log space already? (saving computations)

        """

        # TODO: add method to update y? (instead of having to reinstantiate the full object any time y is updated?)

        super().__init__(forward_map, D, L, N, y)
        self.logy = np.log(self.y)

        # ! trigger an error is the mean y contains less than N elements
        if not y.shape == (N, L):
            raise ValueError(
                "y must have the shape (N, L) = ({}, {}) elements".format(
                    self.N, self.L
                )
            )
        if isinstance(sigma, (float, int)):
            self.sigma = sigma * np.ones(
                (N, L)
            )  # ! P.-A.: not sure this is actually needed... (unless broadcast is not enough here)
        else:
            assert sigma.shape == (N, L)
            self.sigma = sigma

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
        full: bool = False,
        idx: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        # TODO: there are a few steps to be clarified in there
        # TODO: (what is the point of the reformatting step in here)
        # a priori, sam echange expected
        if idx is None:
            N_pix = self.N * 1
            logy = self.logy * 1
            sigma = self.sigma * 1
        else:
            n_pix = idx.size
            k_mtm = forward_map_evals["f_Theta"].shape[0] // n_pix
            N_pix = forward_map_evals["f_Theta"].shape[0]

            logy = np.zeros((n_pix, k_mtm, self.L))
            sigma = np.zeros((n_pix, k_mtm, self.L))

            for i_pix in range(n_pix):
                logy[i_pix, :, :] = self.logy[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                sigma[i_pix, :, :] = self.sigma[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            logy = logy.reshape((N_pix, self.L))
            sigma = sigma.reshape((N_pix, self.L))

        nlpdf = logy + (logy - forward_map_evals["log_f_Theta"]) ** 2 / (
            2 * sigma**2
        )  # (N_pix, L)

        if full:
            return nlpdf  # (N_pix, L)

        if pixelwise:
            return np.sum(nlpdf, axis=1)  # (N_pix,)

        return np.sum(nlpdf)  # float

    def gradient_neglog_pdf(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        grad_ = (
            forward_map_evals["grad_log_f_Theta"]
            * ((forward_map_evals["log_f_Theta"] - self.logy) / self.sigma**2)[
                :, None, :
            ]
        )  # (N, D, L)

        # ! issue: do not sum over L if L = D (i.e. identity forward_map)
        if not self.D == self.L:
            grad_ = np.sum(grad_, axis=2)  # (N, D)

        return grad_

    def hess_diag_neglog_pdf(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        r"""Hessian w.r.t to the parameter of the log-normal distribution.

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
        np.ndarray of shape (N, D)
            [description]
        """
        hess_diag = (1 / self.sigma**2)[:, None, :] * (
            forward_map_evals["grad_log_f_Theta"] ** 2
            + forward_map_evals["hess_diag_log_f_Theta"]
            * (forward_map_evals["f_Theta"] - self.logy)[:, None, :]
        )

        # ! issue: do not sum over L if L = D (i.e. identity forward_map)
        if not self.D == self.L:
            hess_diag = np.sum(hess_diag, axis=2)  # (N, D)

        return hess_diag

    def evaluate_all_forward_map(
        self,
        Theta: np.ndarray,
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        assert len(Theta.shape) == 2 and Theta.shape[1] == self.D
        forward_map_evals = self.forward_map.compute_all(
            Theta, True, True, compute_derivatives, compute_derivatives_2nd_order
        )
        return forward_map_evals

    def evaluate_all_nll_utils(
        self,
        forward_map_evals: dict,
        idx: Optional[int] = None,
        compute_derivatives: bool = False,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        nll_utils = {}
        return nll_utils
