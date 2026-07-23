"""Implementation of log-normal likelihood"""

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
        pixelwise: bool = False,
        full: bool = False,
        idx_pix: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        # TODO: there are a few steps to be clarified in there
        # TODO: (what is the point of the reformatting step in here)
        # a priori, sam echange expected
        if idx_pix is None:
            N_pix = self.N * 1
            logy = self.logy * 1
            sigma = self.sigma * 1
        else:
            n_pix = idx_pix.size
            k_mtm = self.forward_map_evals["f_Var"].shape[0] // n_pix
            N_pix = self.forward_map_evals["f_Var"].shape[0]

            logy = np.zeros((n_pix, k_mtm, self.L))
            sigma = np.zeros((n_pix, k_mtm, self.L))

            for i_pix in range(n_pix):
                logy[i_pix, :, :] = self.logy[idx_pix[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                sigma[i_pix, :, :] = self.sigma[idx_pix[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            logy = logy.reshape((N_pix, self.L))
            sigma = sigma.reshape((N_pix, self.L))

        nlpdf = logy + (logy - self.forward_map_evals["log_f_Var"]) ** 2 / (
            2 * sigma**2
        )  # (N_pix, L)

        if full:
            return nlpdf  # (N_pix, L)

        if pixelwise:
            return np.sum(nlpdf, axis=1)  # (N_pix,)

        return np.sum(nlpdf)  # float

    def gradient_neglog_pdf(self) -> np.ndarray:
        grad_ = (
            self.forward_map_evals["grad_log_f_Var"]
            * ((self.forward_map_evals["log_f_Var"] - self.logy) / self.sigma**2)[
                :, None, :
            ]
        )  # (N, D, L)

        # ! issue: do not sum over L if L = D (i.e. identity forward_map)
        if not self.D == self.L:
            grad_ = np.sum(grad_, axis=2)  # (N, D)

        return grad_

    def hess_diag_neglog_pdf(self) -> np.ndarray:
        r"""Hessian w.r.t to the parameter of the log-normal distribution.

        Parameters
        ----------

        Returns
        -------
        np.ndarray of shape (N, D)
            [description]
        """
        hess_diag = (1 / self.sigma**2)[:, None, :] * (
            self.forward_map_evals["grad_log_f_Var"] ** 2
            + self.forward_map_evals["hess_diag_log_f_Var"]
            * (self.forward_map_evals["f_Var"] - self.logy)[:, None, :]
        )

        # ! issue: do not sum over L if L = D (i.e. identity forward_map)
        if not self.D == self.L:
            hess_diag = np.sum(hess_diag, axis=2)  # (N, D)

        return hess_diag

    def evaluate_all_forward_map(
        self,
        Var: np.ndarray,
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        assert len(Var.shape) == 2 and Var.shape[1] == self.D
        self.forward_map_evals = self.forward_map.compute_all(
            Var, True, True, compute_derivatives, compute_derivatives_2nd_order
        )
        return self.forward_map_evals

    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[np.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
    ) -> None:
        self.nlpdf_utils = {}
