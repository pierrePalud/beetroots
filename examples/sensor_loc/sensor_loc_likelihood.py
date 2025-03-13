"""Implementation of the likelihood of the sensor localization problem
"""
from typing import Optional, Union

import numpy as np

from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood


class SensorLocalizationLikelihood(Likelihood):
    """Class implementing the sensor localization likelihood."""

    __slots__ = (
        "forward_map",
        "D",
        "L",
        "N",
        "y",
        "K",
        "mask",
        "sigma",
        "R",
    )

    def __init__(
        self,
        forward_map,
        N: int,
        L: int,
        y: np.ndarray,
        sigma: Union[float, np.ndarray],
        R: float,
    ) -> None:
        """Constructor of the GaussianLikelihood object.

        Parameters
        ----------
        forward_map : ForwardMap instance
            forward map
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
        D = 2
        self.K = L - N
        L = N + self.K
        super().__init__(forward_map, D, L, N, y)

        # ! trigger an error is the mean y contains less than N elements
        if not y.shape == (N, L):
            raise ValueError(
                "y must have the shape (N, L) = ({}, {}) elements".format(
                    self.N, self.L
                )
            )

        self.mask = self.y >= 0  # (N, L) # (1 is detected, 0 is censored)
        assert isinstance(self.mask, np.ndarray)

        assert isinstance(sigma, float)
        self.sigma = sigma

        assert isinstance(R, float)
        self.R = R

    def neglog_pdf(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        L = self.y.shape[1]

        if idx is None:
            assert forward_map_evals["f_Theta"].shape[0] == self.N
            N_pix = self.N * 1
            y = self.y * 1
            # sigma = self.sigma * 1
            mask = self.mask * 1
        else:
            N_pix = forward_map_evals["f_Theta"].shape[0]
            y = self.y[idx, :][None, :] * np.ones((N_pix, self.L))
            # sigma = self.sigma[idx, :][None, :] * np.ones((N_pix, self.L))
            mask = self.mask[idx, :][None, :] * np.ones((N_pix, self.L))

        p_0 = np.exp(-0.5 * (forward_map_evals["f_Theta"] / self.R) ** 2)
        # print(np.where(1 - mask, p_0, -1))

        assert mask.shape == (
            N_pix,
            L,
        ), f"mask.shape is {mask.shape}, should be (N_pix, L) = ({N_pix}, {L})"
        assert forward_map_evals["f_Theta"].shape == (
            N_pix,
            L,
        ), f"f_Theta.shape is {forward_map_evals['f_Theta'].shape}, should be (N_pix, L) = ({N_pix}, {L})"

        nlpdf = np.where(
            mask,  # (N_pix, L)  # (1 is detected, 0 is censored)
            # detected
            forward_map_evals["f_Theta"] ** 2 / (2 * self.R**2)
            + (y - forward_map_evals["f_Theta"]) ** 2 / (2 * self.sigma**2),
            # not detected
            -np.log(1 - p_0),
        )  # (N_pix, L)

        assert np.sum(np.isnan(nlpdf)) == 0
        if full:
            return nlpdf
        if pixelwise:
            return np.sum(nlpdf, axis=1)  # (N_pix,)

        return np.sum(nlpdf)  # float

    def neglog_pdf_candidates(
        self, candidates: np.ndarray, idx: np.ndarray, Theta_t: np.ndarray
    ) -> np.ndarray:
        assert len(candidates.shape) == 2 and candidates.shape[1] == self.D
        assert isinstance(idx, np.ndarray) and idx.size == 1

        n = idx[0]
        assert 0 <= n <= self.N - 1, n

        N_candidates = candidates.shape[0]
        forward_map_evals = {
            "f_Theta": self.forward_map.evaluate_candidates_one_n(
                candidates,
                Theta_t,
                n,
            )
        }
        nll_utils = self.evaluate_all_nll_utils(forward_map_evals, n)

        nll_candidates = self.neglog_pdf(
            forward_map_evals,
            nll_utils,
            pixelwise=True,
            idx=n,
        )
        assert isinstance(nll_candidates, np.ndarray)
        assert nll_candidates.shape == (N_candidates,)

        return nll_candidates

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
        b_ell = np.ones((self.N, self.L))
        b_ell[:, : self.N] = 2.0

        p_0 = np.exp(-0.5 * (forward_map_evals["f_Theta"] / self.R) ** 2)

        grad_ = (
            b_ell[:, None, :]  # (N, D, L)
            * forward_map_evals["grad_f_Theta"]  # (N, D, L)
            * np.where(
                self.mask,
                # pair detected
                forward_map_evals["f_Theta"] / self.R**2
                + (forward_map_evals["f_Theta"] - self.y) / self.sigma**2,
                # pair not detected
                p_0 / (1 - p_0) * forward_map_evals["f_Theta"] / self.R**2,
            )[:, None, :]
        )  # (N, D, L)

        return np.sum(grad_, axis=2)  # (N, D)

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
        np.ndarray of shape (N, D)
            [description]
        """
        b_ell = np.ones((self.N, self.L))
        b_ell[:, : self.N] = 2.0

        p_0 = np.exp(-0.5 * (forward_map_evals["f_Theta"] / self.R) ** 2)  # (N, L)

        hess_diag = b_ell[:, None, :] * (
            np.where(
                self.mask[:, None, :] * np.ones((self.N, self.D, self.L)),
                # detected
                forward_map_evals["grad_f_Theta"] ** 2
                * (1 / self.R**2 + 1 / self.sigma**2)
                + forward_map_evals["hess_diag_f_Theta"]
                * (
                    forward_map_evals["f_Theta"] / self.R**2
                    + (forward_map_evals["f_Theta"] - self.y) / self.sigma**2
                )[:, None, :],
                # not detected
                (p_0 / (1 - p_0))[:, None, :]
                / self.R**2
                * (
                    forward_map_evals["hess_diag_f_Theta"]
                    * forward_map_evals["f_Theta"][:, None, :]
                    + forward_map_evals["grad_f_Theta"] ** 2
                    - (
                        forward_map_evals["grad_f_Theta"]
                        * (forward_map_evals["f_Theta"] / self.R)[:, None, :]
                    )
                    ** 2
                    / (1 - p_0)[:, None, :]
                ),
            )
        )
        return np.sum(hess_diag, axis=2)  # (N, D)

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
    ):
        eps = self.sigma * rng.standard_normal(
            size=forward_map_evals["f_Theta"].shape,
        )
        y_rep = forward_map_evals["f_Theta"] + eps
        y_rep = np.where(self.mask == 1, y_rep, 0.0)
        return y_rep
