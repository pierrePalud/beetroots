"""Implementation of an approximation of the likelihood function of a mixture of Gaussian and multiplicative noises with censorship with a lower limit.
"""
from typing import List, Optional, Union

import numba
import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from beetroots.modelling.likelihoods import utils
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood


@numba.njit()
def cost_au(
    y: np.ndarray, f_Theta: np.ndarray, m_a: np.ndarray, s_a: np.ndarray
) -> np.ndarray:
    nll_au = 0.5 * ((y - f_Theta - m_a) / s_a) ** 2 + np.log(s_a)
    return nll_au


@numba.njit()
def cost_mu(
    log_y: np.ndarray, log_f_Theta: np.ndarray, m_m: np.ndarray, s_m: np.ndarray
) -> np.ndarray:
    nll_mu = 0.5 * ((log_y - log_f_Theta - m_m) / s_m) ** 2 + np.log(s_m)
    # nll_mu = np.where(np.isnan(log_y) | np.isinf(log_y), 10.0 ** 8, nll_mu)

    # shape = nll_mu.shape
    # nll_mu = nll_mu.ravel()
    # nll_mu[np.isnan(nll_mu)] = 0
    # nll_mu = nll_mu.reshape(shape)
    return nll_mu


@numba.njit()
def gradient_cost_au(
    y: np.ndarray,
    grad_f_Theta: np.ndarray,
    f_Theta: np.ndarray,
    grad_m_a: np.ndarray,
    m_a: np.ndarray,
    grad_s_a2: np.ndarray,
    s_a2: np.ndarray,
) -> np.ndarray:
    u_1 = (f_Theta - m_a - y) / s_a2**2
    assert u_1.shape == f_Theta.shape  # (N, L)

    u_2 = (
        2 * np.expand_dims(s_a2, axis=1) * (grad_f_Theta + grad_m_a)
        - np.expand_dims(f_Theta - m_a - y, axis=1) * grad_s_a2
    )
    assert u_2.shape == grad_f_Theta.shape  # (N, D, L)

    g_au = 0.5 * grad_s_a2 / np.expand_dims(s_a2, axis=1)
    g_au += 0.5 * np.expand_dims(u_1, axis=1) * u_2

    assert g_au.shape == grad_f_Theta.shape
    return g_au


@numba.njit()
def gradient_cost_mu(
    log_y: np.ndarray,
    grad_log_f_Theta: np.ndarray,
    log_f_Theta: np.ndarray,
    grad_m_m: np.ndarray,
    m_m: np.ndarray,
    grad_s_m2: np.ndarray,
    s_m2: np.ndarray,
) -> np.ndarray:

    u_1 = (log_f_Theta - m_m - log_y) / s_m2**2
    assert u_1.shape == log_f_Theta.shape  # (N, L)

    u_2 = (
        2 * np.expand_dims(s_m2, axis=1) * (grad_log_f_Theta + grad_m_m)
        - np.expand_dims(log_f_Theta - m_m - log_y, axis=1) * grad_s_m2
    )
    assert u_2.shape == grad_log_f_Theta.shape  # (N, D, L)

    g_mu = 0.5 * grad_s_m2 / np.expand_dims(s_m2, axis=1)
    g_mu += np.where(
        np.expand_dims(np.isnan(log_y), axis=1) + np.zeros_like(grad_log_f_Theta),
        # fake output
        np.zeros_like(grad_log_f_Theta),
        # true grad
        0.5 * np.expand_dims(u_1, axis=1) * u_2,
    )

    assert g_mu.shape == grad_log_f_Theta.shape
    return g_mu


class MixingModelsLikelihood(Likelihood):
    r"""Class implementing a Gaussian likelihood model with lower censorship. This likelihood function is introduced in Section II.C of :cite:t:`paludEfficientSamplingNon2023`.

    Note
    ----
    This likelihood is a parametric approximation of the true likelihood model.
    The associated parameter, denoted :math:`a_\ell` in the article (as there is one such parameter per observable :math:`\ell `), should be adjusted before any inversion.
    To adjust this parameter, see the ``beetroots.approx_optim`` subpackage.
    """

    __slots__ = (
        "N",
        "L",
        "D",
        "y",
        "forward_map",
        "log_y",
        "sigma_a",
        "sigma_m",
        "omega",
        "log_fm1",
        "log_fp1",
        "P_lambda",
        "grad_P_lambda",
        "hess_diag_P_lambda",
    )

    def __init__(
        self,
        forward_map,
        D: int,
        L: int,
        N: int,
        y: np.ndarray,
        sigma_a: Union[float, np.ndarray],
        sigma_m: Union[float, np.ndarray],
        omega: Union[float, np.ndarray],
        path_transition_params: str,
        list_lines_fit: List[str],
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
        sigma_a : float or np.ndarray of shape (N, L)
            standard deviation of the Gaussian distribution
        sigma_m : float or np.ndarray of shape (N, L)
            scale parameter of the lognormal distribution
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
        if isinstance(sigma_a, (float, int)):
            self.sigma_a = sigma_a * np.ones((N, L))
        else:
            assert sigma_a.shape == (N, L)
            self.sigma_a = sigma_a

        if isinstance(sigma_m, (float, int)):
            self.sigma_m = sigma_m * np.ones((N, L))
        else:
            assert sigma_m.shape == (N, L)
            self.sigma_m = sigma_m

        if isinstance(omega, (float, int)):
            self.omega = omega * np.ones((N, L))
        else:
            assert omega.shape == (N, L)
            self.omega = omega

        self.log_y = np.log(y)

        df_transition = pd.read_csv(path_transition_params)
        num_distinct_n = np.unique(df_transition["n"]).size

        if num_distinct_n == 1:
            # if only one n value, use the transition params for all pixels
            df_transition = df_transition.set_index("line")

            transition_center = (
                df_transition.loc[list_lines_fit, "a0_best"].values[None, :]
                * np.ones((self.N, self.L))
                * np.log(10)
            )
            transition_radius = (
                df_transition.loc[list_lines_fit, "a1_best"].values[None, :]
                * np.ones((self.N, self.L))
                * np.log(10)
            )

        else:
            # assert (
            #     num_distinct_n == self.N
            # ), f"the transition file {path_transition_params} should have {self.N} or 1 distinct values of n, and has {num_distinct_n}"

            df_transition = df_transition.set_index(["n", "line"])
            index = pd.MultiIndex.from_product(
                [list(range(self.N)), list_lines_fit],
                names=["n", "line"],
            )
            # df_transition

            transition_center = df_transition.loc[index, "a0_best"].values.reshape(
                (self.N, self.L)
            ) * np.log(10)
            transition_radius = df_transition.loc[index, "a1_best"].values.reshape(
                (self.N, self.L)
            ) * np.log(10)

        # for col in ["ell", "a0_best", "a1_best", "target_best"]:
        #     assert col in list(df_transition.columns)

        # df_transition = df_transition.set_index("ell")
        # df_transition = df_transition.sort_index()
        # transition_center = (
        #     df_transition.loc[:, "a0_best"].values[None, :]
        #     * np.ones((self.N, self.L))
        #     * np.log(10)
        # )
        # transition_radius = (
        #     df_transition.loc[:, "a1_best"].values[None, :]
        #     * np.ones((self.N, self.L))
        #     * np.log(10)
        # )

        self.log_fm1 = transition_center - transition_radius  # (N, L)
        self.log_fp1 = transition_center + transition_radius  # (N, L)

        self.P_lambda = np.poly1d(np.array([-6.0, 15.0, -10.0, 0.0, 0.0, 1.0]))
        self.grad_P_lambda = self.P_lambda.deriv(m=1)
        self.hess_diag_P_lambda = self.P_lambda.deriv(m=2)

    def sample_observation_model(
        self,
        forward_map_evals: dict,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        eps_a = rng.normal(loc=0.0, scale=self.sigma_a)
        eps_m = rng.lognormal(
            mean=-(self.sigma_m**2) / 2,
            sigma=self.sigma_m,
        )
        f_Theta = forward_map_evals["f_Theta"] * 1
        y_rep = np.maximum(self.omega, eps_m * f_Theta + eps_a)
        return y_rep

    def model_mixing_param(
        self,
        forward_map_evals: dict,
        idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""computes the weight of the additive model :math:`\lambda_{n, \ell}` (line-wise and pixel-wise). In this model, :math:`\lambda_{n, \ell}` is a functino of the observation :math:`y_{n, \ell}`, and therefore constant during the sampling

        .. math::
            \lambda_{n, \ell} = \frac{\sigma_a^{2b}}{\sigma_a^{2b} + (a \sigma_m y_{n, \ell})^{2b}}

        with :math:`a` a transition location parameter and :math:`b` a transition speed parameter
        """
        if idx is None:
            N_pix = self.N * 1
            log_fm1 = self.log_fm1 * 1
            log_fp1 = self.log_fp1 * 1
            # sigma_a = self.sigma_a * 1
            # sigma_m = self.sigma_m * 1
        else:
            n_pix = idx.size
            k_mtm = forward_map_evals["f_Theta"].shape[0] // n_pix
            N_pix = forward_map_evals["f_Theta"].shape[0]

            log_fm1 = np.zeros((n_pix, k_mtm, self.L))
            log_fp1 = np.zeros((n_pix, k_mtm, self.L))

            for i_pix in range(n_pix):
                log_fm1[i_pix, :, :] = self.log_fm1[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                log_fp1[i_pix, :, :] = self.log_fp1[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            log_fm1 = log_fm1.reshape((N_pix, self.L))
            log_fp1 = log_fp1.reshape((N_pix, self.L))

            # N_pix = forward_map_evals["f_Theta"].shape[0]

            # log_fm1 = np.zeros((N_pix, self.L))
            # log_fp1 = np.zeros((N_pix, self.L))
            # for i in range(idx.size):
            #     log_fm1[i, :] = self.log_fm1[idx[i], :] * 1
            #     log_fp1[i, :] = self.log_fp1[idx[i], :] * 1

            # log_fp1 = self.log_fp1[idx, :][None, :] * np.ones((N_pix, self.L))

            # sigma_a = self.sigma_a[idx, :][None, :] * np.ones((N_pix, self.L))
            # sigma_m = self.sigma_m[idx, :][None, :] * np.ones((N_pix, self.L))

        # lambda_ = 1 / (
        #     1
        #     + np.exp(
        #         (2 * self.transition_speed)
        #         * np.log(
        #             (self.transition_loc * sigma_m * forward_map_evals["f_Theta"]) / sigma_a
        #         )
        #     )
        # )
        lambda_ = np.where(
            forward_map_evals["log_f_Theta"] <= log_fm1,
            1,
            np.where(
                forward_map_evals["log_f_Theta"] >= log_fp1,
                0,
                self.P_lambda(
                    (forward_map_evals["log_f_Theta"] - log_fm1) / (log_fp1 - log_fm1)
                ),
            ),
        )
        return lambda_  # (N, L)

    def grad_model_mixing_param(self, forward_map_evals: dict) -> np.ndarray:
        r"""[summary]

        Parameters
        ----------
        lambda_ : np.ndarray of shape (N, L)
            [description]
        f_Theta : np.ndarray of shape (N, L)
            [description]
        grad_f_Theta : np.ndarray of shape (N, D, L)
            [description]

        Returns
        -------
        np.ndarray of shape (N, D, L)
            [description]
        """
        # grad_ = (
        #     (-2 * self.transition_speed)
        #     * (
        #         (
        #             (self.transition_loc * self.sigma_m / self.sigma_a)
        #             ** (2 * self.transition_speed)
        #         )
        #         * (
        #             forward_map_evals["f_Theta"] ** (2 * self.transition_speed - 1)
        #             * lambda_ ** 2
        #         )
        #     )[:, None, :]
        #     * forward_map_evals["grad_f_Theta"]
        # )
        u = (forward_map_evals["log_f_Theta"] - self.log_fm1) / (
            self.log_fp1 - self.log_fm1
        )  # (N, L)
        grad_u = (
            forward_map_evals["grad_log_f_Theta"]
            / (self.log_fp1 - self.log_fm1)[:, None, :]
        )  # (N, D, L)
        # assert u.shape == (self.N, self.L)
        # assert grad_u.shape == (self.N, self.D, self.L), grad_u.shape

        grad_ = np.where(
            (
                (forward_map_evals["log_f_Theta"] <= self.log_fm1)
                | (forward_map_evals["log_f_Theta"] >= self.log_fp1)
            )[:, None, :],
            np.zeros((self.N, self.D, self.L)),
            grad_u * self.grad_P_lambda(u)[:, None, :],
        )
        assert grad_.shape == (self.N, self.D, self.L), grad_.shape

        return grad_  # (N, D, L)

    def hess_diag_model_mixing_param(self, forward_map_evals: dict) -> np.ndarray:
        r"""[summary]

        Parameters
        ----------
        lambda_ : np.ndarray of shape (N, L)
            [description]
        grad_lambda_ : np.ndarray of shape (N, D, L)
            [description]
        f_Theta : np.ndarray of shape (N, L)
            [description]
        grad_f_Theta : np.ndarray of shape (N, D, L)
            [description]
        hess_diag_f_Theta : np.ndarray of shape (N, D, L)
            [description]

        Returns
        -------
        np.ndarray of shape (N, D, L)
            [description]
        """
        # hess_diag = (
        #     (-2 * self.transition_speed)
        #     * (self.transition_loc * self.sigma_m / self.sigma_a)
        #     ** (2 * self.transition_speed)
        # )[:, None, :] * (
        #     (
        #         forward_map_evals["f_Theta"] ** (2 * self.transition_speed - 1)
        #         * lambda_ ** 2
        #     )[:, None, :]
        #     * forward_map_evals["hess_diag_f_Theta"]
        #     + (2 * self.transition_speed - 1)
        #     * (
        #         forward_map_evals["grad_f_Theta"] ** 2
        #         * (
        #             lambda_ ** 2
        #             * forward_map_evals["f_Theta"] ** (2 * self.transition_speed - 2)
        #         )[:, None, :]
        #     )
        #     + 2
        #     * (forward_map_evals["f_Theta"] ** (2 * self.transition_speed - 1) * lambda_)[
        #         :, None, :
        #     ]
        #     * forward_map_evals["grad_f_Theta"]
        #     * grad_lambda_
        # )

        u = (forward_map_evals["log_f_Theta"] - self.log_fm1) / (
            self.log_fp1 - self.log_fm1
        )  # (N, L)
        grad_u = (
            forward_map_evals["grad_log_f_Theta"]
            / (self.log_fp1 - self.log_fm1)[:, None, :]
        )  # (N, D, L)
        hess_diag_u = (
            forward_map_evals["hess_diag_log_f_Theta"]
            / (self.log_fp1 - self.log_fm1)[:, None, :]
        )  # (N, D, L)

        hess_diag = np.where(
            (
                (forward_map_evals["log_f_Theta"] <= self.log_fm1)
                | (forward_map_evals["log_f_Theta"] >= self.log_fp1)
            )[:, None, :],
            np.zeros((self.N, self.D, self.L)),
            (
                grad_u**2 * self.hess_diag_P_lambda(u)[:, None, :]
                + hess_diag_u * self.grad_P_lambda(u)[:, None, :]
            ),
        )
        assert hess_diag.shape == (self.N, self.D, self.L), hess_diag.shape
        return hess_diag  # (N, D, L)

    def _compute_bias_and_std(
        self,
        forward_map_evals: dict,
        idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""computes the biases and std of additive and multiplicative approximations

        Parameters
        ----------
        f_Theta : numpy.array of shape (N_pix, L)
            image of the current iterate through the forward model
        sigma_a : np

        Returns
        -------
        current_bias_std : np.ndarray of shape (4, N_pix, L)
            array with (m_a, s_a, m_m, s_m)
        """
        if idx is None:
            N_pix = self.N * 1
            sigma_a = self.sigma_a * 1
            sigma_m = self.sigma_m * 1

        else:
            N_pix = forward_map_evals["f_Theta"].shape[0]

            sigma_a = np.zeros((N_pix, self.L))
            sigma_m = np.zeros((N_pix, self.L))
            for i in range(idx.size):
                sigma_a[i, :] = self.sigma_a[idx[i], :] * 1
                sigma_m[i, :] = self.sigma_m[idx[i], :] * 1

        log_combination = (
            np.log(sigma_a) - forward_map_evals["log_f_Theta"] - (sigma_m**2) / 2
        )

        assert sigma_a.min() > 0, sigma_a.min()
        assert sigma_m.min() > 0, sigma_m.min()
        assert np.sum(np.isnan(forward_map_evals["log_f_Theta"])) == 0, np.sum(
            np.isnan(forward_map_evals["log_f_Theta"])
        )
        assert np.sum(np.isnan(log_combination)) == 0, np.sum(np.isnan(log_combination))

        # * computation of bias and variances
        m_a = (np.exp(sigma_m**2 / 2) - 1) * forward_map_evals["f_Theta"]
        s_a = sigma_a * np.sqrt(
            (np.exp(sigma_m**2) - 1) * np.exp(-2 * log_combination) + 1
        )

        m_m = -0.5 * np.log(1 + np.exp(2 * log_combination))
        s_m = np.sqrt(sigma_m**2 - 2 * m_m)

        assert s_m.min() > 0, f"{s_m.min()}, {(sigma_m ** 2 - 2 * m_m).min()}"

        # gather all in one array
        N_pix = forward_map_evals["f_Theta"].shape[0]
        current_bias_std = np.zeros((4, N_pix, self.L))
        current_bias_std[0] = m_a
        current_bias_std[1] = s_a
        current_bias_std[2] = m_m
        current_bias_std[3] = s_m

        return current_bias_std

    def neglog_pdf(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:

        nlpdf = nll_utils["lambda_"] * np.where(
            nll_utils["censored_mask"], nll_utils["nll_ac"], nll_utils["nll_au"]
        ) + (1 - nll_utils["lambda_"]) * np.where(
            nll_utils["censored_mask"], nll_utils["nll_mc"], nll_utils["nll_mu"]
        )  # (N, L)

        nlpdf = np.nan_to_num(nlpdf)
        nlpdf -= np.log(nll_utils["sigma_a"]) + np.log(nll_utils["sigma_m"])
        # nlpdf /= self.N * self.L
        # nll_utils["censored_mask"].size = self.N * self.L if standard eval
        # nll_utils["censored_mask"].size = N_candidates * self.L if candidates
        if full:
            return nlpdf  # (N, L)

        if pixelwise:
            sum_ = np.sum(nlpdf, axis=1)  # (N,)
            assert sum_.size == forward_map_evals["f_Theta"].shape[0]
            return sum_

        return nlpdf.sum()

    def neglog_pdf_ac(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        omega: np.ndarray,
    ) -> np.ndarray:
        z = (omega - forward_map_evals["f_Theta"] - nll_utils["m_a"]) / nll_utils["s_a"]
        # z = np.nan_to_num(z)

        nll_ac = -log_ndtr(z)
        nll_ac = np.nan_to_num(nll_ac)
        return nll_ac

    def neglog_pdf_au(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        y: np.ndarray,
    ) -> np.ndarray:
        nll_au = cost_au(
            y=y,
            f_Theta=forward_map_evals["f_Theta"],
            m_a=nll_utils["m_a"],
            s_a=nll_utils["s_a"],
        )
        # add a constant to `nll_au` to ensure that it is positive
        # nll_au -= np.log(nll_utils["sigma_a"])
        # nll_au = np.nan_to_num(nll_au)
        return nll_au

    def neglog_pdf_mc(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        log_omega: np.ndarray,
    ) -> np.ndarray:
        z = log_omega - forward_map_evals["log_f_Theta"] - nll_utils["m_m"]
        z /= nll_utils["s_m"]
        # z = np.nan_to_num(z)

        nll_mc = -log_ndtr(z)
        nll_mc = np.nan_to_num(nll_mc)
        return nll_mc

    def neglog_pdf_mu(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        log_y: np.ndarray,
    ) -> np.ndarray:
        nll_mu = cost_mu(
            log_y=log_y,
            log_f_Theta=forward_map_evals["log_f_Theta"],
            m_m=nll_utils["m_m"],
            s_m=nll_utils["s_m"],
        )
        # nll_mu -= np.log(nll_utils["sigma_m"])
        return nll_mu

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
        np.ndarray of shape (N, D, L)
            [description]
        """
        grad_ = np.where(
            (self.y == self.omega)[:, None, :],
            # censored
            # grad_nll_ac,
            nll_utils["lambda_"][:, None, :] * nll_utils["grad_nll_ac"]
            + nll_utils["grad_lambda_"] * nll_utils["nll_ac"][:, None, :]
            + (1 - nll_utils["lambda_"])[:, None, :] * nll_utils["grad_nll_mc"]
            - nll_utils["grad_lambda_"] * nll_utils["nll_mc"][:, None, :],
            # uncensored
            nll_utils["lambda_"][:, None, :] * nll_utils["grad_nll_au"]
            + nll_utils["grad_lambda_"] * nll_utils["nll_au"][:, None, :]
            + (1 - nll_utils["lambda_"])[:, None, :] * nll_utils["grad_nll_mu"]
            - nll_utils["grad_lambda_"] * nll_utils["nll_mu"][:, None, :],
        )  # (N, D, L)
        # grad_ = np.where(
        #     np.isfinite(grad_),
        #     grad_,
        #     np.abs(grad_).max(axis=0)[None, :, :],
        # )  # (N, D, L)
        grad_ = np.nan_to_num(grad_)
        return np.sum(grad_, axis=2)  # / (self.N * self.L)

    def gradient_neglog_pdf_ac(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        f_Theta_m_a_omega = forward_map_evals["f_Theta"] + nll_utils["m_a"] - self.omega

        u_1 = utils.norm_pdf_cdf_ratio(-f_Theta_m_a_omega / nll_utils["s_a"])  # (N, L)

        u_2 = (1 / nll_utils["s_a2"])[:, None, :] * (
            (forward_map_evals["grad_f_Theta"] + nll_utils["grad_m_a"])
            * nll_utils["s_a"][:, None, :]
            - f_Theta_m_a_omega[:, None, :] * nll_utils["grad_s_a"]
        )  # (N, D, L)

        grad_ = u_1[:, None, :] * u_2  # (N, D, L)
        # grad_ = np.nan_to_num(grad_)
        # assert np.sum(np.isnan(grad_)) == 0, f"grad_ac : {np.sum(np.isnan(grad_))}"
        return grad_

    def gradient_neglog_pdf_au(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        g_au = gradient_cost_au(
            y=self.y,
            grad_f_Theta=forward_map_evals["grad_f_Theta"],
            f_Theta=forward_map_evals["f_Theta"],
            grad_m_a=nll_utils["grad_m_a"],
            m_a=nll_utils["m_a"],
            grad_s_a2=nll_utils["grad_s_a2"],
            s_a2=nll_utils["s_a2"],
        )
        # g_au = np.nan_to_num(g_au)
        return g_au

    def gradient_neglog_pdf_mc(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        log_f_Theta_m_m_log_omega = (
            forward_map_evals["log_f_Theta"] + nll_utils["m_m"] - np.log(self.omega)
        )

        u_1 = utils.norm_pdf_cdf_ratio(
            -log_f_Theta_m_m_log_omega / nll_utils["s_m"]
        )  # (N, L)
        u_2 = (1 / nll_utils["s_m2"])[:, None, :] * (
            (forward_map_evals["grad_log_f_Theta"] + nll_utils["grad_m_m"])
            * nll_utils["s_m"][:, None, :]
            - log_f_Theta_m_m_log_omega[:, None, :] * nll_utils["grad_s_m"]
        )  # (N, D, L)

        grad_ = u_1[:, None, :] * u_2  # (N, D, L)
        # grad_ = np.nan_to_num(grad_)
        # assert np.sum(np.isnan(grad_)) == 0, f"grad_ac : {np.sum(np.isnan(grad_))}"
        return grad_

    def gradient_neglog_pdf_mu(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        g_mu = gradient_cost_mu(
            log_y=self.log_y,
            grad_log_f_Theta=forward_map_evals["grad_log_f_Theta"],
            log_f_Theta=forward_map_evals["log_f_Theta"],
            grad_m_m=nll_utils["grad_m_m"],
            m_m=nll_utils["m_m"],
            grad_s_m2=nll_utils["grad_s_m2"],
            s_m2=nll_utils["s_m2"],
        )
        # g_mu = np.nan_to_num(g_mu)
        return g_mu

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
            # censored
            # hess_diag_ac,
            (
                nll_utils["lambda_"][:, None, :] * nll_utils["hess_diag_ac"]
                + nll_utils["hess_diag_lambda_"] * nll_utils["nll_ac"][:, None, :]
                + 2 * nll_utils["grad_lambda_"] * nll_utils["grad_nll_ac"]
                #
                + (1 - nll_utils["lambda_"])[:, None, :] * nll_utils["hess_diag_mc"]
                - nll_utils["hess_diag_lambda_"] * nll_utils["nll_mc"][:, None, :]
                - 2 * nll_utils["grad_lambda_"] * nll_utils["grad_nll_mc"]
            ),
            # uncensored
            (
                nll_utils["lambda_"][:, None, :] * nll_utils["hess_diag_au"]
                + nll_utils["hess_diag_lambda_"] * nll_utils["nll_au"][:, None, :]
                + 2 * nll_utils["grad_lambda_"] * nll_utils["grad_nll_au"]
                #
                + (1 - nll_utils["lambda_"])[:, None, :] * nll_utils["hess_diag_mu"]
                - nll_utils["hess_diag_lambda_"] * nll_utils["nll_mu"][:, None, :]
                - 2 * nll_utils["grad_lambda_"] * nll_utils["grad_nll_mu"]
            ),
        )  # (N, D, L)
        # hess_diag = np.where(
        #     np.isfinite(hess_diag),
        #     hess_diag,
        #     np.abs(hess_diag).max(axis=0)[None, :, :],
        # )  # (N, D, L)
        hess_diag = np.nan_to_num(hess_diag)
        return np.sum(hess_diag, axis=2)  # / (self.N * self.L)

    def hess_diag_neglog_pdf_ac(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        f_Theta_m_a_omega = forward_map_evals["f_Theta"] + nll_utils["m_a"] - self.omega
        grad_f_Theta_grad_m_a = (
            forward_map_evals["grad_f_Theta"] + nll_utils["grad_m_a"]
        )

        u_1 = utils.norm_pdf_cdf_ratio(-f_Theta_m_a_omega / nll_utils["s_a"])  # (N, L)
        assert u_1.shape == (self.N, self.L)

        u_2 = (1 / nll_utils["s_a2"])[:, None, :] * (
            grad_f_Theta_grad_m_a * nll_utils["s_a"][:, None, :]
            - f_Theta_m_a_omega[:, None, :] * nll_utils["grad_s_a"]
        )  # (N, D, L)
        assert u_2.shape == (self.N, self.D, self.L)

        grad_u_1 = (
            u_2 * (u_1 * (-f_Theta_m_a_omega / nll_utils["s_a"] + u_1))[:, None, :]
        )  # (N, D, L)
        assert grad_u_1.shape == (self.N, self.D, self.L)

        grad_u_2 = (1 / nll_utils["s_a2"] ** 2)[:, None, :] * (
            nll_utils["s_a2"][:, None, :]
            * (
                nll_utils["s_a"][:, None, :]
                * (forward_map_evals["hess_diag_f_Theta"] + nll_utils["hess_diag_m_a"])
                - f_Theta_m_a_omega[:, None, :] * nll_utils["hess_diag_s_a"]
            )
            - nll_utils["grad_s_a2"]
            * (
                nll_utils["s_a"][:, None, :] * grad_f_Theta_grad_m_a
                - f_Theta_m_a_omega[:, None, :] * nll_utils["grad_s_a"]
            )
        )  # (N, D, L)
        assert grad_u_2.shape == (self.N, self.D, self.L)

        hess_diag = grad_u_1 * u_2 + u_1[:, None, :] * grad_u_2
        # hess_diag = np.nan_to_num(hess_diag)
        return hess_diag

    def hess_diag_neglog_pdf_au(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        f_Theta_m_a_y = (
            forward_map_evals["f_Theta"] + nll_utils["m_a"] - self.y
        )  # (N, L)
        grad_f_Theta_grad_m_a = (
            forward_map_evals["grad_f_Theta"] + nll_utils["grad_m_a"]
        )  # (N, D, L)

        u_1 = f_Theta_m_a_y / nll_utils["s_a2"] ** 2  # (N, L)

        grad_u_1 = (1 / nll_utils["s_a2"] ** 4)[:, None, :] * (
            grad_f_Theta_grad_m_a * (nll_utils["s_a2"] ** 2)[:, None, :]
            - 2
            * (f_Theta_m_a_y * nll_utils["s_a2"])[:, None, :]
            * nll_utils["grad_s_a2"]
        )  # (N, D, L)

        u_2 = (
            2 * nll_utils["s_a2"][:, None, :] * grad_f_Theta_grad_m_a
            - f_Theta_m_a_y[:, None, :] * nll_utils["grad_s_a2"]
        )  # (N, D, L)

        grad_u_2 = (
            2
            * nll_utils["s_a2"][:, None, :]
            * (forward_map_evals["hess_diag_f_Theta"] + nll_utils["hess_diag_m_a"])
            #
            + nll_utils["grad_s_a2"] * grad_f_Theta_grad_m_a
            #
            - f_Theta_m_a_y[:, None, :] * nll_utils["hess_diag_s_a2"]
        )  # (N, D, L)

        hess_diag = (
            0.5
            * (
                nll_utils["hess_diag_s_a2"] * nll_utils["s_a2"][:, None, :]
                - nll_utils["grad_s_a2"] ** 2
            )
            / (nll_utils["s_a2"] ** 2)[:, None, :]
        )  # (N, D, L)

        hess_diag += 0.5 * (
            grad_u_1 * u_2 + u_1[:, None, :] * grad_u_2
        )  # hess_diag = np.nan_to_num(hess_diag)
        return hess_diag

    def hess_diag_neglog_pdf_mc(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        log_f_Theta_m_m_log_omega = (
            forward_map_evals["log_f_Theta"] + nll_utils["m_m"] - self.omega
        )
        grad_log_f_Theta_grad_m_m = (
            forward_map_evals["grad_log_f_Theta"] + nll_utils["grad_m_m"]
        )

        u_1 = utils.norm_pdf_cdf_ratio(
            -log_f_Theta_m_m_log_omega / nll_utils["s_m"]
        )  # (N, L)
        assert u_1.shape == (self.N, self.L)

        u_2 = (1 / nll_utils["s_m2"])[:, None, :] * (
            grad_log_f_Theta_grad_m_m * nll_utils["s_m"][:, None, :]
            - log_f_Theta_m_m_log_omega[:, None, :] * nll_utils["grad_s_m"]
        )  # (N, D, L)
        assert u_2.shape == (self.N, self.D, self.L)

        grad_u_1 = (
            u_2
            * (u_1 * (-log_f_Theta_m_m_log_omega / nll_utils["s_m"] + u_1))[:, None, :]
        )  # (N, D, L)
        assert grad_u_1.shape == (self.N, self.D, self.L)

        grad_u_2 = (1 / nll_utils["s_m2"] ** 2)[:, None, :] * (
            nll_utils["s_m2"][:, None, :]
            * (
                nll_utils["s_m"][:, None, :]
                * (forward_map_evals["hess_diag_f_Theta"] + nll_utils["hess_diag_m_m"])
                - log_f_Theta_m_m_log_omega[:, None, :] * nll_utils["hess_diag_s_m"]
            )
            - nll_utils["grad_s_m2"]
            * (
                nll_utils["s_m"][:, None, :] * grad_log_f_Theta_grad_m_m
                - log_f_Theta_m_m_log_omega[:, None, :] * nll_utils["grad_s_m"]
            )
        )  # (N, D, L)
        assert grad_u_2.shape == (self.N, self.D, self.L)

        hess_diag = grad_u_1 * u_2 + u_1[:, None, :] * grad_u_2
        # hess_diag = np.nan_to_num(hess_diag)
        return hess_diag

    def hess_diag_neglog_pdf_mu(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        log_f_Theta_m_m_log_y = (
            forward_map_evals["log_f_Theta"] + nll_utils["m_m"] - self.log_y
        )  # (N, L)
        grad_log_f_Theta_grad_m_m = (
            forward_map_evals["grad_log_f_Theta"] + nll_utils["grad_m_m"]
        )  # (N, D, L)

        u_1 = log_f_Theta_m_m_log_y / nll_utils["s_m2"] ** 2  # (N, L)

        grad_u_1 = (1 / nll_utils["s_m2"] ** 4)[:, None, :] * (
            grad_log_f_Theta_grad_m_m * (nll_utils["s_m2"] ** 2)[:, None, :]
            - 2
            * (log_f_Theta_m_m_log_y * nll_utils["s_m2"])[:, None, :]
            * nll_utils["grad_s_m2"]
        )  # (N, D, L)

        u_2 = (
            2 * nll_utils["s_m2"][:, None, :] * grad_log_f_Theta_grad_m_m
            - log_f_Theta_m_m_log_y[:, None, :] * nll_utils["grad_s_m2"]
        )  # (N, D, L)

        grad_u_2 = 2 * (
            nll_utils["grad_s_m2"] * grad_log_f_Theta_grad_m_m
            + nll_utils["s_m2"][:, None, :]
            * (forward_map_evals["hess_diag_log_f_Theta"] + nll_utils["hess_diag_m_m"])
        ) - (
            nll_utils["grad_s_m2"] * grad_log_f_Theta_grad_m_m
            + log_f_Theta_m_m_log_y[:, None, :] * nll_utils["hess_diag_s_m2"]
        )  # (N, D, L)

        hess_diag = (
            0.5
            * (
                nll_utils["hess_diag_s_m2"] * nll_utils["s_m2"][:, None, :]
                - nll_utils["grad_s_m2"] ** 2
            )
            / (nll_utils["s_m2"] ** 2)[:, None, :]
        )  # (N, D, L)

        hess_diag += 0.5 * (
            grad_u_1 * u_2 + u_1[:, None, :] * grad_u_2
        )  # hess_diag = np.nan_to_num(hess_diag)
        return hess_diag

    def evaluate_all_nll_utils(
        self,
        forward_map_evals: dict,
        idx: Optional[np.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        assert (
            np.sum(np.isnan(forward_map_evals["log_f_Theta"])) == 0
        ), f"before entering bias and std : {np.sum(np.isnan(forward_map_evals['log_f_Theta']))}"

        nll_utils = {}
        # * bias and variance
        if idx is None:
            N_pix = self.N * 1
            sigma_a = self.sigma_a * 1
            sigma_m = self.sigma_m * 1
            y = self.y * 1
            log_y = self.log_y * 1
            omega = self.omega * 1
        else:
            n_pix = idx.size
            k_mtm = forward_map_evals["f_Theta"].shape[0] // n_pix
            N_pix = forward_map_evals["f_Theta"].shape[0]
            assert n_pix * k_mtm == N_pix

            sigma_a = np.zeros((n_pix, k_mtm, self.L))
            sigma_m = np.zeros((n_pix, k_mtm, self.L))
            y = np.zeros((n_pix, k_mtm, self.L))
            omega = np.zeros((n_pix, k_mtm, self.L))

            for i_pix in range(n_pix):
                sigma_a[i_pix, :, :] = self.sigma_a[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                sigma_m[i_pix, :, :] = self.sigma_m[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                y[i_pix, :, :] = self.y[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                omega[i_pix, :, :] = self.omega[idx[i_pix], :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            sigma_a = (
                sigma_a.transpose((2, 0, 1)).reshape((self.L, N_pix)).T
            )  # .reshape((N_pix, self.L))
            sigma_m = (
                sigma_m.transpose((2, 0, 1)).reshape((self.L, N_pix)).T
            )  # .reshape((N_pix, self.L))
            y = (
                y.transpose((2, 0, 1)).reshape((self.L, N_pix)).T
            )  # .reshape((N_pix, self.L))
            log_y = np.log(y)
            omega = (
                omega.transpose((2, 0, 1)).reshape((self.L, N_pix)).T
            )  # .reshape((N_pix, self.L))            omega

        # * -----
        nll_utils["censored_mask"] = (y <= omega) * 1
        nll_utils["sigma_a"] = sigma_a * 1
        nll_utils["sigma_m"] = sigma_m * 1

        sigma_m2 = sigma_m**2
        log_combination = (
            np.log(sigma_a) - forward_map_evals["log_f_Theta"] - sigma_m2 / 2
        )
        exp_m2_log_combin = np.exp(-2 * log_combination)
        exp_sigma_m_squared = np.exp(sigma_m2)
        # exp_sigma_m_squared_div2_m1 = np.exp(sigma_m2 / 2) - 1  # (N, L)

        # * computation of bias and variances
        nll_utils["m_a"] = np.zeros_like(y)
        nll_utils["s_a"] = sigma_a * np.sqrt(
            (exp_sigma_m_squared - 1)
            * np.exp(2 * (forward_map_evals["log_f_Theta"] - np.log(sigma_a)))
            + 1
        )
        nll_utils["s_a2"] = nll_utils["s_a"] ** 2

        nll_utils["m_m"] = -0.5 * (sigma_m2 + np.log(1 + 1 / exp_m2_log_combin))

        nll_utils["s_m2"] = -2 * nll_utils["m_m"]
        nll_utils["s_m"] = np.sqrt(nll_utils["s_m2"])

        assert (
            nll_utils["s_m"].min() > 0
        ), f"{nll_utils['s_m'].min()}, {(sigma_m2 - 2 * nll_utils['m_m']).min()}"

        if compute_derivatives:
            nll_utils["grad_m_a"] = np.zeros((N_pix, self.D, self.L))

            if compute_derivatives_2nd_order:
                nll_utils["hess_diag_m_a"] = np.zeros((N_pix, self.D, self.L))

            nll_utils["grad_s_a2"] = (
                2
                * (forward_map_evals["f_Theta"] * (exp_sigma_m_squared - 1))[:, None, :]
                * forward_map_evals["grad_f_Theta"]
            )  # (N, D, L)

            if compute_derivatives_2nd_order:
                nll_utils["hess_diag_s_a2"] = (
                    2
                    * (exp_sigma_m_squared - 1)[:, None, :]
                    * (
                        forward_map_evals["f_Theta"][:, None, :]
                        * forward_map_evals["hess_diag_f_Theta"]
                        + forward_map_evals["grad_f_Theta"] ** 2
                    )
                )  # (N, D, L)

            nll_utils["grad_s_a"] = (
                1 / (2 * nll_utils["s_a"])[:, None, :]
            ) * nll_utils["grad_s_a2"]

            if compute_derivatives_2nd_order:
                nll_utils["hess_diag_s_a"] = (1 / (2 * nll_utils["s_a2"]))[
                    :, None, :
                ] * (
                    nll_utils["hess_diag_s_a2"] * nll_utils["s_a"][:, None, :]
                    - nll_utils["grad_s_a2"] * nll_utils["grad_s_a"]
                )  # (N, D, L)

            nll_utils["grad_m_m"] = (
                1 / (1 + exp_m2_log_combin) / forward_map_evals["f_Theta"]
            )[:, None, :] * forward_map_evals[
                "grad_f_Theta"
            ]  # (N, D, L)

            if compute_derivatives_2nd_order:
                nll_utils["hess_diag_m_m"] = (
                    1 / (forward_map_evals["f_Theta"] * (1 + exp_m2_log_combin)) ** 2
                )[:, None, :] * (
                    forward_map_evals["hess_diag_f_Theta"]
                    * (forward_map_evals["f_Theta"] * (1 + exp_m2_log_combin))[
                        :, None, :
                    ]
                    - forward_map_evals["grad_f_Theta"] ** 2
                    * (1 + 3 * exp_m2_log_combin)[:, None, :]
                )  # (N, D, L)

            nll_utils["grad_s_m2"] = -2 * nll_utils["grad_m_m"]  # (N, D, L)

            if compute_derivatives_2nd_order:
                nll_utils["hess_diag_s_m2"] = (
                    -2 * nll_utils["hess_diag_m_m"]
                )  # (N, D, L)

            nll_utils["grad_s_m"] = (1 / (2 * nll_utils["s_m"]))[
                :, None, :
            ] * nll_utils[
                "grad_s_m2"
            ]  # (N, D, L)

            if compute_derivatives_2nd_order:
                nll_utils["hess_diag_s_m"] = (1 / (2 * nll_utils["s_m2"]))[
                    :, None, :
                ] * (
                    nll_utils["hess_diag_s_m2"] * nll_utils["s_m"][:, None, :]
                    - nll_utils["grad_s_m"] * nll_utils["grad_s_m2"]
                )  # (N, D, L)

            # assert np.sum(np.isnan(forward_map_evals["f_Theta"])) == 0.0
            # assert (
            #     forward_map_evals["f_Theta"].min() > 0.0
            # ), f"{forward_map_evals['f_Theta'].min()}, {forward_map_evals['log_f_Theta'].min()}"
            # assert np.sum(np.isnan(forward_map_evals["grad_f_Theta"])) == 0.0
            # assert np.sum(np.isnan(forward_map_evals["hess_diag_f_Theta"])) == 0.0

            # assert np.sum(np.isnan(forward_map_evals["log_f_Theta"])) == 0.0
            # assert np.sum(np.isnan(forward_map_evals["grad_log_f_Theta"])) == 0.0
            # assert np.sum(np.isnan(forward_map_evals["hess_diag_log_f_Theta"])) == 0.0

            # assert np.sum(np.isnan(m_a)) == 0.0
            # assert np.sum(np.isnan(grad_m_a)) == 0.0
            # assert np.sum(np.isnan(hess_diag_m_a)) == 0.0
            # assert m_a.shape == (self.N, self.L)
            # assert grad_m_a.shape == (self.N, self.D, self.L)
            # assert hess_diag_m_a.shape == (self.N, self.D, self.L)

            # assert np.sum(np.isnan(s_a2)) == 0.0
            # assert np.all(s_a2 > 0)
            # assert np.sum(np.isnan(grad_s_a2)) == 0.0
            # assert np.sum(np.isnan(hess_diag_s_a2)) == 0.0
            # assert s_a2.shape == (self.N, self.L)
            # assert grad_s_a2.shape == (self.N, self.D, self.L)
            # assert hess_diag_s_a2.shape == (self.N, self.D, self.L)

            # assert np.sum(np.isnan(s_a)) == 0.0
            # assert np.all(s_a > 0)
            # assert np.sum(np.isnan(grad_s_a)) == 0.0
            # assert np.sum(np.isnan(hess_diag_s_a)) == 0.0
            # assert s_a.shape == (self.N, self.L)
            # assert grad_s_a.shape == (self.N, self.D, self.L)
            # assert hess_diag_s_a.shape == (self.N, self.D, self.L)

            # assert np.sum(np.isnan(m_m)) == 0.0
            # assert np.sum(np.isnan(grad_m_m)) == 0.0
            # assert np.sum(np.isnan(hess_diag_m_m)) == 0.0
            # assert m_m.shape == (self.N, self.L)
            # assert grad_m_m.shape == (self.N, self.D, self.L)
            # assert hess_diag_m_m.shape == (self.N, self.D, self.L)

            # assert np.sum(np.isnan(s_m2)) == 0.0
            # assert np.sum(np.isnan(grad_s_m2)) == 0.0
            # assert np.sum(np.isnan(hess_diag_s_m2)) == 0.0
            # assert s_m2.shape == (self.N, self.L)
            # assert grad_s_m2.shape == (self.N, self.D, self.L)
            # assert hess_diag_s_m2.shape == (self.N, self.D, self.L)

            # assert np.sum(np.isnan(s_m)) == 0.0
            # assert np.sum(np.isnan(grad_s_m)) == 0.0
            # assert np.sum(np.isnan(hess_diag_s_m)) == 0.0
            # assert s_m.shape == (self.N, self.L)
            # assert grad_s_m.shape == (self.N, self.D, self.L)
            # assert hess_diag_s_m.shape == (self.N, self.D, self.L)

        # * mixture weight
        nll_utils["lambda_"] = self.model_mixing_param(forward_map_evals, idx)

        if compute_derivatives:
            nll_utils["grad_lambda_"] = self.grad_model_mixing_param(forward_map_evals)
            if compute_derivatives_2nd_order:
                nll_utils["hess_diag_lambda_"] = self.hess_diag_model_mixing_param(
                    forward_map_evals
                )

        nll_utils["nll_au"] = self.neglog_pdf_au(forward_map_evals, nll_utils, y)
        nll_utils["nll_ac"] = self.neglog_pdf_ac(forward_map_evals, nll_utils, y)
        nll_utils["nll_mu"] = self.neglog_pdf_mu(forward_map_evals, nll_utils, log_y)
        nll_utils["nll_mc"] = self.neglog_pdf_mc(
            forward_map_evals, nll_utils, np.log(omega)
        )

        if compute_derivatives:
            nll_utils["grad_nll_au"] = self.gradient_neglog_pdf_au(
                forward_map_evals, nll_utils
            )
            nll_utils["grad_nll_mu"] = self.gradient_neglog_pdf_mu(
                forward_map_evals, nll_utils
            )
            nll_utils["grad_nll_ac"] = self.gradient_neglog_pdf_ac(
                forward_map_evals, nll_utils
            )
            nll_utils["grad_nll_mc"] = self.gradient_neglog_pdf_mc(
                forward_map_evals, nll_utils
            )

            if compute_derivatives_2nd_order:
                nll_utils["hess_diag_ac"] = self.hess_diag_neglog_pdf_ac(
                    forward_map_evals, nll_utils
                )
                nll_utils["hess_diag_au"] = self.hess_diag_neglog_pdf_au(
                    forward_map_evals, nll_utils
                )
                nll_utils["hess_diag_mc"] = self.hess_diag_neglog_pdf_mc(
                    forward_map_evals, nll_utils
                )
                nll_utils["hess_diag_mu"] = self.hess_diag_neglog_pdf_mu(
                    forward_map_evals, nll_utils
                )

        return nll_utils
