"""Implementation of an approximation of the likelihood function of a mixture of Gaussian and multiplicative noises with censorship with a lower limit."""

import warnings
from typing import List, Optional, Union

import numba
import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from beetroots.modelling.likelihoods import utils
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood

SQRT_2_PI = np.sqrt(2 * np.pi)


@numba.njit()
def _peak_and_derivatives(u, c0, c1, c2, c3, c4, B, d, omega_chirp, k, phi):
    """Peak model and its first/second derivatives with respect to u."""
    eps = 1e-12
    u_safe = np.clip(u, eps, 1.0 - eps)

    # Polynomial envelope p(u) = c0*u^4 + ... + c4
    p = ((((c0 * u_safe + c1) * u_safe + c2) * u_safe + c3) * u_safe) + c4
    p_1 = (((4.0 * c0 * u_safe + 3.0 * c1) * u_safe + 2.0 * c2) * u_safe) + c3
    p_2 = ((12.0 * c0 * u_safe + 6.0 * c1) * u_safe) + 2.0 * c2

    base = np.exp(p)
    base_1 = base * p_1
    base_2 = base * (p_2 + p_1**2)

    # Damped chirped oscillation
    phase = omega_chirp * u_safe + k * (u_safe**2) + phi
    phase_1 = omega_chirp + 2.0 * k * u_safe
    phase_2 = 2.0 * k

    amp = B * np.exp(-d * u_safe)
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)

    osc = amp * sin_phase
    osc_1 = amp * (-d * sin_phase + cos_phase * phase_1)
    osc_2 = amp * (
        d**2 * sin_phase
        - 2.0 * d * cos_phase * phase_1
        - sin_phase * (phase_1**2)
        + cos_phase * phase_2
    )

    s = base + osc
    s_1 = base_1 + osc_1
    s_2 = base_2 + osc_2

    # Window to enforce null peak at the transition boundaries
    w = u_safe * (1.0 - u_safe)
    w_1 = 1.0 - 2.0 * u_safe
    w_2 = -2.0

    peak = w * s
    peak_1 = w_1 * s + w * s_1
    peak_2 = w_2 * s + 2.0 * w_1 * s_1 + w * s_2

    return peak, peak_1, peak_2


@numba.njit()
def log_Z(u, c0, c1, c2, c3, c4, B, d, omega_chirp, k, phi):
    """
    Numba-compiled evaluation of -log(Z) for the exp-poly chirped-window model.
    """
    peak, _, _ = _peak_and_derivatives(u, c0, c1, c2, c3, c4, B, d, omega_chirp, k, phi)
    z = np.maximum(1.0 - peak, 1e-12)
    return -np.log(z)


@numba.njit()
def grad_log_Z(u, c0, c1, c2, c3, c4, B, d, omega_chirp, k, phi):
    """
    Computes d/du (-log(Z(u))) for the exp-poly chirped-window model.
    """
    peak, peak_1, _ = _peak_and_derivatives(
        u, c0, c1, c2, c3, c4, B, d, omega_chirp, k, phi
    )
    z = np.maximum(1.0 - peak, 1e-12)
    return peak_1 / z


@numba.njit()
def njit_hess_diag_log_Z_core(u, c0, c1, c2, c3, c4, B, d, omega_chirp, k, phi):
    """
    Computes d²/du² (-log(Z(u))) for the exp-poly chirped-window model.
    """
    peak, peak_1, peak_2 = _peak_and_derivatives(
        u, c0, c1, c2, c3, c4, B, d, omega_chirp, k, phi
    )
    z = np.maximum(1.0 - peak, 1e-12)
    return (peak_2 / z) + (peak_1 / z) ** 2


@numba.njit()
def cost_au(
    y: np.ndarray, f_Var: np.ndarray, m_a: np.ndarray, s_a: np.ndarray
) -> np.ndarray:
    nll_au = 0.5 * ((y - f_Var - m_a) / s_a) ** 2 + np.log(s_a * SQRT_2_PI)
    return nll_au


@numba.njit()
def cost_mu(
    log_y: np.ndarray, log_f_Var: np.ndarray, m_m: np.ndarray, s_m: np.ndarray
) -> np.ndarray:
    nll_mu = (
        0.5 * ((log_y - log_f_Var - m_m) / s_m) ** 2 + np.log(s_m * SQRT_2_PI) + log_y
    )
    # nll_mu = np.where(np.isnan(log_y) | np.isinf(log_y), 10.0 ** 8, nll_mu)

    # shape = nll_mu.shape
    # nll_mu = nll_mu.ravel()
    # nll_mu[np.isnan(nll_mu)] = 0
    # nll_mu = nll_mu.reshape(shape)
    return nll_mu


@numba.njit()
def gradient_cost_au(
    y: np.ndarray,
    grad_f_Var: np.ndarray,
    f_Var: np.ndarray,
    grad_m_a: np.ndarray,
    m_a: np.ndarray,
    grad_s_a2: np.ndarray,
    s_a2: np.ndarray,
) -> np.ndarray:
    u_1 = (f_Var - m_a - y) / s_a2**2
    assert u_1.shape == f_Var.shape  # (N, L)

    u_2 = (
        2 * np.expand_dims(s_a2, axis=1) * (grad_f_Var + grad_m_a)
        - np.expand_dims(f_Var - m_a - y, axis=1) * grad_s_a2
    )
    assert u_2.shape == grad_f_Var.shape  # (N, D, L)

    g_au = 0.5 * grad_s_a2 / np.expand_dims(s_a2, axis=1)
    g_au += 0.5 * np.expand_dims(u_1, axis=1) * u_2

    assert g_au.shape == grad_f_Var.shape
    return g_au


@numba.njit()
def gradient_cost_mu(
    log_y: np.ndarray,
    grad_log_f_Var: np.ndarray,
    log_f_Var: np.ndarray,
    grad_m_m: np.ndarray,
    m_m: np.ndarray,
    grad_s_m2: np.ndarray,
    s_m2: np.ndarray,
) -> np.ndarray:
    u_1 = (log_f_Var - m_m - log_y) / s_m2**2
    assert u_1.shape == log_f_Var.shape  # (N, L)

    u_2 = (
        2 * np.expand_dims(s_m2, axis=1) * (grad_log_f_Var + grad_m_m)
        - np.expand_dims(log_f_Var - m_m - log_y, axis=1) * grad_s_m2
    )
    assert u_2.shape == grad_log_f_Var.shape  # (N, D, L)

    g_mu = 0.5 * grad_s_m2 / np.expand_dims(s_m2, axis=1)
    g_mu += np.where(
        np.expand_dims(np.isnan(log_y), axis=1) + np.zeros_like(grad_log_f_Var),
        # fake output
        np.zeros_like(grad_log_f_Var),
        # true grad
        0.5 * np.expand_dims(u_1, axis=1) * u_2,
    )

    assert g_mu.shape == grad_log_f_Var.shape
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
        var_name: str,
        **kwargs,
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
        super().__init__(forward_map, y, D, L, N, var_name, **kwargs)

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

        df_transition = pd.read_csv(
            path_transition_params
        )  # e.g. best_params.csv (see list_mixing_model_params in run_mcmc.py and simulation files)
        num_distinct_n = np.unique(df_transition["n"]).size

        cols_to_check = [
            "fit_Z_poly_c0",
            "fit_Z_poly_c1",
            "fit_Z_poly_c2",
            "fit_Z_poly_c3",
            "fit_Z_poly_c4",
            "fit_Z_B",
            "fit_Z_d",
            "fit_Z_omega",
            "fit_Z_k",
            "fit_Z_phi",
        ]
        self.apply_normalization = all(
            col in df_transition.columns for col in cols_to_check
        )
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

            if self.apply_normalization:
                self.Z_poly_c0 = df_transition.loc[
                    list_lines_fit, "fit_Z_poly_c0"
                ].values[None, :] * np.ones((self.N, self.L))
                self.Z_poly_c1 = df_transition.loc[
                    list_lines_fit, "fit_Z_poly_c1"
                ].values[None, :] * np.ones((self.N, self.L))
                self.Z_poly_c2 = df_transition.loc[
                    list_lines_fit, "fit_Z_poly_c2"
                ].values[None, :] * np.ones((self.N, self.L))
                self.Z_poly_c3 = df_transition.loc[
                    list_lines_fit, "fit_Z_poly_c3"
                ].values[None, :] * np.ones((self.N, self.L))
                self.Z_poly_c4 = df_transition.loc[
                    list_lines_fit, "fit_Z_poly_c4"
                ].values[None, :] * np.ones((self.N, self.L))
                self.Z_B = df_transition.loc[list_lines_fit, "fit_Z_B"].values[
                    None, :
                ] * np.ones((self.N, self.L))
                self.Z_d = df_transition.loc[list_lines_fit, "fit_Z_d"].values[
                    None, :
                ] * np.ones((self.N, self.L))
                self.Z_omega = df_transition.loc[list_lines_fit, "fit_Z_omega"].values[
                    None, :
                ] * np.ones((self.N, self.L))
                self.Z_k = df_transition.loc[list_lines_fit, "fit_Z_k"].values[
                    None, :
                ] * np.ones((self.N, self.L))
                self.Z_phi = df_transition.loc[list_lines_fit, "fit_Z_phi"].values[
                    None, :
                ] * np.ones((self.N, self.L))
            else:
                warnings.warn(
                    "The normalization constant of the likelihood is not taken into account. To take it into account, the transition parameters file should contain the columns fit_Z_poly_c0..fit_Z_poly_c4, fit_Z_B, fit_Z_d, fit_Z_omega, fit_Z_k and fit_Z_phi."
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
            if self.apply_normalization:
                self.Z_poly_c0 = df_transition.loc[
                    index, "fit_Z_poly_c0"
                ].values.reshape((self.N, self.L))
                self.Z_poly_c1 = df_transition.loc[
                    index, "fit_Z_poly_c1"
                ].values.reshape((self.N, self.L))
                self.Z_poly_c2 = df_transition.loc[
                    index, "fit_Z_poly_c2"
                ].values.reshape((self.N, self.L))
                self.Z_poly_c3 = df_transition.loc[
                    index, "fit_Z_poly_c3"
                ].values.reshape((self.N, self.L))
                self.Z_poly_c4 = df_transition.loc[
                    index, "fit_Z_poly_c4"
                ].values.reshape((self.N, self.L))
                self.Z_B = df_transition.loc[index, "fit_Z_B"].values.reshape(
                    (self.N, self.L)
                )
                self.Z_d = df_transition.loc[index, "fit_Z_d"].values.reshape(
                    (self.N, self.L)
                )
                self.Z_omega = df_transition.loc[index, "fit_Z_omega"].values.reshape(
                    (self.N, self.L)
                )
                self.Z_k = df_transition.loc[index, "fit_Z_k"].values.reshape(
                    (self.N, self.L)
                )
                self.Z_phi = df_transition.loc[index, "fit_Z_phi"].values.reshape(
                    (self.N, self.L)
                )
            else:
                warnings.warn(
                    "The normalization constant of the likelihood is not taken into account. To take it into account, the transition parameters file should contain the columns fit_Z_poly_c0..fit_Z_poly_c4, fit_Z_B, fit_Z_d, fit_Z_omega, fit_Z_k and fit_Z_phi."
                )

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
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        eps_a = rng.normal(loc=0.0, scale=self.sigma_a)
        eps_m = rng.lognormal(
            mean=-(self.sigma_m**2) / 2,
            sigma=self.sigma_m,
        )
        f_Var = self.forward_map_evals["f_Var"] * 1
        y_rep = np.maximum(self.omega, eps_m * f_Var + eps_a)
        return y_rep

    def model_mixing_param(
        self,
        idx_pix: Optional[np.ndarray] = None,
        mtm: bool = False,
    ) -> np.ndarray:
        r"""computes the weight of the additive model :math:`\lambda_{n, \ell}` (line-wise and pixel-wise). In this model, :math:`\lambda_{n, \ell}` is a function of the observation :math:`y_{n, \ell}`, and therefore constant during the sampling

        .. math::
            \lambda_{n, \ell} = \frac{\sigma_a^{2b}}{\sigma_a^{2b} + (a \sigma_m y_{n, \ell})^{2b}}

        with :math:`a` a transition location parameter and :math:`b` a transition speed parameter
        """
        if mtm is False:
            N_pix = idx_pix.size
            log_fm1 = self.log_fm1[idx_pix] * 1
            log_fp1 = self.log_fp1[idx_pix] * 1
            # sigma_a = self.sigma_a * 1
            # sigma_m = self.sigma_m * 1
        else:
            n_pix = idx_pix.size
            k_mtm = self.forward_map_evals["f_Var"].shape[0] // n_pix
            N_pix = self.forward_map_evals["f_Var"].shape[0]

            log_fm1 = np.zeros((n_pix, k_mtm, self.L))
            log_fp1 = np.zeros((n_pix, k_mtm, self.L))

            for i, i_pix in enumerate(idx_pix):
                log_fm1[i, :, :] = self.log_fm1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                log_fp1[i, :, :] = self.log_fp1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            log_fm1 = log_fm1.reshape((N_pix, self.L))
            log_fp1 = log_fp1.reshape((N_pix, self.L))

            # N_pix = self.forward_map_evals["f_Var"].shape[0]

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
        #             (self.transition_loc * sigma_m * self.forward_map_evals["f_Var"]) / sigma_a
        #         )
        #     )
        # )
        lambda_ = np.where(
            self.forward_map_evals["log_f_Var"] <= log_fm1,
            1,
            np.where(
                self.forward_map_evals["log_f_Var"] >= log_fp1,
                0,
                self.P_lambda(
                    (self.forward_map_evals["log_f_Var"] - log_fm1)
                    / (log_fp1 - log_fm1)
                ),
            ),
        )

        return lambda_  # (N, L)

    def grad_model_mixing_param(self, idx_pix, mtm) -> np.ndarray:
        r"""[summary]

        Parameters
        ----------

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
        #             forward_map_evals["f_Var"] ** (2 * self.transition_speed - 1)
        #             * lambda_ ** 2
        #         )
        #     )[:, None, :]
        #     * forward_map_evals["
        # _f_Var"]
        # )
        if mtm is False:
            N_pix = idx_pix.size
            log_fm1 = self.log_fm1[idx_pix] * 1
            log_fp1 = self.log_fp1[idx_pix] * 1
            # sigma_a = self.sigma_a * 1
            # sigma_m = self.sigma_m * 1
        else:
            n_pix = idx_pix.size
            k_mtm = self.forward_map_evals["f_Var"].shape[0] // n_pix
            N_pix = self.forward_map_evals["f_Var"].shape[0]

            log_fm1 = np.zeros((n_pix, k_mtm, self.L))
            log_fp1 = np.zeros((n_pix, k_mtm, self.L))

            for i, i_pix in enumerate(idx_pix):
                log_fm1[i, :, :] = self.log_fm1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                log_fp1[i, :, :] = self.log_fp1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            log_fm1 = log_fm1.reshape((N_pix, self.L))
            log_fp1 = log_fp1.reshape((N_pix, self.L))

        u = (self.forward_map_evals["log_f_Var"] - log_fm1) / (
            log_fp1 - log_fm1
        )  # (N, L)
        grad_u = (
            self.forward_map_evals["grad_log_f_Var"] / (log_fp1 - log_fm1)[:, None, :]
        )  # (N, D, L)
        # assert u.shape == (self.N, self.L)
        # assert grad_u.shape == (self.N, self.D, self.L), grad_u.shape

        grad_ = np.where(
            (
                (self.forward_map_evals["log_f_Var"] <= log_fm1)
                | (self.forward_map_evals["log_f_Var"] >= log_fp1)
            )[:, None, :],
            np.zeros((N_pix, self.D, self.L)),
            grad_u * self.grad_P_lambda(u)[:, None, :],
        )
        assert grad_.shape == (N_pix, self.D, self.L), grad_.shape

        return grad_  # (N, D, L)

    def hess_diag_model_mixing_param(self, idx_pix, mtm) -> np.ndarray:
        r"""[summary]

        Parameters
        ----------

        Returns
        -------
        np.ndarray of shape (N, D, L)
            [description]
        """
        n_pix = self.nlpdf_utils["n_pix"]

        u = (self.forward_map_evals["log_f_Var"] - self.log_fm1[idx_pix]) / (
            self.log_fp1[idx_pix] - self.log_fm1[idx_pix]
        )  # (n_pix, L)
        grad_u = (
            self.forward_map_evals["grad_log_f_Var"]
            / (self.log_fp1[idx_pix] - self.log_fm1[idx_pix])[:, None, :]
        )  # (n_pix, D, L)
        hess_diag_u = (
            self.forward_map_evals["hess_diag_log_f_Var"]
            / (self.log_fp1[idx_pix] - self.log_fm1[idx_pix])[:, None, :]
        )  # (n_pix, D, L)

        hess_diag = np.where(
            (
                (self.forward_map_evals["log_f_Var"] <= self.log_fm1[idx_pix])
                | (self.forward_map_evals["log_f_Var"] >= self.log_fp1[idx_pix])
            )[:, None, :],
            np.zeros((n_pix, self.D, self.L)),
            (
                grad_u**2 * self.hess_diag_P_lambda(u)[:, None, :]
                + hess_diag_u * self.grad_P_lambda(u)[:, None, :]
            ),
        )
        assert hess_diag.shape == (n_pix, self.D, self.L), hess_diag.shape
        return hess_diag  # (n_pix, D, L)

    def normalization(
        self,
        idx_pix: Optional[np.ndarray] = None,
        mtm: bool = False,
    ) -> np.ndarray:
        r"""
        Computes the negative log normalization constant -log(Z).
        The contribution is set to 0 outside the transition regime.
        """
        if mtm is False:
            N_pix = idx_pix.size
            log_fm1 = self.log_fm1[idx_pix] * 1
            log_fp1 = self.log_fp1[idx_pix] * 1
            Z_poly_c0 = self.Z_poly_c0[idx_pix] * 1
            Z_poly_c1 = self.Z_poly_c1[idx_pix] * 1
            Z_poly_c2 = self.Z_poly_c2[idx_pix] * 1
            Z_poly_c3 = self.Z_poly_c3[idx_pix] * 1
            Z_poly_c4 = self.Z_poly_c4[idx_pix] * 1
            Z_B = self.Z_B[idx_pix] * 1
            Z_d = self.Z_d[idx_pix] * 1
            Z_omega = self.Z_omega[idx_pix] * 1
            Z_k = self.Z_k[idx_pix] * 1
            Z_phi = self.Z_phi[idx_pix] * 1
        else:
            n_pix = idx_pix.size
            k_mtm = self.forward_map_evals["f_Var"].shape[0] // n_pix
            N_pix = self.forward_map_evals["f_Var"].shape[0]

            log_fm1 = np.zeros((n_pix, k_mtm, self.L))
            log_fp1 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c0 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c1 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c2 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c3 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c4 = np.zeros((n_pix, k_mtm, self.L))
            Z_B = np.zeros((n_pix, k_mtm, self.L))
            Z_d = np.zeros((n_pix, k_mtm, self.L))
            Z_omega = np.zeros((n_pix, k_mtm, self.L))
            Z_k = np.zeros((n_pix, k_mtm, self.L))
            Z_phi = np.zeros((n_pix, k_mtm, self.L))

            for i, i_pix in enumerate(idx_pix):
                log_fm1[i, :, :] = self.log_fm1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                log_fp1[i, :, :] = self.log_fp1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c0[i, :, :] = self.Z_poly_c0[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c1[i, :, :] = self.Z_poly_c1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c2[i, :, :] = self.Z_poly_c2[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c3[i, :, :] = self.Z_poly_c3[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c4[i, :, :] = self.Z_poly_c4[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_B[i, :, :] = self.Z_B[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                Z_d[i, :, :] = self.Z_d[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                Z_omega[i, :, :] = self.Z_omega[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_k[i, :, :] = self.Z_k[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                Z_phi[i, :, :] = self.Z_phi[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            log_fm1 = log_fm1.reshape((N_pix, self.L))
            log_fp1 = log_fp1.reshape((N_pix, self.L))
            Z_poly_c0 = Z_poly_c0.reshape((N_pix, self.L))
            Z_poly_c1 = Z_poly_c1.reshape((N_pix, self.L))
            Z_poly_c2 = Z_poly_c2.reshape((N_pix, self.L))
            Z_poly_c3 = Z_poly_c3.reshape((N_pix, self.L))
            Z_poly_c4 = Z_poly_c4.reshape((N_pix, self.L))
            Z_B = Z_B.reshape((N_pix, self.L))
            Z_d = Z_d.reshape((N_pix, self.L))
            Z_omega = Z_omega.reshape((N_pix, self.L))
            Z_k = Z_k.reshape((N_pix, self.L))
            Z_phi = Z_phi.reshape((N_pix, self.L))

        # Map to standard u domain
        u = (self.forward_map_evals["log_f_Var"] - log_fm1) / (log_fp1 - log_fm1)

        # Evaluate using the fast compiled Numba core
        neglog_normalization_u = log_Z(
            u,
            Z_poly_c0,
            Z_poly_c1,
            Z_poly_c2,
            Z_poly_c3,
            Z_poly_c4,
            Z_B,
            Z_d,
            Z_omega,
            Z_k,
            Z_phi,
        )

        # -log(Z) = 0 outside the transition
        neglog_normalization = np.where(
            (self.forward_map_evals["log_f_Var"] <= log_fm1)
            | (self.forward_map_evals["log_f_Var"] >= log_fp1),
            0.0,
            neglog_normalization_u,
        )

        return neglog_normalization  # (N, L)

    def grad_normalization(self, idx_pix, mtm) -> np.ndarray:
        r"""
        Computes the first derivative of -log(Z) with respect to the parameters.
        """
        if mtm is False:
            N_pix = idx_pix.size
            log_fm1 = self.log_fm1[idx_pix] * 1
            log_fp1 = self.log_fp1[idx_pix] * 1
            Z_poly_c0 = self.Z_poly_c0[idx_pix] * 1
            Z_poly_c1 = self.Z_poly_c1[idx_pix] * 1
            Z_poly_c2 = self.Z_poly_c2[idx_pix] * 1
            Z_poly_c3 = self.Z_poly_c3[idx_pix] * 1
            Z_poly_c4 = self.Z_poly_c4[idx_pix] * 1
            Z_B = self.Z_B[idx_pix] * 1
            Z_d = self.Z_d[idx_pix] * 1
            Z_omega = self.Z_omega[idx_pix] * 1
            Z_k = self.Z_k[idx_pix] * 1
            Z_phi = self.Z_phi[idx_pix] * 1
        else:
            n_pix = idx_pix.size
            k_mtm = self.forward_map_evals["f_Var"].shape[0] // n_pix
            N_pix = self.forward_map_evals["f_Var"].shape[0]

            log_fm1 = np.zeros((n_pix, k_mtm, self.L))
            log_fp1 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c0 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c1 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c2 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c3 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c4 = np.zeros((n_pix, k_mtm, self.L))
            Z_B = np.zeros((n_pix, k_mtm, self.L))
            Z_d = np.zeros((n_pix, k_mtm, self.L))
            Z_omega = np.zeros((n_pix, k_mtm, self.L))
            Z_k = np.zeros((n_pix, k_mtm, self.L))
            Z_phi = np.zeros((n_pix, k_mtm, self.L))

            for i, i_pix in enumerate(idx_pix):
                log_fm1[i, :, :] = self.log_fm1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                log_fp1[i, :, :] = self.log_fp1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c0[i, :, :] = self.Z_poly_c0[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c1[i, :, :] = self.Z_poly_c1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c2[i, :, :] = self.Z_poly_c2[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c3[i, :, :] = self.Z_poly_c3[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c4[i, :, :] = self.Z_poly_c4[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_B[i, :, :] = self.Z_B[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                Z_d[i, :, :] = self.Z_d[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                Z_omega[i, :, :] = self.Z_omega[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_k[i, :, :] = self.Z_k[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                Z_phi[i, :, :] = self.Z_phi[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            log_fm1 = log_fm1.reshape((N_pix, self.L))
            log_fp1 = log_fp1.reshape((N_pix, self.L))
            Z_poly_c0 = Z_poly_c0.reshape((N_pix, self.L))
            Z_poly_c1 = Z_poly_c1.reshape((N_pix, self.L))
            Z_poly_c2 = Z_poly_c2.reshape((N_pix, self.L))
            Z_poly_c3 = Z_poly_c3.reshape((N_pix, self.L))
            Z_poly_c4 = Z_poly_c4.reshape((N_pix, self.L))
            Z_B = Z_B.reshape((N_pix, self.L))
            Z_d = Z_d.reshape((N_pix, self.L))
            Z_omega = Z_omega.reshape((N_pix, self.L))
            Z_k = Z_k.reshape((N_pix, self.L))
            Z_phi = Z_phi.reshape((N_pix, self.L))

        u = (self.forward_map_evals["log_f_Var"] - log_fm1) / (
            log_fp1 - log_fm1
        )  # (N, L)
        grad_u = (
            self.forward_map_evals["grad_log_f_Var"] / (log_fp1 - log_fm1)[:, None, :]
        )  # (N, D, L)

        # 1. Evaluate the derivative using the fast Numba core
        grad_neglog_normalization_u = grad_log_Z(
            u,
            Z_poly_c0,
            Z_poly_c1,
            Z_poly_c2,
            Z_poly_c3,
            Z_poly_c4,
            Z_B,
            Z_d,
            Z_omega,
            Z_k,
            Z_phi,
        )

        # 2. Apply chain rule and boundaries
        grad_ = np.where(
            (
                (self.forward_map_evals["log_f_Var"] <= log_fm1)
                | (self.forward_map_evals["log_f_Var"] >= log_fp1)
            )[:, None, :],
            np.zeros((N_pix, self.D, self.L)),
            grad_u * grad_neglog_normalization_u[:, None, :],
        )
        assert grad_.shape == (N_pix, self.D, self.L), grad_.shape

        return grad_  # (N, D, L)

    def hess_diag_normalization(self, idx_pix, mtm) -> np.ndarray:
        r"""
        Computes the diagonal Hessian of -log(Z) with respect to the parameters.
        """
        if mtm is False:
            n_pix = self.nlpdf_utils["n_pix"]
            N_pix = n_pix
            log_fm1 = self.log_fm1[idx_pix] * 1
            log_fp1 = self.log_fp1[idx_pix] * 1
            Z_poly_c0 = self.Z_poly_c0[idx_pix] * 1
            Z_poly_c1 = self.Z_poly_c1[idx_pix] * 1
            Z_poly_c2 = self.Z_poly_c2[idx_pix] * 1
            Z_poly_c3 = self.Z_poly_c3[idx_pix] * 1
            Z_poly_c4 = self.Z_poly_c4[idx_pix] * 1
            Z_B = self.Z_B[idx_pix] * 1
            Z_d = self.Z_d[idx_pix] * 1
            Z_omega = self.Z_omega[idx_pix] * 1
            Z_k = self.Z_k[idx_pix] * 1
            Z_phi = self.Z_phi[idx_pix] * 1
        else:
            n_pix = idx_pix.size
            k_mtm = self.forward_map_evals["f_Var"].shape[0] // n_pix
            N_pix = self.forward_map_evals["f_Var"].shape[0]

            log_fm1 = np.zeros((n_pix, k_mtm, self.L))
            log_fp1 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c0 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c1 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c2 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c3 = np.zeros((n_pix, k_mtm, self.L))
            Z_poly_c4 = np.zeros((n_pix, k_mtm, self.L))
            Z_B = np.zeros((n_pix, k_mtm, self.L))
            Z_d = np.zeros((n_pix, k_mtm, self.L))
            Z_omega = np.zeros((n_pix, k_mtm, self.L))
            Z_k = np.zeros((n_pix, k_mtm, self.L))
            Z_phi = np.zeros((n_pix, k_mtm, self.L))

            for i, i_pix in enumerate(idx_pix):
                log_fm1[i, :, :] = self.log_fm1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                log_fp1[i, :, :] = self.log_fp1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c0[i, :, :] = self.Z_poly_c0[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c1[i, :, :] = self.Z_poly_c1[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c2[i, :, :] = self.Z_poly_c2[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c3[i, :, :] = self.Z_poly_c3[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_poly_c4[i, :, :] = self.Z_poly_c4[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_B[i, :, :] = self.Z_B[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                Z_d[i, :, :] = self.Z_d[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                Z_omega[i, :, :] = self.Z_omega[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                Z_k[i, :, :] = self.Z_k[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                Z_phi[i, :, :] = self.Z_phi[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            log_fm1 = log_fm1.reshape((N_pix, self.L))
            log_fp1 = log_fp1.reshape((N_pix, self.L))
            Z_poly_c0 = Z_poly_c0.reshape((N_pix, self.L))
            Z_poly_c1 = Z_poly_c1.reshape((N_pix, self.L))
            Z_poly_c2 = Z_poly_c2.reshape((N_pix, self.L))
            Z_poly_c3 = Z_poly_c3.reshape((N_pix, self.L))
            Z_poly_c4 = Z_poly_c4.reshape((N_pix, self.L))
            Z_B = Z_B.reshape((N_pix, self.L))
            Z_d = Z_d.reshape((N_pix, self.L))
            Z_omega = Z_omega.reshape((N_pix, self.L))
            Z_k = Z_k.reshape((N_pix, self.L))
            Z_phi = Z_phi.reshape((N_pix, self.L))

        u = (self.forward_map_evals["log_f_Var"] - log_fm1) / (
            log_fp1 - log_fm1
        )  # (n_pix, L) or (N_pix, L)
        grad_u = (
            self.forward_map_evals["grad_log_f_Var"] / (log_fp1 - log_fm1)[:, None, :]
        )  # (n_pix, D, L) or (N_pix, D, L)
        hess_diag_u = (
            self.forward_map_evals["hess_diag_log_f_Var"]
            / (log_fp1 - log_fm1)[:, None, :]
        )  # (n_pix, D, L) or (N_pix, D, L)

        # 1. Evaluate first and second derivatives using Numba cores
        grad_neglog_Z_u = grad_log_Z(
            u,
            Z_poly_c0,
            Z_poly_c1,
            Z_poly_c2,
            Z_poly_c3,
            Z_poly_c4,
            Z_B,
            Z_d,
            Z_omega,
            Z_k,
            Z_phi,
        )
        hess_diag_neglog_Z_u = njit_hess_diag_log_Z_core(
            u,
            Z_poly_c0,
            Z_poly_c1,
            Z_poly_c2,
            Z_poly_c3,
            Z_poly_c4,
            Z_B,
            Z_d,
            Z_omega,
            Z_k,
            Z_phi,
        )

        # 2. Apply chain rules and boundaries
        hess_diag = np.where(
            (
                (self.forward_map_evals["log_f_Var"] <= log_fm1)
                | (self.forward_map_evals["log_f_Var"] >= log_fp1)
            )[:, None, :],
            np.zeros((N_pix, self.D, self.L)),
            (
                (grad_u**2) * hess_diag_neglog_Z_u[:, None, :]
                + hess_diag_u * grad_neglog_Z_u[:, None, :]
            ),
        )
        assert hess_diag.shape == (N_pix, self.D, self.L), hess_diag.shape
        return hess_diag  # (N_pix, D, L)

    def neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        nlpdf = self.nlpdf_utils["lambda_"] * np.where(
            self.nlpdf_utils["censored_mask"],
            self.nlpdf_utils["nll_ac"],
            self.nlpdf_utils["nll_au"],
        ) + (1 - self.nlpdf_utils["lambda_"]) * np.where(
            self.nlpdf_utils["censored_mask"],
            self.nlpdf_utils["nll_mc"],
            self.nlpdf_utils["nll_mu"],
        )  # (N, L)
        if self.apply_normalization:
            nlpdf += self.nlpdf_utils["log_Z"]  # (N, L)
        nlpdf = np.nan_to_num(nlpdf)
        nlpdf -= np.log(self.nlpdf_utils["sigma_a"]) + np.log(
            self.nlpdf_utils["sigma_m"]
        )  # NOTE: used to make sure the nlpdf is positive (better for plots)
        # nlpdf /= self.N * self.L
        # self.nlpdf_utils["censored_mask"].size = self.N * self.L if standard eval
        # self.nlpdf_utils["censored_mask"].size = N_candidates * self.L if candidates

        if self.nlpdf_utils["k_mtm"] > 0:
            nlpdf = nlpdf.reshape(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L)
            )

        if full:
            return nlpdf  # (n_pix, L) or (n_pix, k_mtm, L)

        if pixelwise:
            sum_ = np.sum(nlpdf, axis=-1)  # (n_pix,) or (n_pix, k_mtm)
            assert sum_.size == self.forward_map_evals["f_Var"].shape[0]
            return sum_

        return (
            nlpdf.sum() if self.nlpdf_utils["k_mtm"] == 0 else nlpdf.sum(axis=(0, -1))
        )

    def neglog_pdf_ac(
        self,
        omega: np.ndarray,
    ) -> np.ndarray:
        z = (
            omega - self.forward_map_evals["f_Var"] - self.nlpdf_utils["m_a"]
        ) / self.nlpdf_utils["s_a"]
        # z = np.nan_to_num(z)

        nll_ac = -log_ndtr(z)
        nll_ac = np.nan_to_num(nll_ac)

        return nll_ac

    def neglog_pdf_au(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        nll_au = cost_au(
            y=y,
            f_Var=self.forward_map_evals["f_Var"],
            m_a=self.nlpdf_utils["m_a"],
            s_a=self.nlpdf_utils["s_a"],
        )
        # add a constant to `nll_au` to ensure that it is positive
        # nll_au -= np.log(self.nlpdf_utils["sigma_a"])
        # nll_au = np.nan_to_num(nll_au)
        return nll_au

    def neglog_pdf_mc(
        self,
        log_omega: np.ndarray,
    ) -> np.ndarray:
        z = log_omega - self.forward_map_evals["log_f_Var"] - self.nlpdf_utils["m_m"]
        z /= self.nlpdf_utils["s_m"]
        # z = np.nan_to_num(z)

        nll_mc = -log_ndtr(z)
        nll_mc = np.nan_to_num(nll_mc)
        return nll_mc

    def neglog_pdf_mu(
        self,
        log_y: np.ndarray,
    ) -> np.ndarray:
        nll_mu = cost_mu(
            log_y=log_y,
            log_f_Var=self.forward_map_evals["log_f_Var"],
            m_m=self.nlpdf_utils["m_m"],
            s_m=self.nlpdf_utils["s_m"],
        )
        # nll_mu -= np.log(self.nlpdf_utils["sigma_m"])
        return nll_mu

    def gradient_neglog_pdf(
        self,
        **kwargs,
    ) -> np.ndarray:
        """[summary]

        [extended_summary]

        Parameters
        ----------

        Returns
        -------
        np.ndarray of shape (N, D)
            [description]
        """
        grad_ = np.where(
            (self.nlpdf_utils["y"] == self.nlpdf_utils["omega"])[:, None, :],
            # censored
            # grad_nll_ac,
            self.nlpdf_utils["lambda_"][:, None, :] * self.nlpdf_utils["grad_nll_ac"]
            + self.nlpdf_utils["grad_lambda_"] * self.nlpdf_utils["nll_ac"][:, None, :]
            + (1 - self.nlpdf_utils["lambda_"])[:, None, :]
            * self.nlpdf_utils["grad_nll_mc"]
            - self.nlpdf_utils["grad_lambda_"] * self.nlpdf_utils["nll_mc"][:, None, :],
            # uncensored
            self.nlpdf_utils["lambda_"][:, None, :] * self.nlpdf_utils["grad_nll_au"]
            + self.nlpdf_utils["grad_lambda_"] * self.nlpdf_utils["nll_au"][:, None, :]
            + (1 - self.nlpdf_utils["lambda_"])[:, None, :]
            * self.nlpdf_utils["grad_nll_mu"]
            - self.nlpdf_utils["grad_lambda_"] * self.nlpdf_utils["nll_mu"][:, None, :],
        )  # (N, D, L)
        # grad_ = np.where(
        #     np.isfinite(grad_),
        #     grad_,
        #     np.abs(grad_).max(axis=0)[None, :, :],
        # )  # (N, D, L)
        if self.apply_normalization:
            grad_ += self.nlpdf_utils["grad_log_Z"]
        grad_ = np.nan_to_num(grad_)

        if self.nlpdf_utils["k_mtm"] > 0:
            grad_ = grad_.reshape(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.D, self.L)
            )

        return np.sum(grad_, axis=-1)  # / (self.N * self.L)

    def gradient_neglog_pdf_ac(
        self,
    ) -> np.ndarray:
        f_Var_m_a_omega = (
            self.forward_map_evals["f_Var"]
            + self.nlpdf_utils["m_a"]
            - self.nlpdf_utils["omega"]
        )

        u_1 = utils.norm_pdf_cdf_ratio(
            -f_Var_m_a_omega / self.nlpdf_utils["s_a"]
        )  # (N, L)

        u_2 = (1 / self.nlpdf_utils["s_a2"])[:, None, :] * (
            (self.forward_map_evals["grad_f_Var"] + self.nlpdf_utils["grad_m_a"])
            * self.nlpdf_utils["s_a"][:, None, :]
            - f_Var_m_a_omega[:, None, :] * self.nlpdf_utils["grad_s_a"]
        )  # (N, D, L)

        grad_ = u_1[:, None, :] * u_2  # (N, D, L)
        # grad_ = np.nan_to_num(grad_)
        # assert np.sum(np.isnan(grad_)) == 0, f"grad_ac : {np.sum(np.isnan(grad_))}"
        return grad_

    def gradient_neglog_pdf_au(
        self,
    ) -> np.ndarray:
        g_au = gradient_cost_au(
            y=self.nlpdf_utils["y"],
            grad_f_Var=self.forward_map_evals["grad_f_Var"],
            f_Var=self.forward_map_evals["f_Var"],
            grad_m_a=self.nlpdf_utils["grad_m_a"],
            m_a=self.nlpdf_utils["m_a"],
            grad_s_a2=self.nlpdf_utils["grad_s_a2"],
            s_a2=self.nlpdf_utils["s_a2"],
        )
        # g_au = np.nan_to_num(g_au)
        return g_au

    def gradient_neglog_pdf_mc(
        self,
    ) -> np.ndarray:
        log_f_Var_m_m_log_omega = (
            self.forward_map_evals["log_f_Var"]
            + self.nlpdf_utils["m_m"]
            - np.log(self.nlpdf_utils["omega"])
        )

        u_1 = utils.norm_pdf_cdf_ratio(
            -log_f_Var_m_m_log_omega / self.nlpdf_utils["s_m"]
        )  # (N, L)
        u_2 = (1 / self.nlpdf_utils["s_m2"])[:, None, :] * (
            (self.forward_map_evals["grad_log_f_Var"] + self.nlpdf_utils["grad_m_m"])
            * self.nlpdf_utils["s_m"][:, None, :]
            - log_f_Var_m_m_log_omega[:, None, :] * self.nlpdf_utils["grad_s_m"]
        )  # (N, D, L)

        grad_ = u_1[:, None, :] * u_2  # (N, D, L)
        # grad_ = np.nan_to_num(grad_)
        # assert np.sum(np.isnan(grad_)) == 0, f"grad_ac : {np.sum(np.isnan(grad_))}"
        return grad_

    def gradient_neglog_pdf_mu(
        self,
    ) -> np.ndarray:
        g_mu = gradient_cost_mu(
            log_y=self.nlpdf_utils["log_y"],
            grad_log_f_Var=self.forward_map_evals["grad_log_f_Var"],
            log_f_Var=self.forward_map_evals["log_f_Var"],
            grad_m_m=self.nlpdf_utils["grad_m_m"],
            m_m=self.nlpdf_utils["m_m"],
            grad_s_m2=self.nlpdf_utils["grad_s_m2"],
            s_m2=self.nlpdf_utils["s_m2"],
        )
        # g_mu = np.nan_to_num(g_mu)
        return g_mu

    def hess_diag_neglog_pdf(self) -> np.ndarray:
        r"""[summary]

        [extended_summary]

        Parameters
        ----------

        Returns
        -------
        np.ndarray of shape (N, D)
            [description]
        """
        hess_diag = np.where(
            (self.nlpdf_utils["y"] == self.nlpdf_utils["omega"])[:, None, :],
            # censored
            # hess_diag_ac,
            (
                self.nlpdf_utils["lambda_"][:, None, :]
                * self.nlpdf_utils["hess_diag_ac"]
                + self.nlpdf_utils["hess_diag_lambda_"]
                * self.nlpdf_utils["nll_ac"][:, None, :]
                + 2 * self.nlpdf_utils["grad_lambda_"] * self.nlpdf_utils["grad_nll_ac"]
                #
                + (1 - self.nlpdf_utils["lambda_"])[:, None, :]
                * self.nlpdf_utils["hess_diag_mc"]
                - self.nlpdf_utils["hess_diag_lambda_"]
                * self.nlpdf_utils["nll_mc"][:, None, :]
                - 2 * self.nlpdf_utils["grad_lambda_"] * self.nlpdf_utils["grad_nll_mc"]
            ),
            # uncensored
            (
                self.nlpdf_utils["lambda_"][:, None, :]
                * self.nlpdf_utils["hess_diag_au"]
                + self.nlpdf_utils["hess_diag_lambda_"]
                * self.nlpdf_utils["nll_au"][:, None, :]
                + 2 * self.nlpdf_utils["grad_lambda_"] * self.nlpdf_utils["grad_nll_au"]
                #
                + (1 - self.nlpdf_utils["lambda_"])[:, None, :]
                * self.nlpdf_utils["hess_diag_mu"]
                - self.nlpdf_utils["hess_diag_lambda_"]
                * self.nlpdf_utils["nll_mu"][:, None, :]
                - 2 * self.nlpdf_utils["grad_lambda_"] * self.nlpdf_utils["grad_nll_mu"]
            ),
        )  # (N, D, L)

        if self.apply_normalization:
            hess_diag += self.nlpdf_utils["hess_diag_log_Z"]
        hess_diag = np.nan_to_num(hess_diag)

        if self.nlpdf_utils["k_mtm"] > 0:
            hess_diag = hess_diag.reshape(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.D, self.L)
            )

        return np.sum(
            hess_diag, axis=-1
        )  # / (self.N * self.L). Output of shape (N, D) or (N, k_mtm, D)

    def hess_diag_neglog_pdf_ac(
        self,
    ) -> np.ndarray:
        n_pix = self.nlpdf_utils["n_pix"]

        f_Var_m_a_omega = (
            self.forward_map_evals["f_Var"]
            + self.nlpdf_utils["m_a"]
            - self.nlpdf_utils["omega"]
        )
        grad_f_Var_grad_m_a = (
            self.forward_map_evals["grad_f_Var"] + self.nlpdf_utils["grad_m_a"]
        )

        u_1 = utils.norm_pdf_cdf_ratio(
            -f_Var_m_a_omega / self.nlpdf_utils["s_a"]
        )  # (N, L)
        assert u_1.shape == (n_pix, self.L)

        u_2 = (1 / self.nlpdf_utils["s_a2"])[:, None, :] * (
            grad_f_Var_grad_m_a * self.nlpdf_utils["s_a"][:, None, :]
            - f_Var_m_a_omega[:, None, :] * self.nlpdf_utils["grad_s_a"]
        )  # (N, D, L)
        assert u_2.shape == (n_pix, self.D, self.L)

        grad_u_1 = (
            u_2 * (u_1 * (-f_Var_m_a_omega / self.nlpdf_utils["s_a"] + u_1))[:, None, :]
        )  # (N, D, L)
        assert grad_u_1.shape == (n_pix, self.D, self.L)

        grad_u_2 = (1 / self.nlpdf_utils["s_a2"] ** 2)[:, None, :] * (
            self.nlpdf_utils["s_a2"][:, None, :]
            * (
                self.nlpdf_utils["s_a"][:, None, :]
                * (
                    self.forward_map_evals["hess_diag_f_Var"]
                    + self.nlpdf_utils["hess_diag_m_a"]
                )
                - f_Var_m_a_omega[:, None, :] * self.nlpdf_utils["hess_diag_s_a"]
            )
            - self.nlpdf_utils["grad_s_a2"]
            * (
                self.nlpdf_utils["s_a"][:, None, :] * grad_f_Var_grad_m_a
                - f_Var_m_a_omega[:, None, :] * self.nlpdf_utils["grad_s_a"]
            )
        )  # (N, D, L)
        assert grad_u_2.shape == (n_pix, self.D, self.L)

        hess_diag = grad_u_1 * u_2 + u_1[:, None, :] * grad_u_2
        # hess_diag = np.nan_to_num(hess_diag)
        return hess_diag

    def hess_diag_neglog_pdf_au(
        self,
    ) -> np.ndarray:
        f_Var_m_a_y = (
            self.forward_map_evals["f_Var"]
            + self.nlpdf_utils["m_a"]
            - self.nlpdf_utils["y"]
        )  # (N, L)
        grad_f_Var_grad_m_a = (
            self.forward_map_evals["grad_f_Var"] + self.nlpdf_utils["grad_m_a"]
        )  # (N, D, L)

        u_1 = f_Var_m_a_y / self.nlpdf_utils["s_a2"] ** 2  # (N, L)

        grad_u_1 = (1 / self.nlpdf_utils["s_a2"] ** 4)[:, None, :] * (
            grad_f_Var_grad_m_a * (self.nlpdf_utils["s_a2"] ** 2)[:, None, :]
            - 2
            * (f_Var_m_a_y * self.nlpdf_utils["s_a2"])[:, None, :]
            * self.nlpdf_utils["grad_s_a2"]
        )  # (N, D, L)

        u_2 = (
            2 * self.nlpdf_utils["s_a2"][:, None, :] * grad_f_Var_grad_m_a
            - f_Var_m_a_y[:, None, :] * self.nlpdf_utils["grad_s_a2"]
        )  # (N, D, L)

        grad_u_2 = (
            2
            * self.nlpdf_utils["s_a2"][:, None, :]
            * (
                self.forward_map_evals["hess_diag_f_Var"]
                + self.nlpdf_utils["hess_diag_m_a"]
            )
            #
            + self.nlpdf_utils["grad_s_a2"] * grad_f_Var_grad_m_a
            #
            - f_Var_m_a_y[:, None, :] * self.nlpdf_utils["hess_diag_s_a2"]
        )  # (N, D, L)

        hess_diag = (
            0.5
            * (
                self.nlpdf_utils["hess_diag_s_a2"]
                * self.nlpdf_utils["s_a2"][:, None, :]
                - self.nlpdf_utils["grad_s_a2"] ** 2
            )
            / (self.nlpdf_utils["s_a2"] ** 2)[:, None, :]
        )  # (N, D, L)

        hess_diag += 0.5 * (
            grad_u_1 * u_2 + u_1[:, None, :] * grad_u_2
        )  # hess_diag = np.nan_to_num(hess_diag)
        return hess_diag

    def hess_diag_neglog_pdf_mc(
        self,
    ) -> np.ndarray:
        n_pix = self.nlpdf_utils["n_pix"]

        log_f_Var_m_m_log_omega = (
            self.forward_map_evals["log_f_Var"]
            + self.nlpdf_utils["m_m"]
            - self.nlpdf_utils["omega"]
        )
        grad_log_f_Var_grad_m_m = (
            self.forward_map_evals["grad_log_f_Var"] + self.nlpdf_utils["grad_m_m"]
        )

        u_1 = utils.norm_pdf_cdf_ratio(
            -log_f_Var_m_m_log_omega / self.nlpdf_utils["s_m"]
        )  # (N, L)
        assert u_1.shape == (n_pix, self.L)

        u_2 = (1 / self.nlpdf_utils["s_m2"])[:, None, :] * (
            grad_log_f_Var_grad_m_m * self.nlpdf_utils["s_m"][:, None, :]
            - log_f_Var_m_m_log_omega[:, None, :] * self.nlpdf_utils["grad_s_m"]
        )  # (N, D, L)
        assert u_2.shape == (n_pix, self.D, self.L)

        grad_u_1 = (
            u_2
            * (u_1 * (-log_f_Var_m_m_log_omega / self.nlpdf_utils["s_m"] + u_1))[
                :, None, :
            ]
        )  # (N, D, L)
        assert grad_u_1.shape == (n_pix, self.D, self.L)

        grad_u_2 = (1 / self.nlpdf_utils["s_m2"] ** 2)[:, None, :] * (
            self.nlpdf_utils["s_m2"][:, None, :]
            * (
                self.nlpdf_utils["s_m"][:, None, :]
                * (
                    self.forward_map_evals["hess_diag_f_Var"]
                    + self.nlpdf_utils["hess_diag_m_m"]
                )
                - log_f_Var_m_m_log_omega[:, None, :]
                * self.nlpdf_utils["hess_diag_s_m"]
            )
            - self.nlpdf_utils["grad_s_m2"]
            * (
                self.nlpdf_utils["s_m"][:, None, :] * grad_log_f_Var_grad_m_m
                - log_f_Var_m_m_log_omega[:, None, :] * self.nlpdf_utils["grad_s_m"]
            )
        )  # (N, D, L)
        assert grad_u_2.shape == (n_pix, self.D, self.L)

        hess_diag = grad_u_1 * u_2 + u_1[:, None, :] * grad_u_2
        # hess_diag = np.nan_to_num(hess_diag)
        return hess_diag

    def hess_diag_neglog_pdf_mu(
        self,
    ) -> np.ndarray:
        log_f_Var_m_m_log_y = (
            self.forward_map_evals["log_f_Var"]
            + self.nlpdf_utils["m_m"]
            - self.nlpdf_utils["log_y"]
        )  # (N, L)
        grad_log_f_Var_grad_m_m = (
            self.forward_map_evals["grad_log_f_Var"] + self.nlpdf_utils["grad_m_m"]
        )  # (N, D, L)

        u_1 = log_f_Var_m_m_log_y / self.nlpdf_utils["s_m2"] ** 2  # (N, L)

        grad_u_1 = (1 / self.nlpdf_utils["s_m2"] ** 4)[:, None, :] * (
            grad_log_f_Var_grad_m_m * (self.nlpdf_utils["s_m2"] ** 2)[:, None, :]
            - 2
            * (log_f_Var_m_m_log_y * self.nlpdf_utils["s_m2"])[:, None, :]
            * self.nlpdf_utils["grad_s_m2"]
        )  # (N, D, L)

        u_2 = (
            2 * self.nlpdf_utils["s_m2"][:, None, :] * grad_log_f_Var_grad_m_m
            - log_f_Var_m_m_log_y[:, None, :] * self.nlpdf_utils["grad_s_m2"]
        )  # (N, D, L)

        grad_u_2 = 2 * (
            self.nlpdf_utils["grad_s_m2"] * grad_log_f_Var_grad_m_m
            + self.nlpdf_utils["s_m2"][:, None, :]
            * (
                self.forward_map_evals["hess_diag_log_f_Var"]
                + self.nlpdf_utils["hess_diag_m_m"]
            )
        ) - (
            self.nlpdf_utils["grad_s_m2"] * grad_log_f_Var_grad_m_m
            + log_f_Var_m_m_log_y[:, None, :] * self.nlpdf_utils["hess_diag_s_m2"]
        )  # (N, D, L)

        hess_diag = (
            0.5
            * (
                self.nlpdf_utils["hess_diag_s_m2"]
                * self.nlpdf_utils["s_m2"][:, None, :]
                - self.nlpdf_utils["grad_s_m2"] ** 2
            )
            / (self.nlpdf_utils["s_m2"] ** 2)[:, None, :]
        )  # (N, D, L)

        hess_diag += 0.5 * (
            grad_u_1 * u_2 + u_1[:, None, :] * grad_u_2
        )  # hess_diag = np.nan_to_num(hess_diag)
        return hess_diag

    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[np.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        mtm: bool = False,
        **kwargs: dict,
    ) -> None:
        shape_var = current[self.var_name]["var"].shape
        assert isinstance(mtm, bool), f"mtm should be a boolean, got {type(mtm)}"

        forward_var_inputs = current[self.var_name]["var"]
        if idx_pix is not None:
            forward_var_inputs = forward_var_inputs[idx_pix]
        if mtm:
            forward_var_inputs = forward_var_inputs.reshape(-1, *shape_var[2:])
        self.evaluate_all_forward_map(
            forward_var_inputs,
            compute_derivatives=compute_derivatives,
            compute_derivatives_2nd_order=compute_derivatives_2nd_order,
        )  # TODO: put variable and other required arguments here

        assert np.sum(np.isnan(self.forward_map_evals["log_f_Var"])) == 0, (
            f"before entering bias and std : {np.sum(np.isnan(self.forward_map_evals['log_f_Var']))}"
        )

        self.nlpdf_utils = {}

        self.nlpdf_utils["mtm"] = mtm

        n_pix = idx_pix.size if idx_pix is not None else self.N * 1
        k_mtm = shape_var[1] if mtm else 0
        if n_pix == self.N:
            idx_pix = np.arange(self.N)
        N_pix = self.forward_map_evals["f_Var"].shape[0]
        if mtm:
            assert n_pix * k_mtm == N_pix
        else:
            assert n_pix == N_pix

        self.nlpdf_utils["n_pix"] = n_pix
        self.nlpdf_utils["N_pix"] = N_pix
        self.nlpdf_utils["k_mtm"] = k_mtm

        # * bias and variance
        if mtm is False:
            sigma_a = self.sigma_a[idx_pix] * 1
            sigma_m = self.sigma_m[idx_pix] * 1
            y = self.y[idx_pix] * 1
            log_y = self.log_y[idx_pix] * 1
            omega = self.omega[idx_pix] * 1
        else:
            sigma_a = np.zeros((n_pix, k_mtm, self.L))
            sigma_m = np.zeros((n_pix, k_mtm, self.L))
            y = np.zeros(
                (n_pix, k_mtm, self.L)
            )  # NOTE: y is reshaped in order to match the new dimension that comes with the MTM kernel. Nevertheless, self.y is never modified.
            omega = np.zeros((n_pix, k_mtm, self.L))

            for i, i_pix in enumerate(idx_pix):
                sigma_a[i, :, :] = self.sigma_a[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                sigma_m[i, :, :] = self.sigma_m[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                y[i, :, :] = self.y[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                omega[i, :, :] = self.omega[i_pix, :][None, :] * np.ones(
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
        self.nlpdf_utils["censored_mask"] = (y <= omega) * 1
        self.nlpdf_utils["omega"] = omega * 1
        self.nlpdf_utils["sigma_a"] = sigma_a * 1
        self.nlpdf_utils["sigma_m"] = sigma_m * 1
        self.nlpdf_utils["y"] = y * 1
        self.nlpdf_utils["log_y"] = log_y * 1

        sigma_m2 = sigma_m**2
        log_combination = (
            np.log(sigma_a) - self.forward_map_evals["log_f_Var"] - sigma_m2 / 2
        )
        exp_m2_log_combin = np.exp(-2 * log_combination)
        exp_sigma_m_squared = np.exp(sigma_m2)
        # exp_sigma_m_squared_div2_m1 = np.exp(sigma_m2 / 2) - 1  # (N, L)

        # * computation of bias and variances
        self.nlpdf_utils["m_a"] = np.zeros_like(y)
        self.nlpdf_utils["s_a"] = np.sqrt(
            sigma_a**2
            + (exp_sigma_m_squared - 1)
            * np.exp(2 * self.forward_map_evals["log_f_Var"])
        )
        self.nlpdf_utils["s_a2"] = self.nlpdf_utils["s_a"] ** 2

        self.nlpdf_utils["m_m"] = -0.5 * (sigma_m2 + np.log(1 + 1 / exp_m2_log_combin))

        self.nlpdf_utils["s_m2"] = -2 * self.nlpdf_utils["m_m"]
        self.nlpdf_utils["s_m"] = np.sqrt(self.nlpdf_utils["s_m2"])

        assert self.nlpdf_utils["s_m"].min() > 0, (
            f"{self.nlpdf_utils['s_m'].min()}, {(sigma_m2 - 2 * self.nlpdf_utils['m_m']).min()}"
        )

        if compute_derivatives:
            self.nlpdf_utils["grad_m_a"] = np.zeros((N_pix, self.D, self.L))

            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_m_a"] = np.zeros((N_pix, self.D, self.L))

            self.nlpdf_utils["grad_s_a2"] = (
                2
                * (self.forward_map_evals["f_Var"] * (exp_sigma_m_squared - 1))[
                    :, None, :
                ]
                * self.forward_map_evals["grad_f_Var"]
            )  # (N, D, L)

            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_s_a2"] = (
                    2
                    * (exp_sigma_m_squared - 1)[:, None, :]
                    * (
                        self.forward_map_evals["f_Var"][:, None, :]
                        * self.forward_map_evals["hess_diag_f_Var"]
                        + self.forward_map_evals["grad_f_Var"] ** 2
                    )
                )  # (N, D, L)

            self.nlpdf_utils["grad_s_a"] = (
                1 / (2 * self.nlpdf_utils["s_a"])[:, None, :]
            ) * self.nlpdf_utils["grad_s_a2"]

            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_s_a"] = (
                    1 / (2 * self.nlpdf_utils["s_a2"])
                )[:, None, :] * (
                    self.nlpdf_utils["hess_diag_s_a2"]
                    * self.nlpdf_utils["s_a"][:, None, :]
                    - self.nlpdf_utils["grad_s_a2"] * self.nlpdf_utils["grad_s_a"]
                )  # (N, D, L)

            self.nlpdf_utils["grad_m_m"] = (
                1 / (1 + exp_m2_log_combin) / self.forward_map_evals["f_Var"]
            )[:, None, :] * self.forward_map_evals["grad_f_Var"]  # (N, D, L)

            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_m_m"] = (
                    1 / (self.forward_map_evals["f_Var"] * (1 + exp_m2_log_combin)) ** 2
                )[:, None, :] * (
                    self.forward_map_evals["hess_diag_f_Var"]
                    * (self.forward_map_evals["f_Var"] * (1 + exp_m2_log_combin))[
                        :, None, :
                    ]
                    - self.forward_map_evals["grad_f_Var"] ** 2
                    * (1 + 3 * exp_m2_log_combin)[:, None, :]
                )  # (N, D, L)

            self.nlpdf_utils["grad_s_m2"] = (
                -2 * self.nlpdf_utils["grad_m_m"]
            )  # (N, D, L)

            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_s_m2"] = (
                    -2 * self.nlpdf_utils["hess_diag_m_m"]
                )  # (N, D, L)

            self.nlpdf_utils["grad_s_m"] = (1 / (2 * self.nlpdf_utils["s_m"]))[
                :, None, :
            ] * self.nlpdf_utils["grad_s_m2"]  # (N, D, L)

            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_s_m"] = (
                    1 / (2 * self.nlpdf_utils["s_m2"])
                )[:, None, :] * (
                    self.nlpdf_utils["hess_diag_s_m2"]
                    * self.nlpdf_utils["s_m"][:, None, :]
                    - self.nlpdf_utils["grad_s_m"] * self.nlpdf_utils["grad_s_m2"]
                )  # (N, D, L)

            # assert np.sum(np.isnan(self.forward_map_evals["f_Var"])) == 0.0
            # assert (
            #     self.forward_map_evals["f_Var"].min() > 0.0
            # ), f"{self.forward_map_evals['f_Var'].min()}, {self.forward_map_evals['log_f_Var'].min()}"
            # assert np.sum(np.isnan(self.forward_map_evals["grad_f_Var"])) == 0.0
            # assert np.sum(np.isnan(self.forward_map_evals["hess_diag_f_Var"])) == 0.0

            # assert np.sum(np.isnan(self.forward_map_evals["log_f_Var"])) == 0.0
            # assert np.sum(np.isnan(self.forward_map_evals["grad_log_f_Var"])) == 0.0
            # assert np.sum(np.isnan(self.forward_map_evals["hess_diag_log_f_Var"])) == 0.0

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

        # * mixture weight and normalization
        if self.apply_normalization:
            self.nlpdf_utils["log_Z"] = self.normalization(idx_pix=idx_pix, mtm=mtm)
        self.nlpdf_utils["lambda_"] = self.model_mixing_param(idx_pix, mtm)

        if compute_derivatives:
            self.nlpdf_utils["grad_lambda_"] = self.grad_model_mixing_param(
                idx_pix, mtm
            )
            if self.apply_normalization:
                self.nlpdf_utils["grad_log_Z"] = self.grad_normalization(
                    idx_pix=idx_pix, mtm=mtm
                )
            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_lambda_"] = (
                    self.hess_diag_model_mixing_param(idx_pix, mtm)
                )
                if self.apply_normalization:
                    self.nlpdf_utils["hess_diag_log_Z"] = self.hess_diag_normalization(
                        idx_pix=idx_pix, mtm=mtm
                    )

        self.nlpdf_utils["nll_au"] = self.neglog_pdf_au(y)
        self.nlpdf_utils["nll_ac"] = self.neglog_pdf_ac(y)
        self.nlpdf_utils["nll_mu"] = self.neglog_pdf_mu(log_y)
        self.nlpdf_utils["nll_mc"] = self.neglog_pdf_mc(np.log(omega))

        if compute_derivatives:
            self.nlpdf_utils["grad_nll_au"] = self.gradient_neglog_pdf_au()
            self.nlpdf_utils["grad_nll_mu"] = self.gradient_neglog_pdf_mu()
            self.nlpdf_utils["grad_nll_ac"] = self.gradient_neglog_pdf_ac()
            self.nlpdf_utils["grad_nll_mc"] = self.gradient_neglog_pdf_mc()

            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_ac"] = self.hess_diag_neglog_pdf_ac()
                self.nlpdf_utils["hess_diag_au"] = self.hess_diag_neglog_pdf_au()
                self.nlpdf_utils["hess_diag_mc"] = self.hess_diag_neglog_pdf_mc()
                self.nlpdf_utils["hess_diag_mu"] = self.hess_diag_neglog_pdf_mu()
