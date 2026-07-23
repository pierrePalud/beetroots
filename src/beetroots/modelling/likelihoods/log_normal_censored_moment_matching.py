"""Implementation of an approximation of the likelihood function of a mixture of Gaussian and multiplicative noises with censorship with a lower limit."""

from typing import Optional, Union

import numba
import numpy as np
from scipy.special import log_ndtr

from beetroots.modelling.likelihoods import utils
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood


@numba.njit()
def cost_mu(
    log_y: np.ndarray, log_f_Var: np.ndarray, m_m: np.ndarray, s_m: np.ndarray
) -> np.ndarray:
    nll_mu = 0.5 * ((log_y - log_f_Var - m_m) / s_m) ** 2 + np.log(s_m)
    # nll_mu = np.where(np.isnan(log_y) | np.isinf(log_y), 10.0 ** 8, nll_mu)

    # shape = nll_mu.shape
    # nll_mu = nll_mu.ravel()
    # nll_mu[np.isnan(nll_mu)] = 0
    # nll_mu = nll_mu.reshape(shape)
    return nll_mu


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


class CensoredLogNormMomentMatchingLikelihood(Likelihood):
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

    def neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        nlpdf = np.where(
            self.nlpdf_utils["censored_mask"],
            self.nlpdf_utils["nll_mc"],
            self.nlpdf_utils["nll_mu"],
        )  # (N, L)

        nlpdf = np.nan_to_num(nlpdf)

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
            self.nlpdf_utils["grad_nll_mc"],
            # uncensored
            self.nlpdf_utils["grad_nll_mu"],
        )  # (N, D, L)

        grad_ = np.nan_to_num(grad_)

        if self.nlpdf_utils["k_mtm"] > 0:
            grad_ = grad_.reshape(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.D, self.L)
            )

        return np.sum(grad_, axis=-1)  # / (self.N * self.L)

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
            self.nlpdf_utils["hess_diag_mc"],
            # uncensored
            self.nlpdf_utils["hess_diag_mu"],
        )  # (N, D, L)

        hess_diag = np.nan_to_num(hess_diag)

        if self.nlpdf_utils["k_mtm"] > 0:
            hess_diag = hess_diag.reshape(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.D, self.L)
            )

        return np.sum(
            hess_diag, axis=-1
        )  # / (self.N * self.L). Output of shape (N, D) or (N, k_mtm, D)

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

        assert (
            np.sum(np.isnan(self.forward_map_evals["log_f_Var"])) == 0
        ), f"before entering bias and std : {np.sum(np.isnan(self.forward_map_evals['log_f_Var']))}"

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
        self.nlpdf_utils["s_a"] = sigma_a * np.sqrt(
            (exp_sigma_m_squared - 1)
            * np.exp(2 * (self.forward_map_evals["log_f_Var"] - np.log(sigma_a)))
            + 1
        )
        self.nlpdf_utils["s_a2"] = self.nlpdf_utils["s_a"] ** 2

        self.nlpdf_utils["m_m"] = -0.5 * (sigma_m2 + np.log(1 + 1 / exp_m2_log_combin))

        self.nlpdf_utils["s_m2"] = -2 * self.nlpdf_utils["m_m"]
        self.nlpdf_utils["s_m"] = np.sqrt(self.nlpdf_utils["s_m2"])

        assert (
            self.nlpdf_utils["s_m"].min() > 0
        ), f"{self.nlpdf_utils['s_m'].min()}, {(sigma_m2 - 2 * self.nlpdf_utils['m_m']).min()}"

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
            )[:, None, :] * self.forward_map_evals[
                "grad_f_Var"
            ]  # (N, D, L)

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
            ] * self.nlpdf_utils[
                "grad_s_m2"
            ]  # (N, D, L)

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

        # * mixture weight
        self.nlpdf_utils["nll_mu"] = self.neglog_pdf_mu(log_y)
        self.nlpdf_utils["nll_mc"] = self.neglog_pdf_mc(np.log(omega))

        if compute_derivatives:
            self.nlpdf_utils["grad_nll_mu"] = self.gradient_neglog_pdf_mu()
            self.nlpdf_utils["grad_nll_mc"] = self.gradient_neglog_pdf_mc()

            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_mc"] = self.hess_diag_neglog_pdf_mc()
                self.nlpdf_utils["hess_diag_mu"] = self.hess_diag_neglog_pdf_mu()
