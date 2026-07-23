"""Implementation of LogNormal likelihood with censorship (with a lower limit)"""

from typing import Optional, Union

import numpy as np
from scipy.special import log_ndtr

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
        "log_y",
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
        sigma_m: Union[float, np.ndarray],
        sigma_a: Union[float, np.ndarray],
        omega: Union[float, np.ndarray],
        var_name: str,
        **kwargs,
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
        super().__init__(forward_map, y, D, L, N, var_name, **kwargs)
        self.log_y = np.log(self.y)

        # ! trigger an error is the mean y contains less than N elements
        if not (y.shape == (N, L)):
            raise ValueError(
                "y must have the shape (N, L) = ({}, {}) elements".format(
                    self.N, self.L
                )
            )
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
        self.log_y = np.log(self.y)

    def neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        r"""[summary]

        .. math::
            p(y_{n,\ell} \vert x) \propto \exp \left\{- [y_{n,\ell} = \omega] \Phi( \frac{\omega - f_{\ell}(x_n)}{\sigma^2} \right) - [y_{n,\ell} > \omega] \frac{\omega - f_{\ell}(x_n)}{\sigma^2} \right\}
        """

        nlpdf = np.where(
            self.nlpdf_utils["censored_mask"],
            self.neglog_pdf_c(),
            self.neglog_pdf_u(),
        )  # (N_pix, L)

        if self.nlpdf_utils["k_mtm"] > 0:
            nlpdf = nlpdf.reshape(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L)
            )  # (n_pix, k_mtm, L)

        if full:
            return nlpdf  # (N_pix, L)

        if pixelwise:
            return np.sum(nlpdf, axis=-1)  # (N_pix,)

        return (
            nlpdf.sum() if self.nlpdf_utils["k_mtm"] == 0 else nlpdf.sum(axis=(0, -1))
        )

    def neglog_pdf_c(
        self,
    ) -> np.ndarray:
        z = (
            self.nlpdf_utils["log_omega"]
            - self.forward_map_evals["log_f_Var"]
            + 0.5 * self.nlpdf_utils["sigma"] ** 2
        )
        z /= self.nlpdf_utils["sigma"]
        # z = np.nan_to_num(z)

        nll_mc = -log_ndtr(z)
        nll_mc = np.nan_to_num(nll_mc)
        return nll_mc

    def neglog_pdf_u(
        self,
    ) -> np.ndarray:
        return (
            self.nlpdf_utils["log_y"]
            - self.forward_map_evals["log_f_Var"]
            + 0.5 * self.nlpdf_utils["sigma"] ** 2
        ) ** 2 / (2 * self.nlpdf_utils["sigma"] ** 2)

    def gradient_neglog_pdf(self, **kwargs) -> np.ndarray:
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
            self.gradient_neglog_pdf_c(),
            self.gradient_neglog_pdf_u(),
        )  # (N, D, L)

        # ! issue: do not sum over L if L = D (i.e. identity forward_map)
        if not self.D == self.L:
            grad_ = np.sum(grad_, axis=2)  # (N, D)

        return grad_

    def gradient_neglog_pdf_c(
        self,
    ) -> np.ndarray:
        grad_ = (
            self.forward_map_evals["grad_log_f_Var"]
            * (
                utils.norm_pdf_cdf_ratio(
                    (
                        self.nlpdf_utils["log_omega"]
                        - self.forward_map_evals["log_f_Var"]
                        + 0.5 * self.nlpdf_utils["sigma"] ** 2
                    )
                    / self.nlpdf_utils["sigma"]
                )
                / self.nlpdf_utils["sigma"]
            )[:, None, :]
        )
        return grad_  # (N_pix, D, L)

    def gradient_neglog_pdf_u(
        self,
    ) -> np.ndarray:
        grad_ = self.forward_map_evals["grad_log_f_Var"] * np.expand_dims(
            (
                self.nlpdf_utils["log_y"]
                - self.forward_map_evals["log_f_Var"]
                + 0.5 * self.nlpdf_utils["sigma"] ** 2
            )
            / self.nlpdf_utils["sigma"] ** 2,
            axis=1,
        )
        return grad_  # (N_pix, D, L)

    def hess_diag_neglog_pdf(
        self,
        **kwargs,
    ) -> np.ndarray:
        r"""[summary]

        [extended_summary]

        Parameters
        ----------

        Returns
        -------
        np.ndarray of shape (N, D, L)
            [description]
        """
        hess_diag = np.where(
            (self.y == self.omega)[:, None, :],
            self.hess_diag_neglog_pdf_c(),
            self.hess_diag_neglog_pdf_u(),
        )  # (N, D, L)

        # ! issue: do not sum over L if L = D (i.e. identity forward_map)
        if not self.D == self.L:
            hess_diag = np.sum(hess_diag, axis=2)  # (N, D)

        return hess_diag

    def hess_diag_neglog_pdf_c(
        self,
    ) -> np.ndarray:
        hess_diag = (
            utils.norm_pdf_cdf_ratio(
                (self.nlpdf_utils["log_omega"] - self.forward_map_evals["f_Var"])
                / self.nlpdf_utils["sigma"]
            )
            / self.nlpdf_utils["sigma"]
        )[:, None, :] * (
            self.forward_map_evals["hess_diag_f_Var"]
            + self.forward_map_evals["grad_f_Var"] ** 2
            * (
                (
                    (self.nlpdf_utils["log_omega"] - self.forward_map_evals["f_Var"])
                    / self.nlpdf_utils["sigma"]
                    + utils.norm_pdf_cdf_ratio(
                        (
                            self.nlpdf_utils["log_omega"]
                            - self.forward_map_evals["f_Var"]
                        )
                        / self.nlpdf_utils["sigma"]
                    )
                )
                / self.nlpdf_utils["sigma"]
            )[:, None, :]
        )
        return hess_diag  # (N_pix, D, L)

    def hess_diag_neglog_pdf_u(
        self,
    ) -> np.ndarray:
        return (1 / self.nlpdf_utils["sigma"] ** 2)[:, None, :] * (
            self.forward_map_evals["grad_f_Var"] ** 2
            + self.forward_map_evals["hess_diag_f_Var"]
            * (self.forward_map_evals["f_Var"] - self.nlpdf_utils["log_y"])[:, None, :]
        )  # (N_pix, D, L)

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
            sigma = self.sigma[idx_pix] * 1
            y = self.y[idx_pix] * 1
            log_y = self.log_y[idx_pix] * 1
            omega = self.omega[idx_pix] * 1
        else:
            sigma = np.zeros((n_pix, k_mtm, self.L))
            y = np.zeros(
                (n_pix, k_mtm, self.L)
            )  # NOTE: y is reshaped in order to match the new dimension that comes with the MTM kernel. Nevertheless, self.y is never modified.
            omega = np.zeros((n_pix, k_mtm, self.L))

            for i, i_pix in enumerate(idx_pix):
                sigma[i, :, :] = self.sigma[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )
                y[i, :, :] = self.y[i_pix, :][None, :] * np.ones((k_mtm, self.L))
                omega[i, :, :] = self.omega[i_pix, :][None, :] * np.ones(
                    (k_mtm, self.L)
                )

            sigma = (
                sigma.transpose((2, 0, 1)).reshape((self.L, N_pix)).T
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
        self.nlpdf_utils["log_omega"] = np.log(omega) * 1
        self.nlpdf_utils["sigma"] = sigma * 1
        self.nlpdf_utils["y"] = y * 1
        self.nlpdf_utils["log_y"] = log_y * 1

        self.nlpdf_utils["nll_au"] = self.neglog_pdf_u()
        self.nlpdf_utils["nll_ac"] = self.neglog_pdf_c()

        if compute_derivatives:
            self.nlpdf_utils["grad_nll_au"] = self.gradient_neglog_pdf_u()
            self.nlpdf_utils["grad_nll_ac"] = self.gradient_neglog_pdf_c()

            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_ac"] = self.hess_diag_neglog_pdf_c()
                self.nlpdf_utils["hess_diag_au"] = self.hess_diag_neglog_pdf_u()

    def sample_observation_model(
        self,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        eps_m = rng.lognormal(
            mean=-(self.sigma**2) / 2,
            sigma=self.sigma,
        )
        f_Var = self.forward_map_evals["f_Var"] * 1
        y_rep = np.maximum(self.omega, eps_m * f_Var)
        return y_rep
