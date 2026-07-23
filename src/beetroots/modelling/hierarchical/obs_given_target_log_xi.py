from typing import Optional, Union

from beetroots.modelling.component_distribution import ComponentDistribution
from beetroots.modelling.likelihoods import utils
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood
from beetroots.modelling.priors.abstract_prior import PriorProbaDistribution

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from scipy.special import log_ndtr


class ObservationsGivenTargetLogXi(Likelihood):
    r"""Likelihood component p(y | theta, log_xi) with u = exp(log_xi) * f(theta).
    Alternative observation model that uses a non-centered parameterization for the auxiliary variable."""

    def __init__(
        self,
        forward_map,
        D: int,
        L: int,
        N: int,
        y: xp.ndarray,
        sigma_a: Union[float, xp.ndarray],
        sigma_m: Union[float, xp.ndarray],
        omega: Union[float, xp.ndarray],
        var_names_dict: dict,
        eps: float = 1e-12,
        distribution_name: str = "obs_given_target_log_xi",
    ) -> None:
        """Initializes the likelihood component."""
        super().__init__(forward_map=forward_map, y=y, D=D, L=L, N=N)

        self.forward_map = forward_map
        self.forward_map_evals = {}

        if isinstance(sigma_a, (float, int)):
            self.sigma_a = float(sigma_a) * xp.ones((N, L))
        else:
            assert sigma_a.shape == (N, L)
            self.sigma_a = sigma_a
        self.sigma_a2 = xp.square(self.sigma_a)

        if isinstance(sigma_m, (float, int)):
            self.sigma_m = float(sigma_m) * xp.ones((N, L))
        else:
            assert sigma_m.shape == (N, L)
            self.sigma_m = sigma_m
        self.sigma_m2 = xp.square(self.sigma_m)

        if isinstance(omega, (float, int)):
            self.omega = float(omega) * xp.ones((N, L))
        else:
            assert omega.shape == (N, L)
            self.omega = omega

        if set(var_names_dict.keys()) != {"target", "log_xi"}:
            raise ValueError("var_names_dict must contain keys 'target' and 'log_xi'.")
        self.var_names_dict = var_names_dict

        self.eps = float(eps)
        self.name = distribution_name
        self.censored = self.y == self.omega

    def _reshape_like_input(self, arr: xp.ndarray) -> xp.ndarray:
        """
        Reshapes the input array to match the shape of the input variables, considering the MTM axis if present.
        """
        if not self.nlpdf_utils["mtm"]:
            return arr
        return arr.reshape(
            self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], *arr.shape[1:]
        )

    def neglog_pdf(self, pixelwise: bool = False, full: bool = False, **kwargs):
        out = xp.where(
            self.nlpdf_utils["is_censored"],
            self.nlpdf_utils["nlpdf_c"],
            self.nlpdf_utils["nlpdf_u"],
        )
        out = self._reshape_like_input(out)

        if full:
            return out
        if pixelwise:
            return out.sum(axis=-1)
        if self.nlpdf_utils["mtm"]:
            return out.swapaxes(0, 1).sum(axis=tuple(range(1, out.ndim)))
        return float(out.sum())

    def gradient_neglog_pdf(self, deriv_var_name: str, **kwargs):
        grad_du = xp.where(
            self.nlpdf_utils["is_censored"],
            self.nlpdf_utils["grad_du_c"],
            self.nlpdf_utils["grad_du_u"],
        )

        if deriv_var_name == self.var_names_dict["log_xi"]:
            out = grad_du * self.nlpdf_utils["u"]
            return self._reshape_like_input(out)

        if deriv_var_name == self.var_names_dict["target"]:
            du_dtheta = (
                xp.expand_dims(self.nlpdf_utils["u"], axis=1)
                * self.forward_map_evals["grad_log_f_Var"]
            )
            out = xp.expand_dims(grad_du, axis=1) * du_dtheta
            out = out.sum(axis=-1)
            return self._reshape_like_input(out)

        raise ValueError(
            f"deriv_var_name must be either {self.var_names_dict['target']} or {self.var_names_dict['log_xi']}"
        )

    def hess_diag_neglog_pdf(self, deriv_var_name: str, **kwargs):
        grad_du = xp.where(
            self.nlpdf_utils["is_censored"],
            self.nlpdf_utils["grad_du_c"],
            self.nlpdf_utils["grad_du_u"],
        )
        hess_du = xp.where(
            self.nlpdf_utils["is_censored"],
            self.nlpdf_utils["hess_du_c"],
            self.nlpdf_utils["hess_du_u"],
        )

        if deriv_var_name == self.var_names_dict["log_xi"]:
            u = self.nlpdf_utils["u"]
            out = hess_du * xp.square(u) + grad_du * u
            return self._reshape_like_input(out)

        if deriv_var_name == self.var_names_dict["target"]:
            grad_log_f = self.forward_map_evals["grad_log_f_Var"]
            hess_log_f = self.forward_map_evals["hess_diag_log_f_Var"]
            u = xp.expand_dims(self.nlpdf_utils["u"], axis=1)
            du_dtheta = u * grad_log_f
            d2u_dtheta2 = u * (hess_log_f + xp.square(grad_log_f))

            out = xp.expand_dims(hess_du, axis=1) * xp.square(du_dtheta)
            out += xp.expand_dims(grad_du, axis=1) * d2u_dtheta2
            out = out.sum(axis=-1)
            return self._reshape_like_input(out)

        raise ValueError(
            f"deriv_var_name must be either {self.var_names_dict['target']} or {self.var_names_dict['log_xi']}"
        )

    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        mtm: bool = False,
        deriv_var_name: str = None,
        **kwargs,
    ) -> None:
        theta_all = current[self.var_names_dict["target"]]["var"]
        log_xi_all = current[self.var_names_dict["log_xi"]]["var"]

        n_pix = theta_all.shape[0] if idx_pix is None else idx_pix.size
        idx_pix = xp.arange(n_pix) if idx_pix is None else idx_pix

        theta_sel = theta_all[idx_pix] * 1
        log_xi_sel = log_xi_all[idx_pix] * 1

        k_mtm = 0
        if mtm:
            if theta_sel.ndim == 3:
                k_mtm = theta_sel.shape[1]
            elif log_xi_sel.ndim == 3:
                k_mtm = log_xi_sel.shape[1]
            else:
                raise ValueError("mtm=True but no MTM axis found in current variables.")

            if theta_sel.ndim == 2:
                theta_sel = xp.repeat(theta_sel[:, None, :], k_mtm, axis=1)
            if log_xi_sel.ndim == 2:
                log_xi_sel = xp.repeat(log_xi_sel[:, None, :], k_mtm, axis=1)

            theta_flat = theta_sel.reshape(-1, theta_sel.shape[-1])
            log_xi_flat = log_xi_sel.reshape(-1, log_xi_sel.shape[-1])

            sigma_a = xp.repeat(self.sigma_a[idx_pix, None, :], k_mtm, axis=1).reshape(
                -1, self.L
            )
            sigma_a2 = xp.repeat(
                self.sigma_a2[idx_pix, None, :], k_mtm, axis=1
            ).reshape(-1, self.L)
            y = xp.repeat(self.y[idx_pix, None, :], k_mtm, axis=1).reshape(-1, self.L)
            omega = xp.repeat(self.omega[idx_pix, None, :], k_mtm, axis=1).reshape(
                -1, self.L
            )
        else:
            theta_flat = theta_sel
            log_xi_flat = log_xi_sel
            sigma_a = self.sigma_a[idx_pix] * 1
            sigma_a2 = self.sigma_a2[idx_pix] * 1
            y = self.y[idx_pix] * 1
            omega = self.omega[idx_pix] * 1

        compute_derivatives_forward = (
            compute_derivatives and deriv_var_name == self.var_names_dict["target"]
        )
        compute_derivatives_2nd_order_forward = (
            compute_derivatives_2nd_order
            and deriv_var_name == self.var_names_dict["target"]
        )

        self.forward_map_evals = self.forward_map.compute_all(
            theta_flat,
            compute_lin=True,
            compute_log=True,
            compute_derivatives=compute_derivatives_forward,
            compute_derivatives_2nd_order=compute_derivatives_2nd_order_forward,
        )

        f_theta = xp.clip(self.forward_map_evals["f_Var"], a_min=self.eps, a_max=None)
        exp_log_xi = xp.exp(log_xi_flat)
        u = exp_log_xi * f_theta

        z = (omega - u) / sigma_a
        is_censored = omega == y

        self.nlpdf_utils = {
            "mtm": mtm,
            "k_mtm": k_mtm,
            "n_pix": n_pix,
            "u": u,
            "is_censored": is_censored,
            "sigma_a": sigma_a,
            "sigma_a2": sigma_a2,
            "y": y,
            "omega": omega,
            "nlpdf_u": xp.square(y - u) / (2.0 * sigma_a2),
            "nlpdf_c": -log_ndtr(z),
        }

        if compute_derivatives:
            norm_ratio = utils.norm_pdf_cdf_ratio(z)

            self.nlpdf_utils["grad_du_u"] = (u - y) / sigma_a2
            self.nlpdf_utils["grad_du_c"] = norm_ratio / sigma_a

            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_du_u"] = 1.0 / sigma_a2
                self.nlpdf_utils["hess_du_c"] = (
                    xp.square(norm_ratio) + norm_ratio * z
                ) / sigma_a2

    def sample_observation_model(
        self,
        rng: xp.random.Generator = xp.random.default_rng(),
    ) -> xp.ndarray:
        if "u" not in self.nlpdf_utils:
            raise ValueError(
                "evaluate_all_nlpdf_utils must be called before sample_observation_model."
            )
        u = self.nlpdf_utils["u"]
        sigma_a = self.nlpdf_utils["sigma_a"]
        return u + sigma_a * rng.normal(loc=0.0, scale=1.0, size=u.shape)


class LogXiPrior(PriorProbaDistribution):
    r"""Independent Normal prior in log-space for log_xi."""

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        var_name: str,
        mean_log_xi: Union[float, xp.ndarray] = 0.0,
        var_log_xi: Union[float, xp.ndarray] = 1.0,
        distribution_name: str = "log_xi_prior",
    ) -> None:
        super().__init__(
            distribution_name=distribution_name, D=D, L=L, N=N, var_name=var_name
        )
        self.D = D
        self.L = L
        self.N = N
        self.var_name = var_name

        if isinstance(mean_log_xi, (float, int)):
            self.mean_log_xi = float(mean_log_xi) * xp.ones((N, L))
        else:
            assert mean_log_xi.shape == (N, L)
            self.mean_log_xi = mean_log_xi

        if isinstance(var_log_xi, (float, int)):
            self.var_log_xi = float(var_log_xi) * xp.ones((N, L))
        else:
            assert var_log_xi.shape == (N, L)
            self.var_log_xi = var_log_xi

        if xp.any(self.var_log_xi <= 0):
            raise ValueError("var_log_xi must be strictly positive.")

    def neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        paramwise: bool = False,
        **kwargs,
    ):
        out = self.nlpdf_utils["nlpdf_full"]

        if full:
            return out
        if pixelwise:
            return out.sum(axis=-1)
        if paramwise:
            return out.sum(axis=0)
        if self.nlpdf_utils["mtm"]:
            return out.swapaxes(0, 1).sum(axis=tuple(range(1, out.ndim)))
        return float(out.sum())

    def gradient_neglog_pdf(self, deriv_var_name: str = None, **kwargs):
        if deriv_var_name is not None and deriv_var_name != self.var_name:
            raise ValueError(f"deriv_var_name must be {self.var_name}.")
        return self.nlpdf_utils["grad"]

    def hess_diag_neglog_pdf(self, deriv_var_name: str = None, **kwargs):
        if deriv_var_name is not None and deriv_var_name != self.var_name:
            raise ValueError(f"deriv_var_name must be {self.var_name}.")
        return self.nlpdf_utils["hess_diag"]

    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        mtm: bool = False,
        **kwargs,
    ) -> None:
        var_all = current[self.var_name]["var"]
        n_pix = self.N if idx_pix is None else idx_pix.size
        idx_pix = xp.arange(n_pix) if idx_pix is None else idx_pix

        var_sel = var_all[idx_pix] * 1
        mean_sel = self.mean_log_xi[idx_pix] * 1
        var_prior_sel = self.var_log_xi[idx_pix] * 1

        k_mtm = var_sel.shape[1] if mtm else 0
        if mtm:
            mean_sel = xp.repeat(mean_sel[:, None, :], k_mtm, axis=1)
            var_prior_sel = xp.repeat(var_prior_sel[:, None, :], k_mtm, axis=1)

        centered = var_sel - mean_sel
        nlpdf_full = 0.5 * xp.square(centered) / var_prior_sel
        nlpdf_full += 0.5 * xp.log(2.0 * xp.pi * var_prior_sel)

        self.nlpdf_utils = {
            "mtm": mtm,
            "k_mtm": k_mtm,
            "n_pix": n_pix,
            "nlpdf_full": nlpdf_full,
        }

        if compute_derivatives:
            self.nlpdf_utils["grad"] = centered / var_prior_sel
            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag"] = 1.0 / var_prior_sel
