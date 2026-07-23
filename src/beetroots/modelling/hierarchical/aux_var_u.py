from typing import Optional, Union

from beetroots.modelling.hierarchical.abstract_hierarchical import Hierarchical
from beetroots.modelling.likelihoods import utils
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed

from scipy.special import log_ndtr


class AuxiliaryGivenTarget(Hierarchical):
    r"""Class implementing the conditional distribution :math:`U|\Theta` for both full conditional distributions :math:`\pi(U|\Theta, Y)` and :math:`\pi(\Theta|U, Y)`."""

    def __init__(
        self,
        forward_map,
        D: int,
        L: int,
        N: int,
        var_names_dict: dict,
        sigma_m: Union[float, xp.ndarray],
        **kwargs,
    ) -> None:
        self.forward_map = forward_map
        self.forward_map_evals = {}
        """dict: forward map evaluations (log and derivatives)"""

        self.D = D
        self.L = L
        self.N = N

        if isinstance(sigma_m, (float, int)):
            self.sigma_m = sigma_m * xp.ones((self.N, self.L))
        else:
            assert sigma_m.shape == (self.N, self.L)
            self.sigma_m = sigma_m
        self.sigma_m2 = xp.square(self.sigma_m)
        """xp.ndarray: variance of the multiplicative noise applied to the forward map"""

        self.var_names_dict = var_names_dict
        """dict: dictionary containing the variable names for the auxiliary and target variables as they are called in the model."""
        if set(self.var_names_dict.keys()) != set(["aux", "target"]) or not isinstance(
            var_names_dict, dict
        ):
            raise ValueError("var_names_dict must contain the keys 'aux' and 'target'.")

    def neglog_pdf(
        self, pixelwise: bool = False, full: bool = False, **kwargs
    ) -> Union[float, xp.ndarray]:
        k_mtm = self.nlpdf_utils["k_mtm"]
        n_pix = self.nlpdf_utils["n_pix"]
        N_pix = self.nlpdf_utils["N_pix"]

        assert xp.sum([pixelwise, full]) <= 1

        out = xp.zeros((N_pix, self.L))
        out += self.nlpdf_utils["log_aux"]

        out += xp.square(
            self.nlpdf_utils["log_aux"]
            + self.nlpdf_utils["sigma_m2"] / 2
            - self.forward_map_evals["log_f_Var"]
        ) / (2 * self.nlpdf_utils["sigma_m2"])

        if full:
            assert out.shape == (N_pix, self.L)
            if k_mtm > 0:
                out = out.reshape(n_pix, k_mtm, *out.shape[1:])
            return out
        elif pixelwise:
            out = xp.sum(out, axis=-1)
            assert out.shape == (N_pix,)
            if k_mtm > 0:
                out = out.reshape(n_pix, k_mtm)
            return out
        else:
            if k_mtm > 0:
                out = out.reshape(n_pix, k_mtm, *out.shape[1:])
                out = out.swapaxes(0, 1).sum(axis=tuple(range(1, out.ndim)))
            else:
                out = out.sum()

        return out

    def gradient_neglog_pdf_wrt_aux(
        self,
    ) -> Union[float, xp.ndarray]:
        """Computes the gradient of the negative log-pdf with respect to the auxiliary variable.

        Returns:
            Union[float, xp.ndarray]: array of shape (N, L)
        """
        n_pix = self.nlpdf_utils["n_pix"]
        out = xp.zeros((n_pix, self.L))

        out += (
            3 / 2
            + (self.nlpdf_utils["log_aux"] - self.forward_map_evals["log_f_Var"])
            / self.nlpdf_utils["sigma_m2"]
        )
        out /= self.nlpdf_utils["aux"]

        assert xp.sum(~xp.isfinite(out)) == 0.0

        return out

    def gradient_neglog_pdf_wrt_target(
        self,
    ) -> Union[float, xp.ndarray]:
        """Computes the gradient of the negative log-pdf with respect to the target variable.

        Returns:
            Union[float, xp.ndarray]: array of shape (N, D)
        """
        n_pix = self.nlpdf_utils["n_pix"]
        out = xp.zeros((n_pix, self.D, self.L))

        out -= self.forward_map_evals["grad_log_f_Var"] * xp.expand_dims(
            self.nlpdf_utils["log_aux"]
            + self.nlpdf_utils["sigma_m2"] / 2
            - self.forward_map_evals["log_f_Var"],
            axis=1,
        )
        out /= xp.expand_dims(self.nlpdf_utils["sigma_m2"], axis=1)

        return out.sum(axis=-1)  # sum over the last axis L, outputs a (N, D) array

    def gradient_neglog_pdf(
        self,
        deriv_var_name: str,
        pixelwise: bool = False,
        full: bool = False,
        **kwargs: dict,
    ) -> Union[float, xp.ndarray]:
        """Computes the gradient of the negative log-pdf with respect to the auxiliary or target variable.

        Args:
            deriv_var_name (str): variable name for which the gradient is computed
            pixelwise (bool, optional): if the gradient of the nlpdf is summed up for eadh pixel. Defaults to False.
            full (bool, optional): keeps the full information (each dimension is kept). Defaults to False.

        Raises:
            ValueError: variable name for differentiation must be either "aux" or "target"

        Returns:
            Union[float, xp.ndarray]: array of shape (n_pix, [k_mtm], L) or (n_pix, [k_mtm], D) if full else if pixelwise array of shape (N,) or (N, [k_mtm]) else float or array of shape (k_mtm,)
        """
        n_pix = self.nlpdf_utils["n_pix"]

        if deriv_var_name == self.var_names_dict["aux"]:
            out = self.gradient_neglog_pdf_wrt_aux()
        elif deriv_var_name == self.var_names_dict["target"]:
            out = self.gradient_neglog_pdf_wrt_target()
        else:
            raise ValueError(
                f"deriv_var_name must be either {self.var_names_dict['aux']} or {self.var_names_dict['target']}"
            )

        # if full:
        #     if deriv_var_name == self.var_names_dict["aux"]:
        #         assert out.shape == (n_pix, self.L)
        #     elif deriv_var_name == self.var_names_dict["target"]:
        #         assert out.shape == (n_pix, self.D)
        #     return out
        # elif pixelwise:
        #     out = xp.sum(out, axis=tuple(range(1, out.ndim)))
        #     assert out.shape == (n_pix,)
        #     return out
        # else:
        #     return xp.sum(out)

        return out

    def hess_diag_neglog_pdf_wrt_aux(
        self,
    ) -> Union[float, xp.ndarray]:
        n_pix = self.nlpdf_utils["n_pix"]
        out = xp.zeros((n_pix, self.L))

        out += (
            3 / 2
            + (self.nlpdf_utils["log_aux"] - self.forward_map_evals["log_f_Var"] - 1)
            / self.nlpdf_utils["sigma_m2"]
        )
        out /= -(self.nlpdf_utils["aux"] ** 2)

        return out

    def hess_diag_neglog_pdf_wrt_target(
        self,
    ) -> Union[float, xp.ndarray]:
        n_pix = self.nlpdf_utils["n_pix"]
        out = xp.zeros((n_pix, self.D, self.L))

        out -= self.forward_map_evals["hess_diag_log_f_Var"] * xp.expand_dims(
            self.nlpdf_utils["log_aux"]
            + self.nlpdf_utils["sigma_m2"] / 2
            - self.forward_map_evals["log_f_Var"],
            axis=1,
        )
        out += self.forward_map_evals["grad_log_f_Var"] ** 2
        out /= xp.expand_dims(self.nlpdf_utils["sigma_m2"], axis=1)

        return out.sum(axis=-1)  # sum over the last axis L, outputs a (N, D) array

    def hess_diag_neglog_pdf(
        self,
        deriv_var_name: str,
        pixelwise: bool = False,
        full: bool = False,
    ) -> Union[float, xp.ndarray]:
        """Computes the diagonal of the Hessian of the negative log-pdf with respect to the auxiliary or target variable.

        Args:
            deriv_var_name (str): variable name for which the Hessian is computed
            pixelwise (bool, optional): if the diagonal of the hessian of the nlpdf is summed up for eadh pixel. Defaults to False.
            full (bool, optional): keeps the full information (each dimension is kept). Defaults to False.

        Raises:
            ValueError: variable name for differentiation must be either "aux" or "target"

        Returns:
            Union[float, xp.ndarray]: array of shape (n_pix, [k_mtm], L) or (n_pix, [k_mtm], D) if full else if pixelwise array of shape (N,) or (N, [k_mtm]) else float or array of shape (k_mtm,)
        """

        n_pix = self.nlpdf_utils["n_pix"]
        k_mtm = self.nlpdf_utils["k_mtm"]

        if deriv_var_name == self.var_names_dict["aux"]:
            out = self.hess_diag_neglog_pdf_wrt_aux()
        elif deriv_var_name == self.var_names_dict["target"]:
            out = self.hess_diag_neglog_pdf_wrt_target()
        else:
            raise ValueError(
                f"deriv_var_name must be either {self.var_names_dict['aux']} or {self.var_names_dict['target']}"
            )

        # if full:
        #     if deriv_var_name == self.var_names_dict["aux"]:
        #         assert out.shape == (n_pix, self.L)
        #     elif deriv_var_name == self.var_names_dict["target"]:
        #         assert out.shape == (n_pix, self.D)
        #     return out
        # elif pixelwise:
        #     out = xp.sum(out, axis=tuple(range(1, out.ndim)))
        #     assert out.shape == (n_pix,)
        #     return out
        # else:
        #     return xp.sum(out)

        return out

    def evaluate_all_forward_map(
        self,
        Var: xp.ndarray,
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
    ) -> dict[str, Union[float, xp.ndarray]]:
        assert len(Var.shape) == 2 and Var.shape[1] == self.D

        forward_map_evals = self.forward_map.compute_all(
            Var, True, True, compute_derivatives, compute_derivatives_2nd_order
        )
        self.forward_map_evals = forward_map_evals

    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        mtm: bool = False,
        deriv_var_name: str = None,
    ) -> dict:
        shape_var_aux = current[self.var_names_dict["aux"]]["var"].shape
        shape_var_target = current[self.var_names_dict["target"]]["var"].shape
        assert isinstance(mtm, bool), f"mtm should be a boolean, got {type(mtm)}"

        self.nlpdf_utils = dict()
        self.nlpdf_utils["mtm"] = mtm * 1

        self.nlpdf_utils["aux"] = current[self.var_names_dict["aux"]]["var"] * 1
        if idx_pix is not None:
            self.nlpdf_utils["aux"] = self.nlpdf_utils["aux"][idx_pix] * 1
        if mtm:
            self.nlpdf_utils["aux"] = (
                self.nlpdf_utils["aux"].reshape(-1, *shape_var_aux[2:]) * 1
            )
        self.nlpdf_utils["log_aux"] = xp.log(self.nlpdf_utils["aux"]) * 1

        forward_var_inputs = current[self.var_names_dict["target"]]["var"] * 1
        if idx_pix is not None:
            forward_var_inputs = forward_var_inputs[idx_pix] * 1
        if mtm:
            forward_var_inputs = (
                forward_var_inputs.reshape(-1, *shape_var_target[2:]) * 1
            )
        compute_derivatives_forward = (
            compute_derivatives and self.var_names_dict["target"] == deriv_var_name
        )
        compute_derivatives_2nd_order_forward = (
            compute_derivatives_2nd_order
            and self.var_names_dict["target"] == deriv_var_name
        )
        self.evaluate_all_forward_map(
            forward_var_inputs,
            compute_derivatives=compute_derivatives_forward,
            compute_derivatives_2nd_order=compute_derivatives_2nd_order_forward,
        )

        n_pix = idx_pix.size if idx_pix is not None else shape_var_aux[0] * 1
        k_mtm = shape_var_aux[1] if mtm else 0 * 1
        N_pix = self.forward_map_evals["f_Var"].shape[0] * 1

        if mtm:
            assert n_pix * k_mtm == N_pix
        else:
            assert n_pix == N_pix

        idx_pix = xp.arange(n_pix) if idx_pix is None else idx_pix
        self.nlpdf_utils["sigma_m2"] = (
            xp.repeat(self.sigma_m2[idx_pix], max(1, k_mtm), axis=0) * 1
        )

        assert self.nlpdf_utils["sigma_m2"].shape[0] == N_pix

        self.nlpdf_utils["n_pix"] = n_pix * 1
        self.nlpdf_utils["N_pix"] = N_pix * 1
        self.nlpdf_utils["k_mtm"] = k_mtm * 1

    def sample_observation_model(
        self,
        rng: xp.random.Generator = xp.random.default_rng(),
    ) -> xp.ndarray:
        out = (
            self.forward_map_evals["log_f_Var"]
            - xp.square(self.sigma_m) / 2
            + self.sigma_m
            * rng.normal(
                loc=0, scale=1, size=self.forward_map_evals["log_f_Var"].shape
            ),
        )  # scale is the std not variance, better to put everything outside to avoid mistakes
        out = xp.exp(out)

        return out


class ObservationsGivenAuxiliary(Likelihood):
    r"""Class implementing the conditional distribution :math:`Y|U` for the full conditional distribution :math:`\pi(U|\Theta, Y)`."""

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        y: xp.ndarray,
        sigma_a: Union[float, xp.ndarray],
        omega: xp.ndarray,
        var_name: str,
        distribution_name: str = "obs_given_auxiliary",
    ) -> None:
        super().__init__(forward_map=None, y=y, D=D, L=L, N=N)
        self.forward_map_evals = None

        if isinstance(sigma_a, (float, int)):
            self.sigma_a = sigma_a * xp.ones((N, L))
            self.sigma_a2 = xp.square(self.sigma_a)
        else:
            assert sigma_a.shape == (N, L)
            self.sigma_a = sigma_a
            self.sigma_a2 = xp.square(self.sigma_a)

        if isinstance(omega, (float, int)):
            self.omega = omega * xp.ones((N, L))
        else:
            assert omega.shape == (N, L)
            self.omega = omega

        assert isinstance(var_name, str)
        self.var_name = var_name
        """str: variable name for the target variable"""

        assert isinstance(distribution_name, str)
        self.name = distribution_name
        """str: name of the distribution"""

        self.censored = self.y == self.omega

    def neglog_pdf_u(
        self,
    ) -> Union[float, xp.ndarray]:
        out = (
            xp.zeros((self.nlpdf_utils["n_pix"], self.L))
            if not self.nlpdf_utils["mtm"]
            else xp.zeros(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L)
            )
        )

        out += xp.square(self.nlpdf_utils["y"] - self.nlpdf_utils["aux"])
        out /= 2 * self.nlpdf_utils["sigma_a2"]

        return out

    def neglog_pdf_c(
        self,
    ) -> Union[float, xp.ndarray]:
        out = (
            xp.zeros((self.nlpdf_utils["n_pix"], self.L))
            if not self.nlpdf_utils["mtm"]
            else xp.zeros(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L)
            )
        )

        z = self.nlpdf_utils["omega"] - self.nlpdf_utils["aux"]
        z /= self.nlpdf_utils["sigma_a"]

        out -= log_ndtr(z)

        return out

    def neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        **kwargs: dict,
    ) -> Union[float, xp.ndarray]:
        out = (
            xp.zeros((self.nlpdf_utils["n_pix"], self.L))
            if not self.nlpdf_utils["mtm"]
            else xp.zeros(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L)
            )
        )

        out += xp.where(
            (self.nlpdf_utils["omega"] == self.nlpdf_utils["y"]),
            self.nlpdf_utils["nlpdf_c"],
            self.nlpdf_utils["nlpdf_u"],
        )

        if full:
            if self.nlpdf_utils["mtm"]:
                assert out.shape == (
                    self.nlpdf_utils["n_pix"],
                    self.nlpdf_utils["k_mtm"],
                    self.L,
                )
            else:
                assert out.shape == (self.nlpdf_utils["n_pix"], self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=-1)
            if self.nlpdf_utils["mtm"]:
                assert out.shape == (
                    self.nlpdf_utils["n_pix"],
                    self.nlpdf_utils["k_mtm"],
                )
            else:
                assert out.shape == (self.nlpdf_utils["n_pix"],)
            return out
        else:
            out = (
                xp.sum(out)
                if not self.nlpdf_utils["mtm"]
                else out.swapaxes(0, 1).sum(axis=tuple(range(1, out.ndim)))
            )

        return out

    def gradient_neglog_pdf_u(
        self,
    ) -> Union[float, xp.ndarray]:
        out = (
            xp.zeros((self.nlpdf_utils["n_pix"], self.L))
            if not self.nlpdf_utils["mtm"]
            else xp.zeros(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L)
            )
        )

        out -= self.nlpdf_utils["y"] - self.nlpdf_utils["aux"]
        out /= self.nlpdf_utils["sigma_a2"]

        return out

    def gradient_neglog_pdf_c(
        self,
    ) -> Union[float, xp.ndarray]:
        out = (
            xp.zeros((self.nlpdf_utils["n_pix"], self.L))
            if not self.nlpdf_utils["mtm"]
            else xp.zeros(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L)
            )
        )

        z = self.nlpdf_utils["omega"] - self.nlpdf_utils["aux"]
        z /= self.nlpdf_utils["sigma_a"]

        out += utils.norm_pdf_cdf_ratio(z)
        out /= self.nlpdf_utils["sigma_a"]

        return out

    def gradient_neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        **kwargs: dict,
    ) -> Union[float, xp.ndarray]:
        out = (
            xp.zeros((self.nlpdf_utils["n_pix"], self.L))
            if not self.nlpdf_utils["mtm"]
            else xp.zeros(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L)
            )
        )

        out += xp.where(
            (self.nlpdf_utils["omega"] == self.nlpdf_utils["y"]),
            self.nlpdf_utils["grad_nlpdf_c"],
            self.nlpdf_utils["grad_nlpdf_u"],
        )

        # if full:
        #     if not self.nlpdf_utils['mtm']:
        #         assert out.shape == (self.nlpdf_utils['n_pix'], self.L)
        #     else:
        #         assert out.shape == (self.nlpdf_utils['n_pix'], self.nlpdf_utils['k_mtm'], self.L)
        #     return out
        # elif pixelwise:
        #     out = xp.sum(out, axis=-1)
        #     if self.nlpdf_utils['mtm']:
        #         assert out.shape == (self.nlpdf_utils['n_pix'], self.nlpdf_utils['k_mtm'])
        #     else:
        #         assert out.shape == (self.nlpdf_utils['n_pix'],)
        #     return out
        # else:
        #     out = xp.sum(out) if not self.nlpdf_utils['mtm'] else out.swapaxes(0,1).sum(axis=tuple(range(1, out.ndim)))

        return out

    def hess_diag_neglog_pdf_u(
        self,
    ) -> Union[float, xp.ndarray]:
        out = (
            xp.ones((self.nlpdf_utils["n_pix"], self.L))
            if not self.nlpdf_utils["mtm"]
            else xp.ones((self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L))
        )

        out /= self.nlpdf_utils["sigma_a2"]

        return out

    def hess_diag_neglog_pdf_c(
        self,
    ) -> Union[float, xp.ndarray]:
        out = (
            xp.zeros((self.nlpdf_utils["n_pix"], self.L))
            if not self.nlpdf_utils["mtm"]
            else xp.zeros(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L)
            )
        )

        z = self.nlpdf_utils["omega"] - self.nlpdf_utils["aux"]
        z /= self.nlpdf_utils["sigma_a"]

        norm_pdf_cdf_ratio = utils.norm_pdf_cdf_ratio(z)

        out += xp.square(norm_pdf_cdf_ratio)
        out += norm_pdf_cdf_ratio * z
        out /= self.nlpdf_utils["sigma_a2"]

        return out

    def hess_diag_neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        **kwargs: dict,
    ) -> Union[float, xp.ndarray]:
        out = (
            xp.zeros((self.nlpdf_utils["n_pix"], self.L))
            if not self.nlpdf_utils["mtm"]
            else xp.zeros(
                (self.nlpdf_utils["n_pix"], self.nlpdf_utils["k_mtm"], self.L)
            )
        )

        out += xp.where(
            (self.nlpdf_utils["omega"] == self.nlpdf_utils["y"]),
            self.nlpdf_utils["hess_diag_nlpdf_c"],
            self.nlpdf_utils["hess_diag_nlpdf_u"],
        )

        return out

    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        mtm: bool = False,
        **kwargs: dict,
    ) -> None:
        self.nlpdf_utils = dict()
        self.nlpdf_utils["aux"] = current[self.var_name]["var"] * 1
        original_var_shape = self.nlpdf_utils["aux"].shape

        assert original_var_shape[-1] == self.L

        self.nlpdf_utils["mtm"] = mtm * 1
        self.nlpdf_utils["k_mtm"] = original_var_shape[1] if mtm else 0
        self.nlpdf_utils["n_pix"] = idx_pix.size if idx_pix is not None else self.N

        idx_pix = xp.arange(self.nlpdf_utils["n_pix"]) if idx_pix is None else idx_pix

        self.nlpdf_utils["aux"] = current[self.var_name]["var"][idx_pix] * 1

        if mtm:
            self.nlpdf_utils["sigma_a"] = self.sigma_a[idx_pix, xp.newaxis, ...] * 1
            self.nlpdf_utils["sigma_a2"] = self.sigma_a2[idx_pix, xp.newaxis, ...] * 1
            self.nlpdf_utils["omega"] = self.omega[idx_pix, xp.newaxis, ...] * 1
            self.nlpdf_utils["y"] = self.y[idx_pix, xp.newaxis, ...] * 1
        else:
            self.nlpdf_utils["sigma_a"] = self.sigma_a[idx_pix] * 1
            self.nlpdf_utils["sigma_a2"] = self.sigma_a2[idx_pix] * 1
            self.nlpdf_utils["omega"] = self.omega[idx_pix] * 1
            self.nlpdf_utils["y"] = self.y[idx_pix] * 1

        self.nlpdf_utils["nlpdf_c"] = self.neglog_pdf_c() * 1
        self.nlpdf_utils["nlpdf_u"] = self.neglog_pdf_u() * 1

        if compute_derivatives:
            self.nlpdf_utils["grad_nlpdf_c"] = self.gradient_neglog_pdf_c() * 1
            self.nlpdf_utils["grad_nlpdf_u"] = self.gradient_neglog_pdf_u() * 1
            if compute_derivatives_2nd_order:
                self.nlpdf_utils["hess_diag_nlpdf_c"] = (
                    self.hess_diag_neglog_pdf_c() * 1
                )
                self.nlpdf_utils["hess_diag_nlpdf_u"] = (
                    self.hess_diag_neglog_pdf_u() * 1
                )

    def sample_observation_model(
        self,
        rng: xp.random.Generator = xp.random.default_rng(),
    ) -> xp.ndarray:
        out = (
            self.nlpdf_utils["aux"]
            + self.sigma_a
            * rng.normal(loc=0, scale=1, size=self.nlpdf_utils["aux"].shape)
        )  # scale is the std not variance, better to put everything outside to avoid mistakes

        return out
