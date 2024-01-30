"""Contains a multi-line polynomial regressor made to approximate the true forward model given a grid of simulations
"""
import os
import pickle
from typing import List, Optional, Sequence, Tuple, Union

import numba as nb
import numpy as np
import pandas as pd

from beetroots.modelling.forward_maps.abstract_exp import ExpForwardMap


def compute_grad_poly_params(
    pow_arr: np.ndarray, coeff_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """computes the power array and the coefficient array of the gradient of the multivariate polynomial defined with the given power and coefficient arrays

    Parameters
    ----------
    pow_arr : np.ndarray of shape (n_coefs, D_no_kappa)
        array of powers of the initial multivariate polynomial
    coeff_arr : np.ndarray of shape (L, n_coefs)
        array of coeffients of the initial multivariate polynomial

    Returns
    -------
    pow_grad : np.ndarray of shape (D_no_kappa, n_coefs, D_no_kappa)
        array of powers of the gradient of multivariate polynomial
    coef_grad : np.ndarray of shape (D_no_kappa, L, n_coefs)
        array of coeffients of the gradient of multivariate polynomial
    """
    L, n_coefs = coeff_arr.shape
    n_coefs_2, D_no_kappa = pow_arr.shape
    assert n_coefs == n_coefs_2

    pow_grad = pow_arr[None, :, :] * np.ones((D_no_kappa, n_coefs, D_no_kappa))
    coef_grad = coeff_arr[None, :, :] * np.ones((D_no_kappa, L, n_coefs))

    for d_deriv in range(D_no_kappa):
        for i_coef in range(n_coefs):
            if pow_grad[d_deriv, i_coef, d_deriv] > 0:
                pow_grad[d_deriv, i_coef, d_deriv] -= 1
                coef_grad[d_deriv, :, i_coef] *= pow_grad[d_deriv, i_coef, d_deriv] + 1
            else:
                coef_grad[d_deriv, :, i_coef] = 0.0

    return pow_grad, coef_grad


def compute_hess_diag_poly_params(
    pow_arr: np.ndarray, coeff_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """computes the power array and the coefficient array of the diagonal of the hessian of the multivariate polynomial defined with the given power and coefficient arrays

    Parameters
    ----------
    pow_arr : np.ndarray of shape (n_coefs, D_no_kappa)
        array of powers of the initial multivariate polynomial
    coeff_arr : np.ndarray of shape (L, n_coefs)
        array of coefficients of the initial multivariate polynomial

    Returns
    -------
    pow_grad : np.ndarray of shape (D_no_kappa, n_coefs, D_no_kappa)
        array of powers of the diagonal hessian of multivariate polynomial
    coef_grad : np.ndarray of shape (D_no_kappa, L, n_coefs)
        array of coefficients of the diagonal hessian of multivariate polynomial
    """
    L, n_coefs = coeff_arr.shape
    n_coefs_2, D_no_kappa = pow_arr.shape
    assert n_coefs == n_coefs_2

    pow_hess_diag = pow_arr[None, :, :] * np.ones((D_no_kappa, n_coefs, D_no_kappa))
    coef_hess_diag = coeff_arr[None, :, :] * np.ones((D_no_kappa, L, n_coefs))

    for d_deriv in range(D_no_kappa):
        for i_coef in range(n_coefs):
            if pow_hess_diag[d_deriv, i_coef, d_deriv] >= 2:
                pow_hess_diag[d_deriv, i_coef, d_deriv] -= 2
                coef_hess_diag[d_deriv, :, i_coef] *= (
                    pow_hess_diag[d_deriv, i_coef, d_deriv] + 1
                ) * (pow_hess_diag[d_deriv, i_coef, d_deriv] + 2)
            else:
                coef_hess_diag[d_deriv, :, i_coef] = 0.0

    return pow_hess_diag, coef_hess_diag


@nb.jit(nopython=True, fastmath=True)
def evaluate_poly(
    Theta: np.ndarray, pow_arr: np.ndarray, coeff_arr: np.ndarray, deg: int
) -> np.ndarray:
    N_pix, D_no_kappa = Theta.shape
    L, n_coefs = coeff_arr.shape

    Theta_pow = np.ones((N_pix, D_no_kappa, deg + 1))
    for pow_ in range(1, deg + 1):
        Theta_pow[:, :, pow_] = Theta_pow[:, :, pow_ - 1] * Theta

    f_Theta = np.zeros((N_pix, L))
    for idx in range(n_coefs):
        prod_ = np.ones((N_pix,))
        for d in range(D_no_kappa):
            prod_ *= Theta_pow[:, d, pow_arr[idx, d]]
        f_Theta[:, :] += np.expand_dims(coeff_arr[:, idx], 0) * np.expand_dims(prod_, 1)

    return f_Theta


@nb.jit(nopython=True, fastmath=True)
def grad_poly(
    Theta: np.ndarray, pow_grad: np.ndarray, coef_grad: np.ndarray, deg: int
) -> np.ndarray:
    N_pix, D_no_kappa = Theta.shape
    _, L, n_coefs = coef_grad.shape

    Theta_pow = np.ones((N_pix, D_no_kappa, deg + 1))
    for pow_ in range(1, deg + 1):
        Theta_pow[:, :, pow_] = Theta_pow[:, :, pow_ - 1] * Theta

    grad_f_Theta = np.zeros((N_pix, D_no_kappa, L))
    for d in range(D_no_kappa):
        for idx in range(n_coefs):
            prod_ = np.ones((N_pix,))
            for i in range(D_no_kappa):
                prod_ *= Theta_pow[:, i, pow_grad[d, idx, i]]
            grad_f_Theta[:, d, :] += np.expand_dims(
                coef_grad[d, :, idx], 0
            ) * np.expand_dims(prod_, 1)

    return grad_f_Theta


@nb.jit(nopython=True, fastmath=True)
def hess_diag_poly(
    Theta: np.ndarray, pow_hess_diag: np.ndarray, coef_hess_diag: np.ndarray, deg: int
) -> np.ndarray:
    N_pix, D_no_kappa = Theta.shape
    _, L, n_coefs = coef_hess_diag.shape

    Theta_pow = np.ones((N_pix, D_no_kappa, deg + 1))
    for pow_ in range(1, deg + 1):
        Theta_pow[:, :, pow_] = Theta_pow[:, :, pow_ - 1] * Theta

    hess_diag_f_Theta = np.zeros((N_pix, D_no_kappa, L))
    for d in range(D_no_kappa):
        for idx in range(n_coefs):
            prod_ = np.ones((N_pix,))
            for i in range(D_no_kappa):
                prod_ *= Theta_pow[:, i, pow_hess_diag[d, idx, i]]
            hess_diag_f_Theta[:, d, :] += np.expand_dims(
                coef_hess_diag[d, :, idx], 0
            ) * np.expand_dims(prod_, 1)

    return hess_diag_f_Theta


class PolynomialApprox(ExpForwardMap):
    r"""multi-line polynomial regressor made to approximate the true forward model given a grid of simulations.

    Given the fact that the true Meudon PDR code varies over orders of magnitudes and in order to guarantee that the interpolations are strictly positive, the regression is done in the log space. Therefore, for each line, the approximation is of the form

    .. math::

            f_{\ell} (\theta_n) = \exp \circ \ln f_{\ell} (\theta_n), \quad \ln f_{\ell} \in \mathbb{R}[X]
    """

    LOGE_10 = np.log(10.0)
    r"""float: natural log (in base :math:`e`) of 10, computed once and saved to limit redundant computations"""

    def __init__(
        self,
        path_model: str,
        model_name: str,
        dict_fixed_values_scaled: dict[str, Optional[float]],
        angle: float,
    ):
        filepath = f"{path_model}/{model_name}/model.pickle"
        assert filepath[-7:] == ".pickle", "incorrect format"

        with open(filepath, "rb") as file_:
            (
                D_no_kappa,
                D,
                L,
                deg,
                pow_arr,
                coeff_arr,
            ) = pickle.load(file_)

        self.D_no_kappa = D_no_kappa
        r"""int: full dimension of the physical parameter vector, except for the scaling parameter :math:`\kappa` (which is not an input of the polynomial)"""

        self.D = D
        r"""int: full dimension of the physical parameter vector, including the scaling parameter :math:`\kappa`"""

        self.L = L
        r"""int: total number of observables per pixel used for inversion"""

        self.deg = deg
        r"""int: degree of the polynomial"""

        self.set_sampled_and_fixed_entries(dict_fixed_values_scaled)

        self.pow_arr = pow_arr  # (n_coefs, D_no_kappa)
        r"""np.ndarray of shape (n_coefs, D_no_kappa): powers of the monomials (loaded from model)"""
        self.coeff_arr = coeff_arr  # (L, n_coefs)
        r"""np.ndarray of shape (L, n_coefs): coefficients associated to the monomials (loaded from model)"""

        self.angle = angle
        r"""float: observation angle"""

        pow_grad, coeff_grad = compute_grad_poly_params(
            self.pow_arr,
            self.coeff_arr,
        )
        self.pow_grad = pow_grad.astype(int)  # (D_no_kappa, n_coefs, D_no_kappa)
        r"""np.ndarray of shape (D_no_kappa, n_coefs, D_no_kappa): powers of the monomials of the first derivative polynomial (computed in ``__init__`` method)"""
        self.coeff_grad = coeff_grad.astype(float)  # (D_no_kappa, L, n_coefs)
        r"""np.ndarray of shape (D_no_kappa, L, n_coefs): coefficients associated to the monomials of the first derivative polynomial (computed in ``__init__`` method)"""

        pow_hess_diag, coeff_hess_diag = compute_hess_diag_poly_params(
            self.pow_arr, self.coeff_arr
        )
        self.pow_hess_diag = pow_hess_diag.astype(int)
        r"""np.ndarray of shape (D_no_kappa, n_coefs, D_no_kappa): powers of the monomials of the diagonal terms of the second derivative polynomial (computed in ``__init__`` method)"""
        self.coeff_hess_diag = coeff_hess_diag.astype(float)
        r"""np.ndarray of shape (D_no_kappa, L, n_coefs): coefficients associated to the monomials of the first derivative polynomial (computed in ``__init__`` method)"""

        with open(
            f"{path_model}/{model_name}/line_names.pickle",
            "rb",
        ) as f:
            self.outputs_names = pickle.load(f)
            r"""list: names of the observables"""

        self.output_subset = self.outputs_names * 1
        r"""list: names of the observables used for inversion"""
        self.output_subset_indices = list(range(L))
        r"""list: indices the observables"""

        self.coeff_arr_subset = self.coeff_arr[self.output_subset_indices, :] * 1
        r"""np.ndarray: coefficients of the polynomial with restricted outputs"""
        self.coeff_grad_subset = self.coeff_grad[:, self.output_subset_indices, :] * 1
        r"""np.ndarray: coefficients of the first derivative polynomial with restricted outputs"""
        self.coeff_hess_diag_subset = (
            self.coeff_hess_diag[:, self.output_subset_indices, :] * 1
        )
        r"""np.ndarray: coefficients of the diagonal of the second derivative polynomial with restricted outputs"""

    @staticmethod
    def _check_if_subsequence(seq: Sequence, subseq: Sequence) -> bool:
        return set(subseq) <= set(seq)

    @staticmethod
    def _indices_of_subsequence(seq: Sequence, subseq: Sequence) -> List[int]:
        index_dict = dict((value, idx) for idx, value in enumerate(seq))
        return [index_dict[value] for value in subseq]  # Remark: the result is ordered

    def indices_output_subset(
        self, output_subset: Union[Sequence[str], Sequence[int]]
    ) -> List[int]:
        if len(output_subset) == 0:
            raise ValueError("output_subset must not be empty")
        if isinstance(output_subset[0], str):
            if self.outputs_names is None:
                raise TypeError(
                    "output_subset cannot be a sequence of str when self.outputs_names is None"
                )
            if not self._check_if_subsequence(self.outputs_names, output_subset):
                raise ValueError("output_subset is not a valid subset")
            indices = indices = self._indices_of_subsequence(
                self.outputs_names, output_subset
            )
        elif isinstance(output_subset[0], int):
            if not self._check_if_subsequence(
                list(range(self.out_features)), output_subset
            ):
                raise ValueError("input_subset is not a valid subset")
            indices = output_subset
        else:
            raise TypeError(
                f"output_subset must contain str or int, not {type(output_subset[0])}"
            )

        return indices

    def restrict_to_output_subset(self, output_subset: Sequence[str]) -> None:
        for line in output_subset:
            assert line in self.outputs_names, line

        self.output_subset = output_subset * 1
        self.output_subset_indices = self.indices_output_subset(output_subset)
        self.L = len(output_subset)

        self.coeff_arr_subset = self.coeff_arr[self.output_subset_indices, :] * 1
        self.coeff_grad_subset = self.coeff_grad[:, self.output_subset_indices, :] * 1
        self.coeff_hess_diag_subset = (
            self.coeff_hess_diag[:, self.output_subset_indices, :] * 1
        )

    def evaluate(self, Theta: np.ndarray) -> np.ndarray:
        Theta_with_angle = np.column_stack(
            (Theta, self.angle * np.ones((Theta.shape[0], 1))),
        )
        val = evaluate_poly(
            Theta_with_angle[:, 1:],
            self.pow_arr,
            self.coeff_arr_subset,
            self.deg,
        )
        val += Theta[:, 0][:, None]  # add log_kappa
        return np.exp(val)  # (N, L)

    def evaluate_log(self, Theta: np.ndarray) -> np.ndarray:
        Theta_with_angle = np.column_stack(
            (Theta, self.angle * np.ones((Theta.shape[0], 1))),
        )
        val = evaluate_poly(
            Theta_with_angle[:, 1:],
            self.pow_arr,
            self.coeff_arr_subset,
            self.deg,
        )
        val += Theta[:, 0][:, None]  # add log_kappa
        return val  # (N, L)

    def gradient(self, Theta: np.ndarray) -> np.ndarray:
        Theta_with_angle = np.column_stack(
            (Theta, self.angle * np.ones((Theta.shape[0], 1))),
        )
        theta = Theta_with_angle[:, 1:]
        grad_P = self._gradient_log(theta)  # (N, D, L)
        Intensities = self.evaluate(Theta)  # (N, L)
        return grad_P * Intensities[:, None, :]  # (N, D, L)

    def gradient_log(self, Theta: np.ndarray) -> np.ndarray:
        Theta_with_angle = np.column_stack(
            (Theta, self.angle * np.ones((Theta.shape[0], 1))),
        )
        theta = Theta_with_angle[:, 1:]
        return self._gradient_log(theta)  # (N, D, L)

    def _gradient_log(self, theta):
        grad_ = grad_poly(
            theta,
            self.pow_grad,
            self.coeff_grad_subset,
            self.deg,
        )
        grad_ = grad_[:, :-1, :]  # (N, D_no_kappa, L) (remove angle)

        # gradient that includes kappa (the grad of p wrt kappa is 1)
        grad_full = np.ones((grad_.shape[0], self.D, self.L))
        grad_full[:, 1:, :] = grad_
        return grad_full  # (N, D, L)

    def _hess_diag_log(self, theta):
        hess_diag = hess_diag_poly(
            theta, self.pow_hess_diag, self.coeff_hess_diag_subset, self.deg
        )
        hess_diag = hess_diag[:, :-1, :]  # (N, D_no_kappa, L) (remove angle)

        hess_diag_full = np.zeros((theta.shape[0], self.D, self.L))
        hess_diag_full[:, 1:, :] = hess_diag  # (N, D, L)

        return hess_diag_full  # (N, D, L)

    def hess_diag(self, Theta: np.ndarray) -> np.ndarray:
        Theta_with_angle = np.column_stack(
            (Theta, self.angle * np.ones((Theta.shape[0], 1))),
        )
        theta = Theta_with_angle[:, 1:]

        diag_hess_log = self._hess_diag_log(theta)
        grad_log = self._gradient_log(theta)
        Intensities = self.evaluate(Theta)

        diag_hess = (diag_hess_log + grad_log**2) * Intensities[:, None, :]
        return diag_hess  # (N, D, L)

    def hess_diag_log(self, Theta: np.ndarray) -> np.ndarray:
        Theta_with_angle = np.column_stack(
            (Theta, self.angle * np.ones((Theta.shape[0], 1))),
        )
        theta = Theta_with_angle[:, 1:]
        return self._hess_diag_log(theta)  # (N, D, L)

    def compute_all(
        self,
        Theta: np.ndarray,
        compute_lin: bool = True,
        compute_log: bool = True,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        r"""gathers the evaluation of the forward map in linear and log scale and of the associated derivatives. Permits to limit repeating computations, but requires the storage in memory of the result.

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            array of points in the input space :math:`\Theta = (\theta_n)_{n=1}^N` with :math:`\theta_n \in \mathbb{R}^D`
        compute_lin : bool, optional
            wether or not to compute the forward model (and possibly the gradient and diagonal of the Hessian), by default True
        compute_log : bool, optional
            wether or not to compute the log-forward model (and possibly the gradient and diagonal of the Hessian), by default True
        compute_derivatives : bool, optional
            wether or not to evaluate the derivatives of the forward map, by default True
        compute_derivatives_2nd_order : bool, optional
            wether or not to evaluate the 2nd order derivatives of the forward map, by default True

        Returns
        -------
        forward_map_evals : dict[str, np.ndarray]
            dictionary with entries such as `f_Theta`, `log_f_Theta`, `grad_f_Theta`, `grad_log_f_Theta`, `hess_diag_f_Theta` and `hess_diag_log_f_Theta`, depending on the input booleans.

        Note
        ----
        To evaluating :math:`f(\theta_n)` and the associated derivatives, 3 evaluations are enough for six functions. Calling each function would result in a total of 9 evaluations."""
        forward_map_evals = dict()

        N_pix = Theta.shape[0]

        if compute_derivatives:
            log_f_Theta = self.evaluate_log(Theta)
            grad_log_f_Theta = self.gradient_log(Theta)

            log_f_Theta *= self.LOGE_10
            grad_log_f_Theta = grad_log_f_Theta[:, : self.D_no_kappa, :] * self.LOGE_10

            if compute_derivatives_2nd_order:
                hess_diag_log_f_Theta = self.hess_diag(Theta)
                hess_diag_log_f_Theta = (
                    hess_diag_log_f_Theta[:, : self.D_no_kappa, :] * self.LOGE_10
                )

            log_f_Theta = Theta[:, 0][:, None] + log_f_Theta

            if compute_log:
                forward_map_evals["log_f_Theta"] = log_f_Theta

                grad_log_f_Theta_full = np.ones((N_pix, self.D, self.L))
                grad_log_f_Theta_full[:, 1:, :] = grad_log_f_Theta * 1
                forward_map_evals["grad_log_f_Theta"] = grad_log_f_Theta_full

                if compute_derivatives_2nd_order:
                    hess_diag_log_f_Theta_full = np.zeros((N_pix, self.D, self.L))
                    hess_diag_log_f_Theta_full[:, 1:, :] = hess_diag_log_f_Theta * 1
                    forward_map_evals[
                        "hess_diag_log_f_Theta"
                    ] = hess_diag_log_f_Theta_full

            if compute_lin:
                f_Theta = np.exp(log_f_Theta)
                forward_map_evals["f_Theta"] = f_Theta

                # (N_pix, D, L)
                forward_map_evals["grad_f_Theta"] = (
                    grad_log_f_Theta_full * f_Theta[:, None, :]
                )

                if compute_derivatives_2nd_order:
                    # (N_pix, D, L)
                    forward_map_evals["hess_diag_f_Theta"] = f_Theta[:, None, :] * (
                        hess_diag_log_f_Theta_full + grad_log_f_Theta_full**2
                    )

            return forward_map_evals

        else:
            log_f_Theta = self.evaluate_log(Theta)
            if compute_log:
                forward_map_evals["log_f_Theta"] = log_f_Theta
            if compute_lin:
                forward_map_evals["f_Theta"] = np.exp(log_f_Theta)
            return forward_map_evals
