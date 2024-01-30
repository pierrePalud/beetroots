import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from nnbma.networks import NeuralNetwork
from torch.func import hessian, jacfwd, vmap

from beetroots.modelling.forward_maps.abstract_exp import ExpForwardMap


class NeuralNetworkApprox(ExpForwardMap):
    r"""Forward model based on a neural network. For every entry :math:`n \in [1, N]`

    .. math::

        f : \theta_n \in \mathbb{R}^D \mapsto f(\theta_n) \in \mathbb{R}^L

    where :math:`f(\theta_n)` can be written with a left composition with the exponential, i.e., :math:`f(\theta_n) = \exp \circ (\ln f) (\theta_n)`. Here the neural network corresponds to :math:`\ln f(\theta_n)`.

    The neural network needs to be defined with the ``nnbma`` package, presented in :cite:t:`paludNeuralNetworkbasedEmulation2023`.
    Here are the links towards the corresponding
    `GitHub repository <https://github.com/einigl/ism-model-nn-approximation>`_,
    `PyPi package <https://pypi.org/project/nnbma/>`_
    and `documentation <https://ism-model-nn-approximation.readthedocs.io/en/latest/?badge=latest>`_.
    """

    LOGE_10 = np.log(10.0)
    r"""float: natural log (in base :math:`e`) of 10, computed once and saved to limit redundant computations"""

    def __init__(
        self,
        path_model: str,
        model_name: str,
        dict_fixed_values_scaled: Dict[str, Optional[float]],
        device: Optional[str] = None,
    ):
        self.network = NeuralNetwork.load(model_name, path_model)
        r"""NeuralNetwork: instance of neural network from the ``nnbma`` package (see https://pypi.org/project/nnbma/)"""

        assert device in [None, "cpu"]
        if device is None:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "cpu" if device is None else device  # force cpu

        # set network to a specific device (either cpu or cuda)
        self.device = device
        r"""str: device on which the neural network is to be run, either cuda or cpu"""

        self.network.set_device(device)
        self.network.double().eval()

        self.D_no_kappa = self.network.input_features * 1
        r"""int: full dimension of the physical parameter vector, except for the scaling parameter :math:`\kappa` (which is not an input of the neural network)"""

        self.D = self.D_no_kappa + 1  # include kappa
        r"""int: full dimension of the physical parameter vector, including the scaling parameter :math:`\kappa`"""

        self.L = self.network.output_features * 1
        r"""int: total number of observables per pixel used for inversion"""

        self._update_derivatives()

        self.set_sampled_and_fixed_entries(dict_fixed_values_scaled)

        self.list_indices_to_sample_for_nn = [
            d - 1 for d in self.list_indices_to_sample if d >= 1
        ]  # remove kappa and start at Pth
        r"""list: indices of the entries to be sampled for the neural network, i.e., with an offset of 1 compared with `list_indices_fixed` due to the scaling parameter :math:`\kappa`"""

        self.list_indices_fixed_for_nn = [
            d - 1 for d in self.list_indices_fixed if d >= 1
        ]  # remove kappa and start at Pth
        r"""list: indices of the entries to be fixed for the neural network, i.e., with an offset of 1 compared with `list_indices_fixed` due to the scaling parameter :math:`\kappa`"""

        # display short message
        msg = f"neural network runs on : {self.network.device}"
        msg += f" (asked: {device})"
        print(msg)
        return

    def _update_derivatives(self):
        self.jacobian_network = vmap(jacfwd(self.network.forward))
        r"""function that yields the first derivative of the network"""
        self.hessian_network = vmap(jacfwd(jacfwd(self.network.forward)))
        r"""function that yields the diagonal of the second derivative of the network"""
        return

    def restrict_to_output_subset(self, output_subset: List[str]) -> None:
        """Restricts the full output set computed by the neural network to a potentially small subset of outputs. Permits to accelerate computations.

        Parameters
        ----------
        output_subset : List[str]
            list of the names of the outputs to be considered in forward map evaluations and derivatives.
        """
        for line in output_subset:
            assert line in self.network.current_output_subset, line

        self.network.restrict_to_output_subset(output_subset=output_subset)
        self.L = len(output_subset)

        self._update_derivatives()
        return

    def evaluate(self, Theta: np.ndarray) -> np.ndarray:
        return np.exp(self.evaluate_log(Theta))  # (N, L)

    def evaluate_log(self, Theta: np.ndarray) -> np.ndarray:
        Theta_combined = np.zeros((Theta.shape[0], self.D))
        Theta_combined += self.arr_fixed_values[None, :]
        for i, idx in enumerate(self.list_indices_to_sample):
            Theta_combined[:, idx] = Theta[:, i] * 1

        # the neural network returns log10 of intensities
        val = self.network.evaluate(Theta_combined[:, 1:])
        val *= self.LOGE_10  # go back to natural log scale

        val += Theta_combined[:, 0][:, None]  # add ln kappa

        return val  # (N, L)

    def gradient(self, Theta):
        msg = "Gradients should be computed with compute_all"
        raise NotImplementedError(msg)

    def gradient_log(self, Theta):
        msg = "Gradients should be computed with compute_all"
        raise NotImplementedError(msg)

    def _gradient_log(self, Theta):
        msg = "Gradients should be computed with compute_all"
        raise NotImplementedError(msg)

    def _hess_diag_log(self, Theta):
        msg = "Gradients should be computed with compute_all"
        raise NotImplementedError(msg)

    def hess_diag(self, Theta):
        msg = "Gradients should be computed with compute_all"
        raise NotImplementedError(msg)

    def hess_diag_log(self, Theta):
        msg = "Gradients should be computed with compute_all"
        raise NotImplementedError(msg)

    def _hess_full_log(self, Theta):
        msg = "Gradients should be computed with compute_all"
        raise NotImplementedError(msg)

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

        # combine fixed and sampled values for model evaluation
        Theta_combined = np.zeros((N_pix, self.D))
        Theta_combined += self.arr_fixed_values[None, :]
        for i, idx in enumerate(self.list_indices_to_sample):
            Theta_combined[:, idx] += Theta[:, i]

        if compute_derivatives:
            _Theta = torch.from_numpy(Theta_combined[:, 1:])  # .float()

            #! integrated intensities and deriviatives in log10 scale
            log_f_Theta = self.network.forward(_Theta).detach().numpy()  # (N, L)

            assert log_f_Theta.shape == (
                N_pix,
                self.L,
            ), f"{log_f_Theta.shape} is not ({N_pix}, {self.L})"
            assert np.max(np.abs(log_f_Theta)) > 0

            grad_log_f_Theta = (
                self.jacobian_network(_Theta)
                .detach()
                .numpy()  # .to(self.network.device)
            ).transpose(
                (0, 2, 1)
            )  # (N, D, L)

            assert grad_log_f_Theta.shape == (
                N_pix,
                self.D_sampling,
                self.L,
            ), f"{grad_log_f_Theta.shape} is not ({N_pix}, {self.D_sampling}, {self.L})"
            assert np.max(np.abs(grad_log_f_Theta)) > 0

            log_f_Theta *= self.LOGE_10
            grad_log_f_Theta = grad_log_f_Theta[:, :, :] * self.LOGE_10

            if compute_derivatives_2nd_order:
                hess_full_log_f_Theta = (
                    self.hessian_network(_Theta)
                    .detach()
                    .numpy()  # .to(self.network.device)
                )  # (N, L, D, D)
                hess_diag_log_f_Theta = hess_full_log_f_Theta.diagonal(
                    offset=0, axis1=2, axis2=3
                ).transpose(
                    (0, 2, 1)
                )  # (N, D, L)

                assert hess_diag_log_f_Theta.shape == (
                    N_pix,
                    self.D_sampling,
                    self.L,
                ), f"{hess_diag_log_f_Theta.shape} is not ({N_pix}, {self.D_sampling}, {self.L})"
                assert np.max(np.abs(hess_diag_log_f_Theta)) > 0

                hess_diag_log_f_Theta = hess_diag_log_f_Theta[:, :, :] * self.LOGE_10

            # add log kappa
            log_f_Theta = Theta_combined[:, 0][:, None] + log_f_Theta

            if compute_log:
                forward_map_evals["log_f_Theta"] = log_f_Theta

                grad_log_f_Theta_full = np.ones((N_pix, self.D_sampling, self.L))
                if 0 in self.list_indices_to_sample:
                    grad_log_f_Theta_full[:, 1:, :] = (
                        grad_log_f_Theta[:, self.list_indices_to_sample[1:], :] * 1
                    )
                else:
                    grad_log_f_Theta_full[:, :, :] = (
                        grad_log_f_Theta[:, self.list_indices_to_sample, :] * 1
                    )
                forward_map_evals["grad_log_f_Theta"] = grad_log_f_Theta_full

                if compute_derivatives_2nd_order:
                    hess_diag_log_f_Theta_full = np.zeros(
                        (N_pix, self.D_sampling, self.L)
                    )
                    if 0 in self.list_indices_to_sample:
                        hess_diag_log_f_Theta_full[:, 1:, :] = (
                            hess_diag_log_f_Theta[:, self.list_indices_to_sample[1:], :]
                            * 1
                        )
                    else:
                        hess_diag_log_f_Theta_full[:, :, :] = (
                            hess_diag_log_f_Theta[:, self.list_indices_to_sample, :] * 1
                        )
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

                # (N_pix, D, L)
                if compute_derivatives_2nd_order:
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
