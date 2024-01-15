"""Abstract forward map
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np


class ForwardMap(ABC):
    r"""Forward model such that for every pixel :math:`n \in [1, N]`

    .. math::

        f : \theta_n \in \mathbb{R}^D \mapsto f(\theta_n) \in \mathbb{R}^L
    """

    __slots__ = (
        "D",
        "L",
        "N",
        "dict_fixed_values_scaled",
        "list_indices_to_sample",
        "list_indices_fixed",
        "D_sampling",
        "arr_fixed_values",
    )

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        dict_fixed_values_scaled: dict[str, Optional[float]] = {},
    ) -> None:
        self.D = D
        """int: dimensionality of input space"""
        self.L = L
        """int: dimensionality of output space"""
        self.N = N
        """int: number of independent pixels"""

        self.set_sampled_and_fixed_entries(dict_fixed_values_scaled)

    def set_sampled_and_fixed_entries(
        self, dict_fixed_values_scaled: dict[str, Optional[float]] = {}
    ) -> None:
        # manually fixed values, constant during sampling / optim
        self.dict_fixed_values_scaled = dict_fixed_values_scaled
        r"""dict: indices of the entries to be fixed and the associated value"""

        self.list_indices_to_sample = [
            d
            for d, value in enumerate(dict_fixed_values_scaled.values())
            if value is None
        ]
        r"""list: indices of the entries to be sampled"""

        self.list_indices_fixed = [
            d
            for d, value in enumerate(dict_fixed_values_scaled.values())
            if value is not None
        ]
        r"""list: indices of the entries to be fixed"""

        self.D_sampling = len(self.list_indices_to_sample)
        r"""int: dimension of the subspace to sample (considering fixed values)"""

        arr_fixed_values = np.zeros((self.D,))
        for d, value in enumerate(dict_fixed_values_scaled.values()):
            if value is not None:
                arr_fixed_values[d] = value * 1
        self.arr_fixed_values = arr_fixed_values
        r"""np.ndarray: values of :math:`\theta` that are not sampled, but set to a specific value. The indices of fixed entries are given in ``list_indices_fixed``"""

    @abstractmethod
    def evaluate(self, Theta: np.ndarray) -> np.ndarray:
        r"""evaluates the forward map for a set of input vectors :math:`(\theta_n))_{n=1}^N`

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            array of points in the input space :math:`\Theta = (\theta_n)_{n=1}^N` with :math:`\theta_n \in \mathbb{R}^D`

        Returns
        -------
        np.ndarray of shape (N, L)
            array of the forward map images in the observation space :math:`(f(\theta_n))_{n=1}^N` with :math:`f(\theta_n) \in \mathbb{R}^L`
        """
        pass

    @abstractmethod
    def gradient(self, Theta: np.ndarray) -> np.ndarray:
        r"""returns the gradient of the forward map for a set of input vectors :math:`(\theta_n))_{n=1}^N`

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            array of points in the input space :math:`\Theta = (\theta_n)_{n=1}^N` with :math:`\theta_n \in \mathbb{R}^D`

        Returns
        -------
        np.ndarray of shape (N, D, L)
            array of gradients :math:`(\nabla f(\theta_n))_{n=1}^N`, with :math:`\nabla f(\theta_n) \in \mathbb{R}^{D \times L}`
        """
        pass

    @abstractmethod
    def hess_diag(self, Theta: np.ndarray) -> np.ndarray:
        r"""returns the diagonal of the hessian of the forward map for a set of input vectors :math:`(\theta_n))_{n=1}^N`

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            array of points in the input space :math:`\Theta = (\theta_n)_{n=1}^N` with :math:`\theta_n \in \mathbb{R}^D`

        Returns
        -------
        np.ndarray of shape (N, D, L)
            array of diagonal of the hessians :math:`(\text{diag} \nabla^2 f(\theta_n))_{n=1}^N`, with :math:`\nabla f(\theta_n) \in \mathbb{R}^{D \times L}`
        """
        pass

    @abstractmethod
    def compute_all(
        self,
        Theta: np.ndarray,
        compute_lin: bool = True,
        compute_log: bool = True,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict[str, np.ndarray]:
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
        """
        pass

    @abstractmethod
    def restrict_to_output_subset(self, list_observables: List[Union[int, str]]):
        r"""restricts the list of outputs to be predicted to a subset, either identified by their indices or by names

        Parameters
        ----------
        list_observables : List[Union[int, str]]
            subset of outputs to be predicted
        """
        pass
