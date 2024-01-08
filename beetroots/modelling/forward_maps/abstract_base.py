"""Abstract forward map
"""
from abc import ABC, abstractmethod

import numpy as np


class ForwardMap(ABC):
    r"""Forward model such that for every pixel :math:`n \in [1, N]`

    .. math::

        f : \theta_n \in \mathbb{R}^D \mapsto f(\theta_n) \in \mathbb{R}^L
    """

    __slots__ = ("D", "L", "N")

    def __init__(self, D: int, L: int, N: int) -> None:
        self.D = D
        """int: dimensionality of input space"""
        self.L = L
        """int: dimensionality of output space"""
        self.N = N
        """int: number of independent pixels"""

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

        Returns
        -------
        forward_map_evals : dict[str, np.ndarray]
            dictionary with entries such as `f_Theta`, `log_f_Theta`, `grad_f_Theta`, `grad_log_f_Theta`, `hess_diag_f_Theta` and `hess_diag_log_f_Theta`, depending on the input booleans.
        """
        pass
