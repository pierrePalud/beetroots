from abc import abstractmethod

import numpy as np

from beetroots.modelling.forward_maps.abstract_base import ForwardMap


class ExpForwardMap(ForwardMap):
    r"""Forward model such that for every pixel :math:`n \in [1, N]`

    .. math::

        f : \theta_n \in \mathbb{R}^D \mapsto f(\theta_n) \in \mathbb{R}^L

    where :math:`f(\theta_n)` can be written with a left composition with the exponential, i.e., :math:`f(\theta_n) = \exp \circ (\ln f) (\theta_n)`
    """

    @abstractmethod
    def evaluate_log(self, Theta: np.ndarray) -> np.ndarray:
        r"""evaluates the log-forward map for a set of input vectors :math:`(\theta_n))_{n=1}^N`

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            array of points in the input space :math:`\Theta = (\theta_n)_{n=1}^N` with :math:`\theta_n \in \mathbb{R}^D`

        Returns
        -------
        np.ndarray of shape (N, L)
            array of the log-forward map images in the observation space :math:`(\ln f(\theta_n))_{n=1}^N` with :math:`\ln f(\theta_n) \in \mathbb{R}^L`
        """
        pass

    @abstractmethod
    def gradient_log(self, Theta: np.ndarray) -> np.ndarray:
        r"""returns the gradient of the log-forward map for a set of input vectors :math:`(\theta_n))_{n=1}^N`

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            array of points in the input space :math:`\Theta = (\theta_n)_{n=1}^N` with :math:`\theta_n \in \mathbb{R}^D`

        Returns
        -------
        np.ndarray of shape (N, D, L)
            array of gradients :math:`(\nabla \ln f(\theta_n))_{n=1}^N`, with :math:`\nabla \ln  f(\theta_n) \in \mathbb{R}^{D \times L}`
        """
        pass

    @abstractmethod
    def hess_diag_log(self, Theta: np.ndarray) -> np.ndarray:
        r"""returns the diagonal of the hessian of the log-forward map for a set of input vectors :math:`(\theta_n))_{n=1}^N`

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            array of points in the input space :math:`\Theta = (\theta_n)_{n=1}^N` with :math:`\theta_n \in \mathbb{R}^D`

        Returns
        -------
        np.ndarray of shape (N, D, L)
            array of diagonal of the hessians :math:`(\text{diag} \nabla^2 \ln f(\theta_n))_{n=1}^N`, with :math:`\text{diag} \nabla^2 \ln f(\theta_n) \in \mathbb{R}^{D \times L}`
        """
        pass
