"""Implementation of the identity forward map
"""
from typing import List, Optional

import numpy as np

from beetroots.modelling.forward_maps.abstract_base import ForwardMap


class BasicForwardMap(ForwardMap):
    r"""Forward model such that for every pixel :math:`n \in [1, N]`

    .. math::
        f :  \theta_n \in \mathbb{R}^D \mapsto \theta_n \in \mathbb{R}^D

    i.e. in this class, :math:`D = L`
    """

    def __init__(
        self, L: int, N: int, dict_fixed_values_scaled: dict[str, Optional[float]] = {}
    ):
        super().__init__(L, L, N, dict_fixed_values_scaled)

        self.output_subset = np.arange(self.D)
        r"""List[int]: subset of outputs to be predicted. Can be updated with ``restrict_to_output_subset``"""

    def evaluate(self, Theta: np.ndarray) -> np.ndarray:
        return Theta[:, self.output_subset]  # (N, L)

    def gradient(self, Theta: np.ndarray) -> np.ndarray:
        return np.ones((Theta.shape[0], self.L, self.L))  # (N, D, L)

    def hess_diag(self, Theta: np.ndarray) -> np.ndarray:
        return np.zeros((Theta.shape[0], self.L, self.L))

    def compute_all(
        self,
        Theta: np.ndarray,
        compute_lin: bool = True,
        compute_log: bool = False,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        r"""gathers the evaluation of the forward map in linear and log scale and of the associated derivatives. Permits to limit repeating computations, but requires the storage in memory of the result.

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            array of points in the input space :math:`\Theta = (\theta_n)_{n=1}^N` with :math:`\theta_n \in \mathbb{R}^D`
        compute_lin : bool, optional
            always considered as True. Kept for this class for consistency.
        compute_log : bool, optional
            always considered as False. Kept for this class for consistency.
        compute_derivatives : bool, optional
            wether or not to evaluate the derivatives of the forward map, by default True
        compute_derivatives_2nd_order : bool, optional
            wether or not to evaluate the 2nd order derivatives of the forward map, by default True

        Returns
        -------
        forward_map_evals : dict[str, np.ndarray]
            dictionary with the `f_Theta` entry and possibly `grad_f_Theta`, and `hess_diag_f_Theta`, depending on the input booleans.
        """
        forward_map_evals = dict()
        forward_map_evals["f_Theta"] = self.evaluate(Theta)[:, self.output_subset]

        if compute_derivatives:
            forward_map_evals["grad_f_Theta"] = self.gradient(Theta)

            if compute_derivatives_2nd_order:
                forward_map_evals["hess_diag_f_Theta"] = self.hess_diag(Theta)

        return forward_map_evals

    def restrict_to_output_subset(self, list_observables: List[int]) -> None:
        for idx in list_observables:
            assert 0 <= idx <= self.D

        self.output_subset = list_observables
        self.L = len(list_observables)
