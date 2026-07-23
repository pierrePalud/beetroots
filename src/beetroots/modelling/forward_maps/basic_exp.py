from typing import List, Optional

import numpy as np

from beetroots.modelling.forward_maps.abstract_exp import ExpForwardMap


class BasicExpForwardMap(ExpForwardMap):
    r"""Forward model such that for every pixel :math:`n \in [1, N]`

    .. math::
        f :  \theta_n \in \mathbb{R}^D \mapsto \exp(\theta_n) \in \mathbb{R}^D

    i.e. in this class, :math:`D = L`
    """

    def __init__(
        self, D, L, N, dict_fixed_values_scaled: dict[str, Optional[float]] = {}
    ):
        assert D == L
        super().__init__(D, L, N, dict_fixed_values_scaled)

        self.output_subset = np.arange(self.D)
        r"""List[int]: subset of outputs to be predicted. Can be updated with ``restrict_to_output_subset``"""

    def evaluate(self, Var: np.ndarray) -> np.ndarray:
        assert Var.shape[1] == self.D
        return np.exp(Var)[:, self.output_subset]

    def evaluate_log(self, Var: np.ndarray) -> np.ndarray:
        assert Var.shape[1] == self.D
        return Var[:, self.output_subset]

    def gradient(self, Var: np.ndarray) -> np.ndarray:
        return np.exp(Var)[:, None, self.output_subset] * np.ones(
            (self.N, self.D, self.L)
        )

    def gradient_log(self, Var: np.ndarray) -> np.ndarray:
        return np.ones((self.N, self.D, self.L))

    def hess_diag(self, Var: np.ndarray) -> np.ndarray:
        return np.exp(Var)[:, None, self.output_subset] * np.ones(
            (self.N, self.D, self.L)
        )

    def hess_diag_log(self, Var: np.ndarray) -> np.ndarray:
        return np.zeros((self.N, self.D, self.L))

    def compute_all(
        self,
        Var: np.ndarray,
        compute_lin: bool = True,
        compute_log: bool = True,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict[str, np.ndarray]:
        forward_map_evals = dict()

        f_Var = self.evaluate(Var)[:, self.output_subset]

        #! not necessarily N (in candidates testing case for MTM)
        N_pix = f_Var.shape[0]

        if compute_lin:
            forward_map_evals["f_Var"] = f_Var

        if compute_lin and compute_derivatives:
            grad_f_Var = np.ones((N_pix, self.D, self.L)) * f_Var[:, None, :]
            forward_map_evals["grad_f_Var"] = grad_f_Var

            if compute_derivatives_2nd_order:
                hess_diag_f_Var = np.ones((N_pix, self.D, self.L)) * f_Var[:, None, :]
                forward_map_evals["hess_diag_f_Var"] = hess_diag_f_Var

        if compute_log:
            log_f_Var = np.log(f_Var)
            forward_map_evals["log_f_Var"] = log_f_Var

        if compute_log and compute_derivatives:
            grad_log_f_Var = np.ones((N_pix, self.D, self.L))
            forward_map_evals["grad_log_f_Var"] = grad_log_f_Var

            if compute_derivatives_2nd_order:
                hess_diag_log_f_Var = np.zeros((N_pix, self.D, self.L))
                forward_map_evals["hess_diag_log_f_Var"] = hess_diag_log_f_Var

        return forward_map_evals

    def restrict_to_output_subset(self, list_observables: List[int]) -> None:
        for idx in list_observables:
            assert 0 <= idx <= self.D

        self.output_subset = list_observables
        self.L = len(list_observables)
