import numpy as np

from beetroots.modelling.forward_maps.abstract_exp import ExpForwardMap


class BasicExpForwardMap(ExpForwardMap):
    r"""Forward model such that for every pixel :math:`n \in [1, N]`

    .. math::
        f :  \theta_n \in \mathbb{R}^D \mapsto \exp(\theta_n) \in \mathbb{R}^D

    i.e. in this class, :math:`D = L`
    """

    def __init__(self, D, L, N):
        assert D == L
        super().__init__(D, L, N)

    def evaluate(self, Theta: np.ndarray) -> np.ndarray:
        return np.exp(Theta)

    def evaluate_log(self, Theta: np.ndarray) -> np.ndarray:
        return Theta

    def gradient(self, Theta: np.ndarray) -> np.ndarray:
        return np.exp(Theta)[:, None, :] * np.ones((self.N, self.D, self.L))

    def gradient_log(self, Theta: np.ndarray) -> np.ndarray:
        return np.ones((self.N, self.D, self.L))

    def hess_diag(self, Theta: np.ndarray) -> np.ndarray:
        return np.exp(Theta)[:, None, :] * np.ones((self.N, self.D, self.L))

    def hess_diag_log(self, Theta: np.ndarray) -> np.ndarray:
        return np.zeros((self.N, self.D, self.L))

    def compute_all(
        self,
        Theta: np.ndarray,
        compute_lin: bool = True,
        compute_log: bool = True,
        compute_derivatives: bool = True,
    ) -> dict[str, np.ndarray]:
        forward_map_evals = dict()

        f_Theta = self.evaluate(Theta)

        #! not necessarily N (in candidates testing case for MTM)
        N_pix = f_Theta.shape[0]

        if compute_lin:
            forward_map_evals["f_Theta"] = f_Theta

        if compute_lin and compute_derivatives:
            grad_f_Theta = np.ones((N_pix, self.D, self.L)) * f_Theta[:, None, :]
            hess_diag_f_Theta = np.ones((N_pix, self.D, self.L)) * f_Theta[:, None, :]

            forward_map_evals["grad_f_Theta"] = grad_f_Theta
            forward_map_evals["hess_diag_f_Theta"] = hess_diag_f_Theta

        if compute_log:
            log_f_Theta = np.log(f_Theta)
            forward_map_evals["log_f_Theta"] = log_f_Theta

        if compute_log and compute_derivatives:
            grad_log_f_Theta = np.ones((N_pix, self.D, self.L))
            hess_diag_log_f_Theta = np.zeros((N_pix, self.D, self.L))

            forward_map_evals["grad_log_f_Theta"] = grad_log_f_Theta
            forward_map_evals["hess_diag_log_f_Theta"] = hess_diag_log_f_Theta

        return forward_map_evals
