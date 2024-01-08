"""Forward model to the sensor localization problem: computes all pairwise distances
"""
import numba as nb
import numpy as np

from beetroots.modelling.forward_maps.abstract_base import ForwardMap


# @nb.jit(nopython=True)
@nb.njit()
def evaluate_distances(Theta: np.ndarray, Theta_ref: np.ndarray) -> np.ndarray:
    """evaluates the distances between all sensors

    Parameters
    ----------
    Theta : np.ndarray (N, D)
        positions of sensors (with unknown positions)
    Theta_ref : np.ndarray (N, K)
        positions of sensors (with known positions)

    Returns
    -------
    np.ndarray
        distances (N, L)
    """
    N = Theta.shape[0]

    Theta_full = np.vstack((Theta_ref, Theta))
    L = Theta_full.shape[0]

    distances = np.zeros((N, L))
    for i in range(N):
        for j in range(L):
            distances[i, j] = np.linalg.norm(Theta[i, :] - Theta_full[j, :])

    return distances


@nb.jit(nopython=True)
def evaluate_distances_candidates(
    Theta_candidates: np.ndarray, Theta_t: np.ndarray, n: int, Theta_ref: np.ndarray
) -> np.ndarray:
    """evaluates the distances between all sensors

    Parameters
    ----------
    Theta_candidates : np.ndarray (N_candidates, D)
        candidate positions for sensor :math:`n` (with unknown positions)
    Theta_t : np.ndarray (N, D)
        current position of sensors (with unknown positions)
    n : int
        index of the sensor for which the candidates are
    Theta_ref : np.ndarray (N, K)
        positions of sensors (with known positions)

    Returns
    -------
    np.ndarray
        distances (N_candidates, L)
    """
    N_candidates = Theta_candidates.shape[0]
    K = Theta_ref.shape[0]

    Theta_full = np.vstack((Theta_ref, Theta_t))
    L = Theta_full.shape[0]

    distances = np.zeros((N_candidates, L))
    for i in range(N_candidates):
        for j in range(L):
            if j != n + K:
                distances[i, j] = np.linalg.norm(
                    Theta_candidates[i, :] - Theta_full[j, :]
                )

    return distances


# @nb.jit(nopython=True)
def compute_gradient_distance(Theta: np.ndarray, Theta_ref: np.ndarray):
    r"""computes the gradient of distances between all sensors

    Parameters
    ----------
    Theta : np.ndarray (N, D)
        positions of sensors (with unknown positions)
    Theta_ref : np.ndarray (N, K)
        positions of sensors (with known positions)

    Returns
    -------
    np.ndarray
        distances (N, D, L)
    """
    N, D = Theta.shape
    K = Theta_ref.shape[0]

    Theta_full = np.vstack((Theta_ref, Theta))
    assert Theta_full.shape == (N + K, D)

    L = Theta_full.shape[0]

    distances = evaluate_distances(Theta, Theta_ref)

    grad_distances = np.zeros((N, D, L))
    for i in range(N):
        for j in range(L):
            dist_ = distances[i, j] * 1
            if dist_ > 0:
                grad_distances[i, :, j] = (
                    Theta[i, :] - Theta_full[j, :]
                ) / dist_  # (D,)

    return grad_distances


@nb.jit(nopython=True)
def compute_hess_diag_distance(Theta: np.ndarray, Theta_ref: np.ndarray):
    r"""computes the diagonal of the Hessian of distances between all sensors

    Parameters
    ----------
    Theta : np.ndarray (N, D)
        positions of sensors (with unknown positions)
    Theta_ref : np.ndarray (N, K)
        positions of sensors (with known positions)

    Returns
    -------
    np.ndarray
        distances (N, D, L)
    """
    N, D = Theta.shape
    K = Theta_ref.shape[0]

    Theta_full = np.vstack((Theta, Theta_ref))
    assert Theta_full.shape == (N + K, D)
    L = Theta_full.shape[0]

    distances = evaluate_distances(Theta, Theta_ref)

    hess_diag_distances = np.zeros((N, D, L))
    for i in range(N):
        for j in range(L):
            dist_ = distances[i, j] * 1
            if dist_ > 0:
                hess_diag_distances[i, :, j] = (
                    dist_**2 - (Theta[i, :] - Theta_full[j, :]) ** 2
                ) / dist_**3  # (D,)

    return hess_diag_distances


class SensorLocForwardMap(ForwardMap):
    r"""Forward model corresponding to the sensor localization problem. For every sensor :math:`n \in [1, N]`

    .. math::

        f : \theta_n \in \mathbb{R}^D \mapsto f(\theta_n) \in \mathbb{R}^L
    """

    __slots__ = ("D", "L", "N", "Theta_ref", "arr_fixed_values")

    def __init__(self, D: int, L: int, N: int, Theta_ref: np.ndarray) -> None:
        self.D = D
        r"""int: dimensionality of input space"""
        self.L = L
        r"""int: dimensionality of output space"""
        self.N = N
        r"""int: number of independant pixels"""

        self.Theta_ref = Theta_ref  # (K, D)
        r"""np.ndarray: positions of the known sensors"""

        self.arr_fixed_values = np.zeros((self.D,))
        r"""np.ndarray of shape (D,): unused array, kept for compatibility."""

    def evaluate(self, Theta: np.ndarray) -> np.ndarray:
        return evaluate_distances(Theta, self.Theta_ref)

    def evaluate_candidates_one_n(
        self, Theta_candidates: np.ndarray, Theta_t: np.ndarray, n: int
    ) -> np.ndarray:
        r"""evaluate the L distances for each of a set of N_candidates candidate position vectors to replace the position of the nth sensor in current iterate.

        Parameters
        ----------
        Theta_candidates : np.ndarray of shape (N_candidates, D)
            set of N_candidates candidate position vectors
        Theta_t : np.ndarray of shape (N, D)
            current set of positions
        n : int
            index of the considered sensor

        Returns
        -------
        np.ndarray of shape (N_candidates, L)
            set of L distances for each of the N_candidates candidates.
        """
        assert 0 <= n <= self.N - 1
        return evaluate_distances_candidates(
            Theta_candidates, Theta_t, n, self.Theta_ref
        )

    def gradient(self, Theta: np.ndarray) -> np.ndarray:
        return compute_gradient_distance(Theta, self.Theta_ref)

    def hess_diag(self, Theta: np.ndarray) -> np.ndarray:
        return compute_hess_diag_distance(Theta, self.Theta_ref)

    def compute_all(
        self,
        Theta: np.ndarray,
        compute_lin: bool = True,
        compute_log: bool = True,  # unused, just to match the signature
        compute_derivatives: bool = True,
    ) -> dict:
        forward_map_evals = dict()
        forward_map_evals["Theta"] = Theta * 1

        f_Theta = self.evaluate(Theta)

        #! not necessarily N (in candidates testing case for MTM)
        N_pix = f_Theta.shape[0]
        forward_map_evals["f_Theta"] = f_Theta
        # print(forward_map_evals["f_Theta"])

        if compute_derivatives:
            forward_map_evals["grad_f_Theta"] = self.gradient(Theta)
            forward_map_evals["hess_diag_f_Theta"] = self.hess_diag(Theta)

        return forward_map_evals
