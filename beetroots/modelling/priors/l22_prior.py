"""(deprecated) Implementation of L2 norm on image gradient prior
"""

import numpy as np

from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior


class L22SpatialPrior(SpatialPrior):
    r"""L22 smooth spatial prior, valid for both 1D and 2D tensors. Its pdf is defined as

    .. math::

        \forall d \in [1, D], \quad p(x_d) \propto \exp \left[- \tau_d \Vert \nabla Theta_d \Vert_F^2 \right]

    where  :math:`\Vert \cdot \Vert_F` denotes the Fröbenius norm
    """

    def neglog_pdf(
        self,
        Theta: np.ndarray,
        with_weights: bool = True,
    ) -> np.ndarray:
        assert Theta.shape == (self.N, self.D)

        neglog_p = np.zeros((self.D,))

        if self.list_edges.size > 0:
            neglog_p += np.sum(
                (Theta[self.list_edges[:, 1], :] - Theta[self.list_edges[:, 0], :])
                ** 2,
                axis=0,
            )
        if with_weights:
            neglog_p *= self.weights

        return neglog_p  # (D,)

    def neglog_pdf_one_pix(
        self,
        Theta: np.ndarray,
        n: int,
        list_pixel_candidates: np.ndarray,
    ) -> np.ndarray:
        """
        computes the neg log-prior when only one pixel is modified

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            current iterate

        n : int
            the index of the pixel to consider (0 <= n <= N - 1)

        list_pixel_candidates : np.ndarray of shape (N_candidates, D)
            the list of all candidates for pixel n

        Returns
        -------
        nl_priors : np.ndarray of shape (N_candidates,)
            the leg log-prior of the candidates
        """
        assert Theta.shape == (self.N, self.D)
        assert 0 <= n <= self.N - 1
        # TODO : à reprendre

        N_candidates = list_pixel_candidates.shape[0]

        neglog_p = np.zeros_like(list_pixel_candidates)  # (N_candidates, D)

        list_edges_pix = self.list_edges[(self.list_edges[:, 0] == n)]

        if list_edges_pix.size > 0:
            neglog_p += (
                Theta[list_edges_pix[:, 1], :] - Theta[list_edges_pix[:, 0], :]
            ) ** 2  # (N_candidates, D)
        neglog_p = neglog_p * self.weights[None, :]
        neglog_p = np.sum(neglog_p, axis=1)
        return neglog_p  # (D,)

    def gradient_neglog_pdf(self, x: np.ndarray) -> np.ndarray:
        assert Theta.shape == (self.N, self.D)

        g = np.zeros_like(Theta)
        for edge in self.list_edges:
            val = 2 * (Theta[edge[1]] - Theta[edge[0]])  # (D,)
            g[edge[0]] -= val
            g[edge[1]] += val

        g = self.weights[None, :] * g
        return g  # (N, D)

    def hess_diag_neglog_pdf(self, Theta: np.ndarray) -> np.ndarray:
        hess_diag = np.zeros_like(x, dtype=np.float64)
        if self.list_edges.size > 0:
            idx, counts = np.unique(self.list_edges.flatten(), return_counts=True)
            # print(counts.dtype, hess_diag.dtype)
            hess_diag[idx, :] += 2 * counts[:, None] * np.ones((idx.size, self.D))

        hess_diag = self.weights[None, :] * hess_diag
        return hess_diag  # (N, D)
