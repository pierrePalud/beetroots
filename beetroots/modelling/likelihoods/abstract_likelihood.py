from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np


class Likelihood(ABC):
    r"""Abstract Base Class for a probability distribution on non-countable set"""

    def __init__(
        self,
        forward_map,
        D: int,
        L: int,
        N: int,
        y: np.ndarray,
    ) -> None:
        self.forward_map = forward_map
        self.D = D
        self.L = L
        self.N = N
        self.y = y
        self.hyperparameters = None

        assert y.shape == (N, L)

    def _update_observations(self, y: np.ndarray):
        r"""Update observations :math:`y` to be updated as :math:`y_{\text{new}}`  whenever the likelihood object :math:`p(y \mid f(\theta)` is interpreted as
        a prior on :math:`y` with hyperparameter :math:`\theta`.

        Parameters
        ----------
        y : np.ndarray
            New state of the observations.

        Example
        -------
        # forward_map_eval contains the parameter \theta
        lklhd._update_observations(y_new)
        lklhd.neglog_pdf(forward_map_eval, nll_utils)

        Note
        ----
        Allows evaluation of :math:`-\log p(y_{\text{new}} \mid \theta)`.
        """
        self.y = y

    @abstractmethod
    def neglog_pdf(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def sample_observation_model(
        self, forward_map_evals: dict, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_neglog_pdf(
        self,
        forward_map_evals: dict[str, np.ndarray],
        nll_utils: dict[str, np.ndarray],
    ) -> np.ndarray:
        pass

    def neglog_pdf_candidates(
        self,
        candidates: np.ndarray,
        idx: np.ndarray,
        Theta_t: Optional[np.ndarray] = None,
        return_forward_map_evals: bool = False,
    ) -> np.ndarray:
        assert len(candidates.shape) == 2 and candidates.shape[1] == self.D
        assert isinstance(idx, np.ndarray)
        assert np.all(0 <= idx)
        assert np.all(idx <= self.N - 1)

        N_candidates = candidates.shape[0]
        n_pix = idx.size

        forward_map_evals = self.evaluate_all_forward_map(
            candidates, compute_derivatives=False, compute_derivatives_2nd_order=False
        )
        nll_utils = self.evaluate_all_nll_utils(
            forward_map_evals,
            idx=idx,
            compute_derivatives=False,
            compute_derivatives_2nd_order=False,
        )

        nll_candidates = self.neglog_pdf(
            forward_map_evals,
            nll_utils,
            pixelwise=True,
            idx=idx,
        )  # (N_candidates,)
        assert isinstance(nll_candidates, np.ndarray)
        assert nll_candidates.shape == (N_candidates,)

        if return_forward_map_evals:
            return nll_candidates, forward_map_evals

        else:
            return nll_candidates

    @abstractmethod
    def hess_diag_neglog_pdf(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> np.ndarray:
        pass

    def evaluate_all_forward_map(
        self,
        Theta: np.ndarray,
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
    ) -> dict[str, Union[float, np.ndarray]]:
        assert len(Theta.shape) == 2 and Theta.shape[1] == self.D
        forward_map_evals = self.forward_map.compute_all(
            Theta, True, True, compute_derivatives, compute_derivatives_2nd_order
        )
        return forward_map_evals

    @abstractmethod
    def evaluate_all_nll_utils(
        self,
        forward_map_evals: dict[str, Union[float, np.ndarray]],
        idx: Optional[np.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
    ) -> dict[str, Union[float, np.ndarray]]:
        pass
