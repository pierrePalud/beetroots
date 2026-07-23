from abc import abstractmethod
from typing import Optional, Union

from beetroots.modelling.component_distribution import ComponentDistribution

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp


class Likelihood(ComponentDistribution):
    r"""Abstract Base Class for a probability distribution on non-countable set"""

    def __init__(
        self,
        forward_map,
        y: xp.ndarray,
        D: int,
        L: int,
        N: int,
        var_name: str = "",
        **kwargs,
    ) -> None:
        self.forward_map = forward_map
        self.forward_map_evals = {}
        """dict: forward map evaluations (log and derivatives)"""
        self.D = D
        self.L = L
        self.N = N
        self.y = y
        self.var_name = var_name
        if not (y.shape == (self.N, self.L)):
            raise ValueError(
                "y must have the shape (N, L) = ({}, {}) elements".format(
                    self.N, self.L
                )
            )
        elif isinstance(y, xp.ndarray):
            self.y = y
        else:
            raise ValueError("y must be an array")

        self.hyperparameters = None

        super().__init__(**kwargs)

        assert y.shape == (N, L)

    def _update_observations(self, y: xp.ndarray):
        r"""Update observations :math:`y` to be updated as :math:`y_{\text{new}}`  whenever the likelihood object :math:`p(y \mid f(\theta)` is interpreted as
        a prior on :math:`y` with hyperparameter :math:`\theta`.

        Parameters
        ----------
        y : xp.ndarray
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
        idx_pix: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def sample_observation_model(
        self,
        current: dict,
        rng: Optional[
            xp.random.Generator
        ],  # Used for model checking (maybe for something else)
    ) -> xp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def gradient_neglog_pdf(
        self,
        forward_map_evals: dict[str, xp.ndarray],
        nll_utils: dict[str, xp.ndarray],
    ) -> xp.ndarray:
        raise NotImplementedError

    def neglog_pdf_candidates(
        self,
        candidates: xp.ndarray,
        idx: xp.ndarray,
        return_forward_map_evals: bool = False,
    ) -> xp.ndarray:
        assert len(candidates.shape) == 2 and candidates.shape[1] == self.D
        assert isinstance(idx, xp.ndarray)
        assert xp.all(0 <= idx)
        assert xp.all(idx <= self.N - 1)

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
        assert isinstance(nll_candidates, xp.ndarray)
        assert nll_candidates.shape == (N_candidates,)

        if return_forward_map_evals:
            return nll_candidates, forward_map_evals

        else:
            return nll_candidates

    @abstractmethod
    def hess_diag_neglog_pdf(
        self, forward_map_evals: dict, nll_utils: dict
    ) -> xp.ndarray:
        raise NotImplementedError

    def evaluate_all_forward_map(
        self,
        Var: xp.ndarray,
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
    ) -> dict[str, Union[float, xp.ndarray]]:
        assert len(Var.shape) == 2 and Var.shape[1] == self.D

        forward_map_evals = self.forward_map.compute_all(
            Var, True, True, compute_derivatives, compute_derivatives_2nd_order
        )
        self.forward_map_evals = forward_map_evals

    @abstractmethod
    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
    ) -> None:
        raise NotImplementedError
