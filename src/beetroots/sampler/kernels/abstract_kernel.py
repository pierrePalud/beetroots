try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

import abc
from typing import Dict, Tuple, Union

from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)


class Kernel(abc.ABC):
    r"""Abstract Base Class for proposal distributions. Children classes should implement at least three methods:

    * sample(): draw samples from the proposal distribution
    * neglog_pdf(): evaluate the neglog-pdf of the proposal distribution
    * get_new_sample(): get a new sample from the proposal distribution, either by sampling or by optimization
    """

    # __slots__ = ("is_stochastic",)

    @abc.abstractmethod
    def sample(
        self,
        Var: xp.ndarray,
        idx_pix: xp.ndarray,
        k_mtm: int,
        rng: xp.random.Generator,
    ) -> xp.ndarray:
        """
        Draw samples from the proposal distribution.
        Parameters:
            Var (xp.ndarray):
                The variable(s) to be updated.
            idx_pix (xp.ndarray):
                The indices of the pixels to be updated.
            k_mtm (int):
                The number of samples to draw.
            rng (xp.random.Generator):
                Random number generator for sampling.
        Returns:
            xp.ndarray:
                The drawn samples from the proposal distribution.
        """
        pass

    @abc.abstractmethod
    def accept_reject(
        self,
        candidates: Union[Dict[str, xp.ndarray], xp.ndarray],
        current_var: Union[Dict[str, xp.ndarray], xp.ndarray],
        neglogpdf_candidates: xp.ndarray,
        neglogpdf_current: xp.ndarray,
        neglog_q_current_given_candidate: xp.ndarray,
        neglog_q_candidate_given_current: xp.ndarray,
        rng: xp.random.Generator,
        joint: bool = False,
        **kwargs,
    ) -> dict:
        r"""
        Perform an accept-reject step for sampling candidates based on their weights.
        Parameters:
            candidates (Union[Dict[str, xp.ndarray], xp.ndarray]):
                The candidate samples.
            current_var (Union[Dict[str, xp.ndarray], xp.ndarray]):
                The current variable(s) to be updated.
            neglogpdf_candidates (xp.ndarray):
                Negative log-pdf values for the candidate and the current state w.r.t. the target distribution.
            neglogpdf_current (xp.ndarray):
                Negative log-pdf values for the current state w.r.t. the target distribution.
            neglog_q_current_given_candidate (xp.ndarray):
                Negative log-proposal values for the current state given the candidates.
            neglog_q_candidate_given_current (xp.ndarray):
                Negative log-proposal values for the candidates given the current state.
            acceptance_wise_type (str):
                Specifies the type of acceptance criteria. Options are:
                - "pixelwise": Acceptance is determined per pixel.
                - "entire_map": Acceptance is determined for the entire map.
                - "full": Acceptance is determined for every element, e.g. every couple \((n, l) \in [1, N] \times [1, L]\).
            rng (xp.random.Generator):
                Random number generator for sampling.
            joint (bool, optional):
                If True, candidates and current_var are dictionaries, and the update is performed jointly.
                Defaults to False.
            kwargs (dict, optional):
                Additional keyword arguments for the accept-reject step.
        """
        pass

    @abc.abstractmethod
    def get_new_sample_sampling(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        """
        Get a new sample from the proposal distribution (MCMC)
        Parameters:
            current (dict):
                The current variable(s) to be updated and related variables.
            t (int):
                The current iteration step.
            target_distributions (Dict[str, TargetDistribution]):
                The target distributions for the proposal distribution.
            rng (xp.random.Generator):
                Random number generator for sampling.
        """
        pass

    @abc.abstractmethod
    def get_new_sample_optim(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        """
        Get a new sample from the proposal distribution (optimization).
        Parameters:
            current (dict):
                The current variable(s) to be updated and related variables.
            t (int):
                The current iteration step.
            target_distributions (Dict[str, TargetDistribution]):
                The target distributions for the proposal distribution.
            rng (xp.random.Generator):
                Random number generator for sampling.
        """
        pass

    def get_new_sample(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool = False,
    ) -> Tuple:
        """
        Get a new sample from the proposal distribution, either by sampling or by optimization.
        Parameters:
            current (dict):
                The current variable(s) to be updated and related variables.
            t (int):
                The current iteration step.
            target_distributions (Dict[str, TargetDistribution]):
                The target distributions for the proposal distribution.
            rng (xp.random.Generator):
                Random number generator for sampling.
        """
        if self.is_stochastic:
            return self.get_new_sample_sampling(
                current=current,
                t=t,
                target_distributions=target_distributions,
                rng=rng,
                compute_derivatives_2nd_order=compute_derivatives_2nd_order,
            )
        elif not self.is_stochastic:
            return self.get_new_sample_optim(
                current=current,
                t=t,
                target_distributions=target_distributions,
                rng=rng,
                compute_derivatives_2nd_order=compute_derivatives_2nd_order,
            )
        else:
            raise ValueError("is_stochastic should be True or False.")


class KernelUtils(abc.ABC):
    r"""Abstract Base Class for proposal distributions. Gather common methods for proposal distributions of a same family."""

    def initialize_hyperparameters(self, **kwargs) -> None:
        """
        Initialize hyperparameters for the proposal distribution.
        Parameters:
            current (dict):
                The current variable(s) to be updated and related variables.
        """
        pass

    def external_update_hyperparameters(self, **kwargs) -> None:
        """
        Update hyperparameters for the proposal distribution based on external factors.
        Parameters:
            current (dict):
                The current variable(s) to be updated and related variables.
            acceptance_stats (dict):
                Statistics related to the acceptance of the last accept/reject step.
        """
        pass

    def internal_update_hyperparameters(self, **kwargs) -> None:
        """
        Update hyperparameters for the proposal distribution based on internal factors.
        Parameters:
            kwargs (dict):
                Additional keyword arguments for the internal update.
        """
        pass

    def _dict_parameters_saving(self, **kwargs) -> None:
        """
        Create a dictionary of parameters to be saved for the proposal distribution.
        Parameters:
            kwargs (dict):
                Additional keyword arguments for the parameters saving.
        Returns:
            dict:
                A dictionary of parameters to be saved for the proposal distribution.
        """
        pass
