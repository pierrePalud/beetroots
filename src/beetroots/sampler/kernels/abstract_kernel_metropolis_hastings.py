import abc
from typing import Dict, Union

from beetroots.sampler.kernels.abstract_kernel import Kernel

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp


class KernelMetropolisHastings(Kernel):
    r"""
    KernelMetropolisHastings is an abstract base class for implementing proposal
    distributions based on the Metropolis-Hastings accept/reject step.
    -------
    accept_reject
        Performs the Metropolis-Hastings accept/reject step to decide whether to
        accept or reject proposed candidates. Returns the updated variables,
        acceptance array, and log acceptance probabilities.
    neglog_proposal_candidate_given_current
        Abstract method to compute the negative log-probability of the proposal
        distribution for candidates given the current state. Must be implemented
        by subclasses.
    neglog_proposal_current_given_candidate
        Abstract method to compute the negative log-probability of the proposal
        distribution for the current state given the candidates. Must be implemented
        by subclasses.
    Notes
    -----
    - The `accept_reject` method supports different acceptance-wise types:
      - "pixelwise": Acceptance is determined pixel by pixel.
      - "entire_map": Acceptance is determined for the entire map.
      - "full": Acceptance is determined for each individual element, e.g. each couple \((n, l) \in [1, N] \times [1, L]\).
    - The `joint` parameter in `accept_reject` determines whether the variables
      are updated jointly (as a dictionary) or individually (as arrays).
    """

    r"""Abstract Base Class for proposal distributions based on Matropolis-Hastings accept/reject step."""

    def accept_reject(
        self,
        candidates: Union[Dict[str, xp.ndarray], xp.ndarray],
        current_var: Union[Dict[str, xp.ndarray], xp.ndarray],
        neglogpdf_candidates: xp.ndarray,
        neglogpdf_current: xp.ndarray,
        neglog_q_current_given_candidate: xp.ndarray,
        neglog_q_candidate_given_current: xp.ndarray,
        acceptance_wise_type: str,
        rng: xp.random.Generator,
        joint: bool = False,
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
        Returns:
            tuple:
                - new_var (Union[Dict[str, xp.ndarray], xp.ndarray]):
                  The updated variable(s) after the accept-reject step.
                - accept_arr (xp.ndarray):
                  Boolean array indicating whether each candidate was accepted.
                - log_proba_accept (xp.ndarray):
                  Logarithm of the acceptance ratio for each candidate.
        """
        log_proba_accept = (
            -neglogpdf_candidates
            - neglog_q_current_given_candidate
            + neglogpdf_current
            + neglog_q_candidate_given_current
        )
        if isinstance(log_proba_accept, float):
            size = 1
        else:
            size = log_proba_accept.shape

        log_u = xp.log(rng.uniform(0, 1, size=size))
        accept_arr = log_u < log_proba_accept

        if acceptance_wise_type == "pixelwise":
            accept_arr = accept_arr[:, None]
        elif acceptance_wise_type == "entire_map":
            accept_arr = xp.array(accept_arr)[:, None]
        elif acceptance_wise_type == "full":
            pass

        if joint:
            new_var = {
                key: xp.where(accept_arr, candidates[key], current_var[key])
                for key in candidates.keys()
            }
        else:
            new_var = xp.where(accept_arr, candidates, current_var)

        return new_var, accept_arr.squeeze(), log_proba_accept

    @abc.abstractmethod
    def neglog_proposal_candidate_given_current(
        self,
        candidates: Union[Dict[str, xp.ndarray], xp.ndarray],
        current_var: Union[Dict[str, xp.ndarray], xp.ndarray],
        idx_pix: xp.ndarray,
    ):
        """
        Abstract method to compute the negative log-probability of the proposal
        distribution for candidates given the current state.
        Parameters:
            candidates (Union[Dict[str, xp.ndarray], xp.ndarray]):
                The candidate samples.
            current_var (Union[Dict[str, xp.ndarray], xp.ndarray]):
                The current variable(s) to be updated.
            idx_pix (xp.ndarray):
                Indices of the pixels to be considered.
        Returns:
            xp.ndarray:
                Negative log-probability of the proposal distribution for candidates given the current state.
        """
        pass

    @abc.abstractmethod
    def neglog_proposal_current_given_candidate(
        self,
        candidates: Union[Dict[str, xp.ndarray], xp.ndarray],
        current_var: Union[Dict[str, xp.ndarray], xp.ndarray],
        idx_pix: xp.ndarray,
    ):
        """
        Abstract method to compute the negative log-probability of the proposal
        distribution for the current state given the candidates.
        Parameters:
            candidates (Union[Dict[str, xp.ndarray], xp.ndarray]):
                The candidate samples.
            current_var (Union[Dict[str, xp.ndarray], xp.ndarray]):
                The current variable(s) to be updated.
            idx_pix (xp.ndarray):
                Indices of the pixels to be considered.
        Returns:
            xp.ndarray:
                Negative log-probability of the proposal distribution for the current state given the candidates.
        """
        pass
