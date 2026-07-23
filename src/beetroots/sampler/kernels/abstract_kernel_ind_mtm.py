from typing import Dict, Union

from scipy.special import softmax

from beetroots.sampler.kernels.abstract_kernel import Kernel

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp


class KernelIndependentMTM(Kernel):
    r"""Abstract Base Class for proposal distributions based on Matropolis-Hastings accept/reject step."""

    def accept_reject(
        self,
        candidates: Union[Dict[str, xp.ndarray], xp.ndarray],
        current_var: Union[Dict[str, xp.ndarray], xp.ndarray],
        neglogpdf_candidates_current: xp.ndarray,
        neglog_proposal_candidates_current: xp.ndarray,
        acceptance_wise_type: str,
        rng: xp.random.Generator,
        joint: bool = False,
    ) -> tuple:
        r"""
        Perform an accept-reject step for sampling candidates based on their weights.
        Parameters:
            candidates (Union[Dict[str, xp.ndarray], xp.ndarray]):
                The candidate samples.
            current_var (Union[Dict[str, xp.ndarray], xp.ndarray]):
                The current variable(s) to be updated.
            neglogpdf_candidates_current (xp.ndarray):
                Negative log-pdf values for the candidateand the current state w.r.t. the target distribution.
            neglog_proposal_candidates_current (xp.ndarray):
                Negative log-proposal values for the candidates and the current state.
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
                - log_rg (xp.ndarray):
                  Logarithm of the acceptance ratio for each candidate.
        """

        # * compute the weights themselves
        neglogpdf_candidates = (
            neglogpdf_candidates_current - neglog_proposal_candidates_current
        )

        neglogpdf_candidates_min = xp.amin(neglogpdf_candidates, axis=1, keepdims=True)
        neglogpdf_candidates_centered = (
            neglogpdf_candidates - neglogpdf_candidates_min
        )  # Center the importance weights (equal shift for all candidates of a given a pixel nd) for numerical purposes in the softmax function

        weights = softmax(-neglogpdf_candidates_centered[:, :-1], axis=1)
        assert xp.sum(1 - xp.isfinite(weights)) == 0, weights

        # * choose challenger candidate
        cumulative_weights = xp.cumsum(weights, axis=1)
        u_challengers = rng.random(
            (neglogpdf_candidates.shape[:1] + (1,) + neglogpdf_candidates.shape[2:])
        )
        idx_challengers = (cumulative_weights < u_challengers).sum(axis=1)

        # * accept/reject step
        neglogpdf_challengers = xp.take_along_axis(
            neglogpdf_candidates_centered, idx_challengers[:, None], axis=1
        ).squeeze(1)

        pdf_candidates = xp.exp(-neglogpdf_candidates_centered)
        log_numerators = xp.log(xp.sum(pdf_candidates[:, :-1], axis=1))
        log_denominators = xp.log(
            xp.sum(pdf_candidates, axis=1) - xp.exp(-neglogpdf_challengers)
        )

        log_rg = log_numerators - log_denominators
        log_rg = xp.where(~xp.isnan(log_rg), log_rg, -1e15)

        log_u = xp.log(rng.random(log_rg.shape))
        accept_arr = log_u < log_rg

        if acceptance_wise_type == "pixelwise":
            accept_arr = accept_arr[:, None]
        elif acceptance_wise_type == "entire_map":
            accept_arr = xp.array(accept_arr)[:, None]
        elif acceptance_wise_type == "full":
            pass

        idx_challengers_expanded = idx_challengers
        if idx_challengers.ndim == 1:
            idx_challengers_expanded = idx_challengers[:, None]

        if joint:
            candidate_challenger = {
                key: xp.take_along_axis(
                    candidates[key], idx_challengers_expanded[:, None], axis=1
                ).squeeze(1)
                for key in candidates.keys()
            }

            new_var = {
                key: xp.where(accept_arr, candidate_challenger[key], current_var[key])
                for key in candidates.keys()
            }
        else:
            candidate_challenger = xp.take_along_axis(
                candidates, idx_challengers_expanded[:, None], axis=1
            ).squeeze(1)

            new_var = xp.where(accept_arr, candidate_challenger, current_var)

        return new_var, accept_arr.squeeze(), log_rg
