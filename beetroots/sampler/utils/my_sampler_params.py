"""Defines a Dataclass that stores the parameters of the augmented PSGLD sampler
"""
from typing import List, Union

import numpy as np
import pandas as pd


class MySamplerParams(object):
    r"""Dataclass that stores the parameters of the sampler proposed in :cite:t:`paludEfficientSamplingNon2023`"""

    __slots__ = (
        "initial_step_size",
        "extreme_grad",
        "history_weight",
        "selection_probas",
        "k_mtm",
        "is_stochastic",
        "compute_correction_term",
    )

    def __init__(
        self,
        initial_step_size: float,
        extreme_grad: float,
        history_weight: float,
        selection_probas: Union[np.ndarray, List[float]],
        k_mtm: int,
        is_stochastic: bool = True,
        compute_correction_term: bool = True,
    ) -> None:
        r"""

        Parameters
        ----------
        initial_step_size : float
            step size used in the Position-dependent MALA transition kernel, denoted :math:`\epsilon` in the article
        extreme_grad : float
            limit value that avoids division by zero when computing the RMSProp preconditioner, denoted :math:`\eta` in the article
        history_weight : float
            weight of past values of :math:`v` in the exponential decay (cf RMSProp preconditioner), denoted :math:`\alpha` in the article
        selection_probas : np.ndarray of shape (2,)
            vector of selection probabilities for the MTM and PMALA kernels, respectively, i.e., :math:`[p_{MTM}, 1 - p_{MTM}]`
        k_mtm : int
            number of candidates in the MTM kernel, denoted :math:`K` in the article
        is_stochastic : bool
            if True, the algorithm performs sampling, and optimization otherwise, by default True
        compute_correction_term : bool
            wether or not to use the correction term (denoted :math:`\gamma` in the article) during the sampling (only used if `is_stochastic=True`), by default True
        """
        assert initial_step_size > 0
        assert extreme_grad > 0
        assert 0 < history_weight < 1

        assert isinstance(k_mtm, int) and k_mtm >= 1

        if isinstance(selection_probas, list):
            selection_probas = np.array(selection_probas)

        assert np.all(0 <= selection_probas)
        assert selection_probas.sum() == 1

        assert isinstance(is_stochastic, bool)
        assert isinstance(compute_correction_term, bool)

        self.initial_step_size = initial_step_size
        self.extreme_grad = extreme_grad
        self.history_weight = history_weight

        self.selection_probas = selection_probas
        self.k_mtm = k_mtm

        self.is_stochastic = is_stochastic
        self.compute_correction_term = compute_correction_term
