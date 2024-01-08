"""Defines a Dataclass that stores the parameters of the augmented PSGLD sampler
"""
from typing import List, Union

import numpy as np
import pandas as pd


class PSGLDParams(object):
    """Data class that stores the parameters of the augmented PSGLD sampler

    Note : selection_probas = (proba MTM, proba PMALA)
    """

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
        """

        Parameters
        ----------
        initial_step_size : float
            step size used in the Position-dependent MALA transition kernel
        extreme_grad : float
            limit value that avoids division by zero when computing the RMSProp preconditioner
        history_weight : float
            weight of past values of `V` in the exponential decay (cf RMSProp preconditioner)
        is_stochastic : bool
            if True, the algorithm performs sampling, and optimization otherwise, by default True
        compute_correction_term : bool
            wether or not to use the correction term during the sampling (only used if `is_stochastic=True`), by default True
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

    def save_to_file(
        self,
        results_path: str,
        filename: str = "sampling_params.csv",
    ):
        list_dict_ = [
            {
                "name": "selection probas",
                "value": self.selection_probas,
            },
            {
                "name": "stochastic (true) optimization (false)",
                "value": self.is_stochastic,
            },
            {
                "name": "step size",
                "value": self.initial_step_size,
            },
            {
                "name": "v damping",
                "value": self.extreme_grad,
            },
            {
                "name": "exponential decay rate",
                "value": self.history_weight,
            },
            {
                "name": "number of wether or not to compute the correction term",
                "value": self.compute_correction_term,
            },
            {
                "name": "number of candidates",
                "value": self.k_mtm,
            },
        ]
        df = pd.DataFrame(list_dict_)
        df.to_csv(f"{results_path}/{filename}")
