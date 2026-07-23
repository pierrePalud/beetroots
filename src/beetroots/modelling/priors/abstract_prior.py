from abc import abstractmethod
from typing import Optional, Union

from beetroots.modelling.component_distribution import ComponentDistribution

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp


class PriorProbaDistribution(ComponentDistribution):
    r"""Abstract Base Class for a probability distribution on non-countable set"""

    def __init__(self, D: int, N: int, var_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.D = D
        """int: number of distinct physical parameters"""
        self.N = N
        """int: number of pixels in each physical dimension"""
        self.var_name = var_name
        """str: name of the variable of the target distribution"""

    @abstractmethod
    def neglog_pdf(
        self,
        Var: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
        chromatic_gibbs: Optional[bool] = False,
    ) -> Union[float, xp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def gradient_neglog_pdf(self, var: xp.ndarray) -> xp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def hess_diag_neglog_pdf(self, var: xp.ndarray) -> xp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
    ) -> None:
        """Evaluate all utilities for the negative log-pdf"""
        raise NotImplementedError
