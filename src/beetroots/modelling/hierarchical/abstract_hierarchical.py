from abc import abstractmethod
from typing import Optional, Union

from beetroots.modelling.component_distribution import ComponentDistribution

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed


class Hierarchical(ComponentDistribution):
    r"""Abstract Base Class for a probability distribution on non-countable set"""

    @abstractmethod
    def neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
        **kwargs,
    ) -> Union[float, xp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def gradient_neglog_pdf(
        self,
        var_name: str,
    ) -> xp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def hess_diag_neglog_pdf(
        self,
        var_name: str,
    ) -> xp.ndarray:
        raise NotImplementedError
