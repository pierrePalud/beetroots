from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class PriorProbaDistribution(ABC):
    r"""Abstract Base Class for a probability distribution on non-countable set"""

    def __init__(self, D: int, N: int) -> None:
        self.D = D
        """int: number of distinct physical parameters"""
        self.N = N
        """int: number of pixels in each physical dimension"""

    @abstractmethod
    def neglog_pdf(self, Theta: np.ndarray) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def gradient_neglog_pdf(self, Theta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def hess_diag_neglog_pdf(self, Theta: np.ndarray) -> np.ndarray:
        pass
