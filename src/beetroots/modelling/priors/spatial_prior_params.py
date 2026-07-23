from typing import List, Union

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp


class SpatialPriorParams:
    r"""Dataclass to contain data on the spatial prior"""

    __slots__ = (
        "name",
        "use_next_nearest_neighbors",
        "initial_regu_weights",
    )

    def __init__(
        self,
        name: str,
        use_next_nearest_neighbors: bool,
        initial_regu_weights: Union[xp.ndarray, List[float]],
    ) -> None:
        r"""

        Parameters
        ----------
        name : str
            name of the spatial regularization type, must be an element of ["L2-laplacian", "L2-gradient"]
        use_next_nearest_neighbors : bool
            wether or not to use the next nearest neighbors, i.e., in diagonal
        initial_regu_weights : Union[xp.ndarray, List[float]]
            initial regularization weights (the regularization weights can be tuned automatically during the Markov chain)
        """
        assert name in ["L2-laplacian", "L2-discrete-grad"]

        self.name = name
        r"""str: name of the spatial regularization type, must be an element of ["L2-laplacian", "L2-gradient"]"""

        self.use_next_nearest_neighbors = use_next_nearest_neighbors
        r"""bool: wether or not to use the next nearest neighbors, i.e., in diagonal"""

        self.initial_regu_weights = initial_regu_weights
        r"""Union[xp.ndarray, List[float]: initial regularization weights (the regularization weights can be tuned automatically during the Markov chain)"""
