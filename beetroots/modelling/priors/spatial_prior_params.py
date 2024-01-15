from typing import List, Union

import numpy as np


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
        initial_regu_weights: Union[np.ndarray, List[float]],
    ) -> None:
        r"""

        Parameters
        ----------
        name : str
            name of the spatial regularization type, must be an element of ["L2-laplacian", "L2-gradient"]
        use_next_nearest_neighbors : bool
            wether or not to use the next nearest neighbors, i.e., in diagonal
        initial_regu_weights : Union[np.ndarray, List[float]]
            initial regularization weights (the regularization weights can be tuned automatically during the Markov chain)
        """
        assert name in ["L2-laplacian", "L2-gradient"]

        self.name = name
        r"""str: name of the spatial regularization type, must be an element of ["L2-laplacian", "L2-gradient"]"""

        self.use_next_nearest_neighbors = use_next_nearest_neighbors
        r"""bool: wether or not to use the next nearest neighbors, i.e., in diagonal"""

        self.initial_regu_weights = initial_regu_weights
        r"""Union[np.ndarray, List[float]: initial regularization weights (the regularization weights can be tuned automatically during the Markov chain)"""
