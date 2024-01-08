"""Dataclass to contain data on the spatial prior
"""
from typing import Union

import numpy as np


class SpatialPriorParams:

    __slots__ = (
        "name",
        "use_next_nearest_neighbours",
        "initial_regu_weights",
        "use_clustering",
        "n_clusters",
        "cluster_algo",
    )

    def __init__(
        self,
        name: str,
        use_next_nearest_neighbours: bool,
        initial_regu_weights: np.ndarray,
        # not used anymore (clustering of pixels)
        use_clustering: bool = False,
        n_clusters: Union[int, None] = None,
        cluster_algo: Union[str, None] = None,
    ) -> None:

        assert name in ["L2-laplacian", "L2-gradient"]
        assert isinstance(use_clustering, bool)
        assert n_clusters is None or n_clusters >= 2

        list_valid_algo = ["spectral_clustering", "kmeans"]
        assert cluster_algo is None or cluster_algo in list_valid_algo

        self.name = name
        self.use_next_nearest_neighbours = use_next_nearest_neighbours
        self.initial_regu_weights = initial_regu_weights
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters
        self.cluster_algo = cluster_algo
