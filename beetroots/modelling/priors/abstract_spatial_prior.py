import abc
import os
from typing import List, Union

import numpy as np
import pandas as pd

from beetroots.modelling.priors.abstract_prior import PriorProbaDistribution
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams


def build_list_edges(
    df: pd.DataFrame,
    use_next_nearest_neighbors: bool = True,
) -> np.ndarray:
    r"""builds the list of edges necessary in the rest of the functions of this file

    Parameters
    ----------
    df : pandas.DataFrame
        observation map data. The index needs to contain 2 columns (one for the x-coordinate of the pixel and one for its y-coordinate)
    use_next_nearest_neighbors : bool, optional
        wether or not to use the next nearest neighbors, i.e., in diagonal, by default True

    Returns
    -------
    list_edges : numpy.array of shape (-1, 2)
        list of 2 by 2 neighboring relations without duplicates (each pair is ordered by lowest id to highest id)
    """
    df_clusters = df.copy()
    df_clusters = df_clusters.reset_index().set_index("idx")
    df_clusters["clusters"] = 0
    df_clusters = df_clusters["clusters"]

    list_edges = []
    list_considered_neighbors = (
        [(1, 0), (0, 1), (1, 1), (-1, 1)]
        if use_next_nearest_neighbors
        else [(1, 0), (0, 1)]
    )
    for (x, y) in list(df.index):
        idx = df.at[(x, y), "idx"]
        cluster_current = df_clusters.at[idx]
        for (delta_x, delta_y) in list_considered_neighbors:
            x2 = x + delta_x
            y2 = y + delta_y
            if (x2, y2) in df.index:
                idx2 = df.at[(x2, y2), "idx"]
                if df_clusters.at[idx2] == cluster_current:
                    list_edges.append([idx, idx2])
    list_edges = np.array(list_edges)
    # list_edges = list_edges[list_edges[:, 0].argsort()]
    print(f"total number of edges in img graph : {list_edges.size // 2}")
    return list_edges


class SpatialPrior(PriorProbaDistribution):
    r"""Abstract Base Class for a spatial regularizing prior"""

    __slots__ = (
        "D",
        "N",
        "use_next_nearest_neighbors",
        "list_edges",
        "dict_sites",
        "initial_weights",
        "weights",
    )

    def __init__(
        self,
        spatial_prior_params: SpatialPriorParams,
        cloud_name: str,
        D: int,
        N: int,
        df: pd.DataFrame,
        list_idx_sampling: List[int],
    ) -> None:
        super().__init__(D, N)

        # list of neighboring relations
        self.use_next_nearest_neighbors = (
            spatial_prior_params.use_next_nearest_neighbors
        )
        r"""bool: wether to use the next nearest neighbors (i.e., neighbors in diagonal)"""

        self.list_edges = build_list_edges(df, self.use_next_nearest_neighbors)
        r"""np.ndarray: set of edges in the graph induced by the spatial regularization"""

        self.dict_sites = self.build_sites(df)
        """dict[int, np.ndarray]: sites of the graph induced the spatial regularization. A site is a set of nodes that are conditionally independent."""

        # one weight per dimension
        initial_weights = spatial_prior_params.initial_regu_weights * 1

        if initial_weights is not None:
            if isinstance(initial_weights, list):
                initial_weights = np.array(initial_weights)

            initial_weights = initial_weights[list_idx_sampling] * 1
            weights = initial_weights[list_idx_sampling] * 1

            # assert self.initial_weights.shape == (
            #     self.D,
            # ), f"{self.initial_weights.shape} should be size {self.D}"
        else:
            initial_weights = np.ones((D,))
            weights = np.ones((D,))

        self.initial_weights = initial_weights
        r"""np.ndarray of shape (D,): initial weights of the spatial regularization :math:`\tau`"""

        self.weights = weights
        r"""np.ndarray of shape (D,): current value of weights of the spatial regularization :math:`\tau`"""

        return

    def build_sites(self, df: pd.DataFrame) -> dict[int, np.ndarray]:
        """builds the site from the DataFrame of positions

        Parameters
        ----------
        df : pd.DataFrame
            positions of the pixels

        Returns
        -------
        dict[int, np.ndarray]
            set of sites and corresponding pixels. Forms a partition of the full set of pixels.
        """
        if self.use_next_nearest_neighbors:
            dict_sites_raw = {i: [] for i in range(4)}
            for (x, y) in list(df.index):
                idx_site = 2 * (x % 2) + y % 2
                dict_sites_raw[idx_site].append(df.at[(x, y), "idx"])
        else:
            dict_sites_raw = {i: [] for i in range(2)}
            for (x, y) in list(df.index):
                idx_site = (x + y) % 2
                dict_sites_raw[idx_site].append(df.at[(x, y), "idx"])

        dict_sites = {}
        for k, list_idx in dict_sites_raw.items():
            dict_sites[k] = np.sort(list_idx)

        return dict_sites

    @abc.abstractmethod
    def neglog_pdf(
        self,
        Theta: np.ndarray,
        with_weights: bool = True,
    ) -> np.ndarray:
        r"""computes the negative log pdf (defined up to some multiplicative constant)

        Parameters
        ----------
        Theta : np.array of shape (N, D)
            set of D maps on which we want to evaluate the neg log prior

        Returns
        -------
        neglog_p : np.array of shape (D,)
            set of the D neg log priors evaluation
        """
        pass

    @abc.abstractmethod
    def neglog_pdf_one_pix(
        self, Theta: np.ndarray, n: int, with_weights: bool = True
    ) -> np.ndarray:
        r"""computes the negative log of the pdf (defined up to some multiplicative constant) in the neighborhood of one pixel

        Parameters
        ----------
        Theta : np.array of shape (N, D)
            set of D maps on which we want to evaluate the neg log prior

        Returns
        -------
        neglog_p : np.array of shape (D,)
            set of the D neg log priors evaluation
        """
        pass

    @abc.abstractmethod
    def gradient_neglog_pdf(self, Theta: np.ndarray) -> np.ndarray:
        r"""Computes the gradient of the spatial regularization

        Parameters
        ----------
        Theta : np.array of shape (N, D)
            current iterate

        Returns
        -------
        np.array of shape (N, D)
            gradient of the spatial regularization
        """
        pass

    @abc.abstractmethod
    def hess_diag_neglog_pdf(self, Theta: np.ndarray) -> np.ndarray:
        r"""Computes the diagonal of the hessian of the spatial regularization

        Parameters
        ----------
        Theta : np.array of shape (N, D)
            current iterate

        Returns
        -------
        np.array of shape (N, D)
            diagonal of the hessian of the spatial regularization
        """
        pass
