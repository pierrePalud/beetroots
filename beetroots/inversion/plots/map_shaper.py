from typing import Dict, List

import numpy as np
import pandas as pd


class MapShaper:
    """Defines the transformation from a vector to an map.
    Necessary for non-rectangular maps."""

    __slots__ = ("index_arr", "N")

    def __init__(self, df: pd.DataFrame) -> None:
        """

        Parameters
        ----------
        df : pandas.DataFrame
            should have (x, y) coordinates in the index and one "idx" column to define the transformation
        """
        index_arr = -np.ones(
            (
                df.index.get_level_values(0).max()
                - df.index.get_level_values(0).min()
                + 1,
                df.index.get_level_values(1).max()
                - df.index.get_level_values(1).min()
                + 1,
            )
        )

        min_theta = df.index.get_level_values(0).min()
        min_y = df.index.get_level_values(1).min()

        for (x, y) in df.index:
            index_arr[x - min_theta, y - min_y] = df.loc[(x, y), "idx"]

        # array of the indices of the vector in an image
        self.index_arr = index_arr.astype(int)
        r"""2D np.ndarray: `map` that contains the indices of the vector entries"""

        self.N = len(df)
        r"""int: number of pixels in map, i.e., dimension of vectors"""

    def from_vector_to_map(self, x: np.ndarray) -> np.ndarray:
        r"""applies a reshaping for a vector to plot it in 2D

        Parameters
        ----------
        x : numpy.array of shape (N,) or (N, -1)
            vector or tensor to be reshaped

        Returns
        -------
        x_shaped : numpy.array
            information of x shaped for plotting

        Raises
        ------
        ValueError
            if ``x`` has an invalid shape
        """
        if x.shape == (self.N,):
            x_shaped = np.empty_like(self.index_arr, dtype=np.float64)
            x_shaped[:, :] = np.nan
            for i in range(self.N):
                mask = self.index_arr == i
                x_shaped[mask] = x[i]

            return x_shaped.T

        if len(x.shape) == 2 and x.shape[0] == self.N:
            x_shaped = np.empty(
                (self.index_arr.shape[0], self.index_arr.shape[1], x.shape[1])
            )
            x_shaped[:, :] = np.nan
            for i in range(self.N):
                mask = self.index_arr == i
                x_shaped[mask] = x[i]
            return x_shaped.transpose((1, 0, 2))

        else:
            msg = f"invalid shape for x : {x.shape} "
            msg += f"(should be ({self.N},) or ({self.N}, -1))"
            raise ValueError(msg)

    def from_vector_idx_to_map_coords(
        self,
        idx: List[int],
    ) -> Dict[int, np.ndarray]:
        r"""transforms vector index in map coordinates

        Parameters
        ----------
        idx : numpy.array of shape (N,) or (N, -1)
            vector or tensor to be reshaped

        Returns
        -------
        x_shaped : numpy.array
            information of x shaped for plotting

        Raises
        ------
        ValueError
            if ``x`` has an invalid shape
        """
        x_idx = np.arange(self.N)
        x_idx_shaped = self.from_vector_to_map(x_idx)

        dict_coords = {k: None for k in idx}

        for i in range(x_idx_shaped.shape[0]):
            for j in range(x_idx_shaped.shape[1]):
                if not np.isnan(x_idx_shaped[i, j]):
                    pix_idx_val = int(x_idx_shaped[i, j])

                    for k, v in dict_coords.items():
                        if v is None and k == pix_idx_val:
                            dict_coords[k] = np.array([j, i])
                            break

        for k, v in dict_coords.items():
            assert v is not None, f"no coordinates found for pixel idx={k}"

        return dict_coords
