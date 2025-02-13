import abc
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib import patches

from beetroots.inversion.plots.map_shaper import MapShaper


class AbstractPlots2D(abc.ABC):
    """abstract class for drawing figures of maps of observations, noise standard deviations, estimated physical parameters, estimated credibility interval sizes, etc."""

    def __init__(
        self,
        map_shaper: MapShaper,
        pixels_of_interest: Dict[int, str] = {},
    ):
        self.map_shaper = map_shaper
        r"""MapShaper: defines the transformation from vectors to 2D maps"""

        self.pixels_of_interest_names = pixels_of_interest
        r"""dict: (coordinate, name) pair of some user-informed pixels to be highlighted. These pixels will be outlines with a black square in figures."""

        coords = map_shaper.from_vector_idx_to_map_coords(
            list(pixels_of_interest.keys())
        )
        self.pixels_of_interest_coords = coords
        r"""list: coordinates of some user-informed pixels to be highlighted. These pixels will be outlines with a black square in figures."""

    def _draw_rect_on_pixels_of_interest(self):
        """Draws a rectangle around the pixels of interest. Requires the class to have a `pixels_of_interest_coords` attribute, as a Dict with the vector index as keys and the corresponding map coordinates as values."""
        ax = plt.gca()

        for _, coords in self.pixels_of_interest_coords.items():
            rect = patches.Rectangle(
                (coords[0] - 0.5, coords[1] - 0.5),
                1,
                1,
                linewidth=1,
                edgecolor="k",
                facecolor="none",
            )
            ax.add_patch(rect)

        return
