import abc

import matplotlib.pyplot as plt
from matplotlib import patches


class AbstractPlots2D(abc.ABC):
    """abstract class for drawing figures of maps of observations, noise standard deviations, estimated physical parameters, estimated credibility interval sizes, etc."""

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
