from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.inversion.plots.plots_2d import AbstractPlots2D


class PlotsEstimator(AbstractPlots2D):
    r"""utilitary class that draws and saves maps related to the inference results."""

    __slots__ = (
        "map_shaper",
        "list_names",
        "lower_bounds_lin",
        "upper_bounds_lin",
        "list_idx_sampling",
        "pixels_of_interest_names",
        "pixels_of_interest_coords",
    )

    def __init__(
        self,
        map_shaper: MapShaper,
        list_names: List[str],
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
        list_idx_sampling: List[int],
        pixels_of_interest: Dict[int, str] = {},
    ):
        self.map_shaper = map_shaper
        r"""MapShaper: defines the transformation from vectors to 2D maps"""

        self.list_names = list_names
        r"""list: physical parameter names"""

        self.lower_bounds_lin = lower_bounds_lin
        r"""1D np.ndarray: contains the lower bounds in linear scale on the physical parameters"""
        self.upper_bounds_lin = upper_bounds_lin
        r"""1D np.ndarray: contains the upper bounds in linear scale on the physical parameters"""

        self.list_idx_sampling = list_idx_sampling
        r"""1D np.ndarray: contains the indices of the physical parameters to be sampled"""

        self.pixels_of_interest_names = pixels_of_interest
        r"""dict: (coordinate, name) pair of some user-informed pixels to be highlighted. These pixels will be outlines with a black square in figures."""

        coords = map_shaper.from_vector_idx_to_map_coords(
            list(pixels_of_interest.keys())
        )
        self.pixels_of_interest_coords = coords
        r"""list: coordinates of some user-informed pixels to be highlighted. These pixels will be outlines with a black square in figures."""

    def plot_estimator(
        self,
        Theta_estimated: np.ndarray,
        estimator_name: str,
        folder_path: str,
        model_name: Optional[str] = "",
    ):
        """plots and saves the 2D map of an estimated physical parameter

        Parameters
        ----------
        Theta_estimated : np.ndarray of shape (N, D)
            vector of the estimated physical parameter
        estimator_name : str
            name of the estimator, e.g., "MLE", "MAP" or "MMSE"
        folder_path : str
            path to the folder where the figure is to be saved
        model_name : Optional[str], optional
            name of the model (not used here, kept for compatibility), by default ""
        """
        Theta_estimated_plot = self.map_shaper.from_vector_to_map(Theta_estimated)
        for d, name in enumerate(self.list_names):
            vmin = self.lower_bounds_lin[d] / 1.1
            vmax = self.upper_bounds_lin[d] * 1.1

            x_estimator_d_plot = Theta_estimated_plot[:, :, d] * 1

            title = f"{estimator_name} for {name}"
            if d not in self.list_idx_sampling:
                title += " (fixed, not estimated)"

            if vmin > 0:
                plt.figure(figsize=(8, 6))
                plt.title(title)
                plt.imshow(
                    x_estimator_d_plot,
                    norm=colors.LogNorm(vmin, vmax),
                    origin="lower",
                    cmap="jet",
                )
                plt.colorbar()
                self._draw_rect_on_pixels_of_interest()
                # plt.tight_layout()

                filename = f"{folder_path}/{estimator_name}_{d}"
                filename = filename.replace("%", "").replace(".", "p")
                filename = filename.replace(" ", "_")
                filename += ".PNG"
                plt.savefig(
                    filename,
                    bbox_inches="tight",
                )
                plt.close()

            # * same in linear scale
            plt.figure(figsize=(8, 6))
            plt.title(title)
            plt.imshow(
                x_estimator_d_plot,
                origin="lower",
                cmap="jet",
            )
            plt.colorbar()
            self._draw_rect_on_pixels_of_interest()

            filename = f"{folder_path}/"
            filename += f"{estimator_name}_linscale_{d}"
            filename = filename.replace("%", "").replace(".", "p")
            filename = filename.replace(" ", "_")
            filename += ".PNG"

            plt.savefig(filename, bbox_inches="tight")
            plt.close()

    def plot_estimator_u(
        self,
        u_estimated: np.ndarray,
        estimator_name: str,
        folder_path: str,
        model_name: Optional[str] = "",
        list_lines: List[str] = [],
    ):
        """Only used in hierarchical models. The sampling of such model is not implemented."""
        u_estimated_plot = self.map_shaper.from_vector_to_map(u_estimated)

        for ell, line in enumerate(list_lines):
            u_estimator_ell_plot = u_estimated_plot[:, :, ell] * 1

            plt.figure(figsize=(8, 6))
            plt.title(f"{estimator_name} for {line}")
            plt.imshow(
                u_estimator_ell_plot,
                norm=colors.LogNorm(),
                origin="lower",
                cmap="jet",
            )
            plt.colorbar()
            self._draw_rect_on_pixels_of_interest()
            # plt.tight_layout()

            filename = f"{folder_path}/{estimator_name}_{ell}_{line}"
            filename = filename.replace("%", "").replace(".", "p")
            filename += ".PNG"
            plt.savefig(filename, bbox_inches="tight")
            plt.close()

            # * same in linear scale
            plt.figure(figsize=(8, 6))
            plt.title(f"{estimator_name} for {line}")
            plt.imshow(u_estimator_ell_plot, origin="lower", cmap="jet")
            plt.colorbar()

            filename = f"{folder_path}/"
            filename += f"{estimator_name}_linscale_{ell}_{line}"
            filename = filename.replace("%", "").replace(".", "p")
            filename += ".PNG"

            plt.savefig(filename, bbox_inches="tight")
            plt.close()

    def plot_CI_size(
        self,
        Theta_ci_size: np.ndarray,
        CI_name: str,
        folder_path: str,
    ) -> None:
        r"""plots the map of credibility interval sizes for a physical parameter

        Parameters
        ----------
        Theta_ci_size : np.ndarray of shape (N, D)
            vector of credibility interval sizes for the D physical parameters
        CI_name : str
            name of the credibility interval, e.g., "95\%" or "99\%"
        folder_path : str
            path to the folder where the figure is to be saved
        """
        Theta_ci_size_plot = self.map_shaper.from_vector_to_map(Theta_ci_size)
        for d, name in enumerate(self.list_names):
            if d in self.list_idx_sampling:
                x_ci_size_d_plot = Theta_ci_size_plot[:, :, d] * 1

                if self.upper_bounds_lin[-1] - self.lower_bounds_lin[-1] > 20:
                    vmin = 1
                    vmax = None
                    # vmax = self.upper_bounds_lin[d] / self.lower_bounds_lin[d]
                else:
                    vmin = 0
                    vmax = None
                    # vmax = self.upper_bounds_lin[d] - self.lower_bounds_lin[d]

                plt.figure(figsize=(8, 6))
                plt.title(f"{CI_name} for {name}")
                plt.imshow(
                    x_ci_size_d_plot,
                    norm=colors.LogNorm(vmin, vmax),
                    origin="lower",
                    cmap="jet",
                )
                plt.colorbar()
                self._draw_rect_on_pixels_of_interest()

                filename = f"{folder_path}/{CI_name}_d{d}.PNG"
                filename = filename.replace("%", "").replace(" ", "_")
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

                # * same in linear scale
                plt.figure(figsize=(8, 6))
                plt.title(f"{CI_name} for {name}")
                plt.imshow(
                    x_ci_size_d_plot,
                    origin="lower",
                    cmap="jet",
                )
                plt.colorbar()
                self._draw_rect_on_pixels_of_interest()

                filename = f"{folder_path}/{CI_name}_linscale_d{d}.PNG"
                filename = filename.replace("%", "").replace(" ", "_")
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

        return

    def plot_CI_size_u(
        self,
        u_ci_size,
        CI_name,
        folder_path: str,
        model_name: Optional[str] = "",
        list_lines: List[str] = [],
    ):
        """

        Only used in hierarchical models. The sampling of such model is not implemented.

        """

        u_ci_size_plot = self.map_shaper.from_vector_to_map(u_ci_size)
        for ell, line in enumerate(list_lines):
            u_ci_size_d_plot = u_ci_size_plot[:, :, ell] * 1

            plt.figure(figsize=(8, 6))
            plt.title(f"{CI_name} for {line}")
            plt.imshow(
                u_ci_size_d_plot,
                norm=colors.LogNorm(),
                origin="lower",
                cmap="jet",
            )
            plt.colorbar()
            self._draw_rect_on_pixels_of_interest()

            filename = f"{folder_path}/{CI_name}_{ell}.PNG"
            filename = filename.replace("%", "")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()

            # * same in linear scale
            plt.figure(figsize=(8, 6))
            plt.title(f"{CI_name} for {line}")
            plt.imshow(
                u_ci_size_d_plot,
                origin="lower",
                cmap="jet",
            )
            plt.colorbar()
            self._draw_rect_on_pixels_of_interest()

            filename = f"{folder_path}/{CI_name}_linscale_{ell}.PNG"
            filename = filename.replace("%", "")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
