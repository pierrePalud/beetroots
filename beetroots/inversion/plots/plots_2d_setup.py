import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from beetroots.inversion.plots import readable_line_names
from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.inversion.plots.plots_2d import AbstractPlots2D


class Plots2DSetup(AbstractPlots2D):
    r"""utilitary class that draws and saves maps related to the inference setup."""

    __slots__ = (
        "path_img",
        "map_shaper",
        "N",
        "pixels_of_interest_names",
        "pixels_of_interest_coords",
    )

    def __init__(
        self,
        path_img: str,
        map_shaper: MapShaper,
        N: int,
        pixels_of_interest: Dict[int, str] = {},
    ):
        self.path_img = path_img
        r"""str: path to the folder in which the figures are to be saved"""

        self.map_shaper = map_shaper
        r"""MapShaper: defines the transformation from vectors to 2D maps"""

        self.N = N
        r"""int: number of pixels in the maps, i.e., dimension of the observation vectors"""

        self.pixels_of_interest_names = pixels_of_interest
        r"""dict: (coordinate, name) pair of some user-informed pixels to be highlighted. These pixels will be outlines with a black square in figures."""

        coords = map_shaper.from_vector_idx_to_map_coords(
            list(pixels_of_interest.keys())
        )
        self.pixels_of_interest_coords = coords
        r"""list: coordinates of some user-informed pixels to be highlighted. These pixels will be outlines with a black square in figures."""

    def plot_indices_map(self):
        r"""plots and saved a map of the indices. Simplifies the choice of pixels of interest."""
        Theta_idx = np.arange(self.N)
        Theta_idx_shaped = self.map_shaper.from_vector_to_map(Theta_idx)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("indices map")
        plt.imshow(
            Theta_idx_shaped,
            origin="lower",
            cmap="jet",
        )

        # Loop over data dimensions and create text annotations.
        font_size = 100 / Theta_idx_shaped.shape[0]
        for i in range(Theta_idx_shaped.shape[0]):
            for j in range(Theta_idx_shaped.shape[1]):
                if not np.isnan(Theta_idx_shaped[i, j]):
                    pix_idx_val = int(Theta_idx_shaped[i, j])
                    _ = ax.text(
                        j,
                        i,
                        f"{pix_idx_val}",
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=font_size,
                    )

        self._draw_rect_on_pixels_of_interest()

        # plt.xticks([]) # one needs the ticks to identify pixel positions
        # plt.yticks([])
        plt.grid()
        plt.savefig(
            f"{self.path_img}/indices_map.PNG",
            bbox_inches="tight",
            dpi=1200,
        )
        plt.close()

    def plot_sites_map(self, dict_sites_: dict[int, np.ndarray]) -> None:
        """plots the map of sites defined from the spatial prior. The pixels within one site are all sampled in parallel in the MTM-chromatic Gibbs sampling kernel (see ``beetroots.sampler.my_sampler.MySampler``)

        Parameters
        ----------
        dict_sites_ : dict[int, np.ndarray]
            dictionary of sites (see spatial prior modules).
        """
        Theta_sites = np.zeros((self.N,))
        for key, idx in dict_sites_.items():
            Theta_sites[idx] = key * 1
        Theta_sites_shaped = self.map_shaper.from_vector_to_map(Theta_sites)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("Sites map")
        plt.imshow(Theta_sites_shaped, origin="lower")
        self._draw_rect_on_pixels_of_interest()

        plt.xticks([])
        plt.yticks([])
        plt.savefig(
            f"{self.path_img}/sites_map.PNG",
            bbox_inches="tight",
        )
        plt.close()

    def plot_censored_lines_proportion(
        self,
        y: np.ndarray,
        omega: np.ndarray,
        folder_path: str,
    ):
        r"""plots the map of the proportion of censored observables per pixel. Only relevant for likelihood models involving censorship.

        Parameters
        ----------
        y : np.ndarray of shape (N, L)
            observation vector of :math:`N` pixels and :math:`L` lines.
        omega : np.ndarray of shape (N, L)
            vector of censorship limits.
        folder_path : str
            path to the folder where the figure is to be saved
        """
        prop_censor = (y <= omega).mean(1)  # float between [0,1]
        prop_censor_shaped = self.map_shaper.from_vector_to_map(prop_censor)

        plt.figure(figsize=(8, 6))
        plt.title("Proportion of censored lines")
        plt.imshow(prop_censor_shaped, origin="lower", cmap="jet")
        plt.colorbar()
        self._draw_rect_on_pixels_of_interest()

        # plt.tight_layout()
        plt.savefig(
            f"{folder_path}/proportion_censored_lines.PNG",
            bbox_inches="tight",
        )
        plt.close()

    def plot_observations(
        self,
        y: np.ndarray,
        list_lines: list,
        folder_path: str,
    ) -> None:
        """list_lines = self.list_lines_fit + self.list_lines_eval"""
        y_shaped = self.map_shaper.from_vector_to_map(y)
        for ell, name in enumerate(list_lines):
            y_ell_shaped = y_shaped[:, :, ell] * 1
            y_ell_shaped[y_ell_shaped >= 0.9] = np.nan
            y_ell_shaped[y_ell_shaped < 1e-14] = np.nan

            readable_name = readable_line_names.lines_to_latex(name)

            plt.figure(figsize=(8, 6))
            plt.title(f"observation of line {readable_name}")
            plt.imshow(
                y_ell_shaped,
                origin="lower",
                norm=colors.LogNorm(),
                cmap="jet",
            )
            plt.colorbar()
            self._draw_rect_on_pixels_of_interest()

            # plt.tight_layout()
            plt.savefig(
                f"{folder_path}/observation_line_{ell}_{name}.PNG",
                bbox_inches="tight",
            )
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.title(f"observation of line {readable_name}")
            plt.imshow(
                y_ell_shaped,
                origin="lower",
                cmap="jet",
            )
            plt.colorbar()
            self._draw_rect_on_pixels_of_interest()

            # plt.tight_layout()
            plt.savefig(
                f"{folder_path}/observation_line_{ell}_{name}_linscale.PNG",
                bbox_inches="tight",
            )
            plt.close()

    def plot_mask_censored_pixels(
        self,
        y: np.ndarray,
        omega: np.ndarray,
        list_lines: list,
        folder_path: str,
    ) -> None:
        """plots and saves the map of censored pixels for each line

        Parameters
        ----------
        y : np.ndarray of shape (N, L)
            observation vector of :math:`N` pixels and :math:`L` lines.
        omega : np.ndarray of shape (N, L)
            vector of censorship limits.
        list_lines : list
            list of the names of the observed lines
        folder_path : str
            path to the folder where the figure is to be saved
        """
        censored_shaped = self.map_shaper.from_vector_to_map(y <= omega)
        for ell, name in enumerate(list_lines):
            censored_ell_shaped = censored_shaped[:, :, ell] * 1

            readable_name = readable_line_names.lines_to_latex(name)

            plt.figure(figsize=(8, 6))
            plt.title(f"mask of censored pixels for line {readable_name}")
            plt.imshow(censored_ell_shaped, origin="lower", cmap="jet")
            plt.colorbar()
            self._draw_rect_on_pixels_of_interest()

            # plt.tight_layout()
            plt.savefig(
                f"{folder_path}/censored_map_line_{ell}_{name}.PNG",
                bbox_inches="tight",
            )
            plt.close()

    def plot_sigma_a(
        self,
        sigma_a: np.ndarray,
        list_lines: list,
        folder_path: str,
    ) -> None:
        r"""plots and saves the maps of standard deviations for each of the :math:`L` observables.

        Parameters
        ----------
        sigma_a : np.ndarray of shape (N, L)
            vector containing the additive noise standard deviations of each of the :math:`N` pixels and :math:`L` lines.
        list_lines : list
            list of the names of the observed lines
        folder_path : str
            path to the folder where the figures are to be saved
        """
        sigma_a_shaped = self.map_shaper.from_vector_to_map(sigma_a)
        for ell, name in enumerate(list_lines):
            sigma_a_ell_shaped = sigma_a_shaped[:, :, ell] * 1
            sigma_a_ell_shaped[sigma_a_ell_shaped >= 0.9] = np.nan

            readable_name = readable_line_names.lines_to_latex(name)

            plt.figure(figsize=(8, 6))
            plt.title(f"standard deviation for line {readable_name}")
            plt.imshow(
                sigma_a_ell_shaped,
                origin="lower",
                norm=colors.LogNorm(),
                cmap="jet",
            )
            plt.colorbar()
            self._draw_rect_on_pixels_of_interest()

            # plt.tight_layout()
            plt.savefig(
                f"{folder_path}/add_err_std_line_{ell}_{name}.PNG",
                bbox_inches="tight",
            )
            plt.close()

    def plot_snr_add(
        self,
        y: np.ndarray,
        sigma_a: np.ndarray,
        list_lines: list,
        folder_path: str,
    ) -> None:
        """plots and saves the maps of signal-to-noise ratio (SNR) for each of the :math:`L` observables. For one pixel :math:`n` and one line :math:`\ell`, the SNR is defined as :math:`y_{n\ell} / \sigma_{a,n\ell}` with :math:`y_{n\ell}` the observed value and :math:`\sigma_{a,n\ell}` the additive noise standard deviation.

        Parameters
        ----------
        y : np.ndarray of shape (N, L)
            observation vector of :math:`N` pixels and :math:`L` lines.
        sigma_a : np.ndarray of shape (N, L)
            vector containing the additive noise standard deviations of each of the :math:`N` pixels and :math:`L` lines.
        list_lines : list
            list of the names of the observed lines
        folder_path : str
            path to the folder where the figures are to be saved
        """
        snr_a_shaped = self.map_shaper.from_vector_to_map(y / sigma_a)
        for ell, name in enumerate(list_lines):
            snr_a_ell_shaped = snr_a_shaped[:, :, ell] * 1

            readable_name = readable_line_names.lines_to_latex(name)

            plt.figure(figsize=(8, 6))
            plt.title(r"y / $\sigma_a$" + f" of line {readable_name}")
            plt.imshow(
                snr_a_ell_shaped,
                origin="lower",
                norm=colors.LogNorm(vmin=1.0),
                cmap="jet",
            )
            plt.colorbar()
            self._draw_rect_on_pixels_of_interest()

            # plt.tight_layout()
            plt.savefig(
                f"{folder_path}/snr_line_{ell}_{name}.PNG",
                bbox_inches="tight",
            )
            plt.close()

    def plot_all(
        self,
        y: np.ndarray,
        sigma_a: np.ndarray,
        omega: np.ndarray,
        list_lines: list,
        dict_sites_: dict[int, np.ndarray],
    ) -> None:
        r"""runs all the class methods, i.e., plots and saves the maps of the observations :math:`y`, of the additive noise standard deviation :math:`\sigma_a`, of the signal-to-noise ratio :math:`y : \sigma_a`, of censored pixels.
        It also plots and saves the map of proportion of censored observables per pixel, the map of indices and of spatial prior sites.

        Parameters
        ----------
        y : np.ndarray of shape (N, L)
            observation vector of :math:`N` pixels and :math:`L` lines.
        sigma_a : np.ndarray of shape (N, L)
            vector containing the additive noise standard deviations of each of the :math:`N` pixels and :math:`L` lines.
        omega : np.ndarray of shape (N, L)
            vector of censorship limits.
        list_lines : list
            list of the names of the observed lines
        dict_sites_ : dict[int, np.ndarray]
            dictionary of sites (see spatial prior modules).
        """
        folder_path = f"{self.path_img}/observations"
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        self.plot_observations(y, list_lines, folder_path)
        self.plot_sigma_a(sigma_a, list_lines, folder_path)
        self.plot_snr_add(y, sigma_a, list_lines, folder_path)
        self.plot_mask_censored_pixels(y, omega, list_lines, folder_path)
        self.plot_censored_lines_proportion(y, omega, folder_path)

        self.plot_indices_map()
        self.plot_sites_map(dict_sites_)
