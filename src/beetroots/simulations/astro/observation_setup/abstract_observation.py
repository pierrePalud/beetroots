import abc
import os
from typing import Optional, Union

import numpy as np

# from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.inversion.plots.plots_2d_setup import Plots2DSetup
from beetroots.inversion.plots.plots_estimator import PlotsEstimator
from beetroots.simulations.astro.abstract_astro_simulation import AstroSimulation
from beetroots.space_transform.abstract_transform import Scaler


class SimulationObservation(AstroSimulation, abc.ABC):
    @abc.abstractmethod
    def setup_observation(self):
        pass

    def save_and_plot_setup(
        self,
        dict_sites_: dict,
        y: np.ndarray,
        sigma_a: np.ndarray,
        omega: np.ndarray,
    ):
        if self.N > 1:

            Plots2DSetup(
                self.path_img,
                self.map_shaper,
                self.N,
                self.pixels_of_interest,
            ).plot_all(
                y,
                sigma_a,
                omega,
                self.list_lines_fit,
                dict_sites_,
            )
        return

    def save_and_plot_estimator(
        self,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
        list_idx_sampling: list,
        list_names_plots: list,
        list_fixed_values: list,
        Var_true_scaled: Optional[Union[None, np.ndarray]] = None,
        scaler: Optional[Scaler] = None,
        sub_var: Optional[str] = "",
    ):
        if self.N > 1:
            plots_estimator = PlotsEstimator(
                self.map_shaper,
                list_names_plots,
                lower_bounds_lin,
                upper_bounds_lin,
                list_idx_sampling,
                self.pixels_of_interest,
            )

            if Var_true_scaled is not None:
                assert scaler is not None
                folder_path = f"{self.path_img}/estimators"
                folder_path_inter = f"{folder_path}/true"
                folder_path_inter = os.path.join(folder_path_inter, sub_var)

                if not os.path.isdir(folder_path_inter):
                    os.makedirs(folder_path_inter)

                Var_dim = len(list_names_plots)

                Var_true_scaled_full = np.zeros((self.N, Var_dim))
                Var_true_scaled_full[:, list_idx_sampling] = Var_true_scaled

                Var_true_lin = scaler.from_scaled_to_lin(Var_true_scaled_full)

                for d in range(Var_dim):
                    if list_fixed_values[d] is not None:
                        Var_true_scaled_full[:, d] += list_fixed_values[d]

                plots_estimator.plot_estimator(
                    Var_true_lin,
                    "true",
                    folder_path_inter,
                )
        else:
            plots_estimator = None

        return plots_estimator
