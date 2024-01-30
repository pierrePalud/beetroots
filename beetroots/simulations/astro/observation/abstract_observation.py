import abc
import os

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
        dict_posteriors: dict,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
        scaler: Scaler,
    ):
        for model_name in list(dict_posteriors.keys()):
            folder_path = f"{self.path_raw}/{model_name}"
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)

        if self.N > 1:
            key = list(dict_posteriors.keys())[0]
            if isinstance(dict_posteriors[key], list):
                dict_sites_ = dict_posteriors[key][1].dict_sites
                y = dict_posteriors[key][0].likelihood.y * 1
                sigma_a = dict_posteriors[key][0].likelihood.sigma * 1
                omega = dict_posteriors[key][0].likelihood.omega * 1

            else:
                dict_sites_ = dict_posteriors[key].dict_sites
                y = dict_posteriors[key].likelihood.y * 1
                sigma_a = dict_posteriors[key].likelihood.sigma_a * 1
                omega = dict_posteriors[key].likelihood.omega * 1

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

            self.plots_estimator = PlotsEstimator(
                self.map_shaper,
                self.list_names_plots,
                lower_bounds_lin,
                upper_bounds_lin,
                self.list_idx_sampling,
                self.pixels_of_interest,
            )

            if self.Theta_true_scaled is not None:
                folder_path = f"{self.path_img}/estimators"
                folder_path_inter = f"{folder_path}/true"

                for path_ in [folder_path, folder_path_inter]:
                    if not os.path.isdir(path_):
                        os.mkdir(path_)

                Theta_true_scaled_full = np.zeros((self.N, self.D))
                Theta_true_scaled_full[
                    :, self.list_idx_sampling
                ] = self.Theta_true_scaled

                Theta_true_lin = scaler.from_scaled_to_lin(Theta_true_scaled_full)

                for d in range(self.D):
                    if self.list_fixed_values[d] is not None:
                        Theta_true_scaled_full[:, d] += self.list_fixed_values[d]

                self.plots_estimator.plot_estimator(
                    Theta_true_lin,
                    "true",
                    folder_path_inter,
                )

        else:
            self.plots_estimator = None
        return
