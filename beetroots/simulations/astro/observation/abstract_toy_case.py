import os

import numpy as np
import pandas as pd

from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.modelling.forward_maps.neural_network_approx import NeuralNetworkApprox
from beetroots.simulations.astro.observation.abstract_observation import (
    SimulationObservation,
)
from beetroots.space_transform.transform import MyScaler


class SimulationToyCase(SimulationObservation):
    # def __init__(
    #     self,
    #     simu_name: str,
    #     cloud_name: str,
    #     N: int,
    #     dict_params_names: Dict[str, str],
    #     list_lines_fit: List[str],
    #     params: dict,
    #     max_workers: int = 10,
    #     small_size: int = 16,
    #     medium_size: int = 20,
    #     bigger_size: int = 24,
    # ):
    #     self.cloud_name = cloud_name
    #     # self.N = N
    #     # N_1_side = int(np.sqrt(self.N))

    #     self.list_lines_fit = list_lines_fit
    #     self.L = len(list_lines_fit)

    #     self.create_empty_output_folders(simu_name, params)
    #     self.setup_plot_text_sizes(small_size, medium_size, bigger_size)

    #     self.max_workers = max_workers

    #     self.list_names = list(dict_params_names.keys())
    #     self.list_names_plots = list(dict_params_names.values())
    #     # self.list_names_plots = [
    #     #     r"$\kappa$",
    #     #     r"$P_{th}$",
    #     #     r"$G_0$",
    #     #     r"$A_V^{tot}$",
    #     #     r"$\alpha$",
    #     # ]
    #     self.D = len(self.list_names)  # Number of physical parameters
    #     self.D_no_kappa = self.D - 1  # number of params that are used in nn

    def setup_observation(
        self,
        scaler: MyScaler,
        forward_map: NeuralNetworkApprox,
        sigma_a: np.ndarray,
        sigma_m: np.ndarray,
        omega: np.ndarray,
    ) -> pd.DataFrame:
        data_int_path = os.path.abspath(
            f"{os.path.dirname(os.path.abspath(__file__))}/../../../../data/toycases"
        )

        data_int_path += f"/{self.cloud_name}.pkl"
        syn_map = pd.read_pickle(data_int_path)

        for col in ["idx", "X", "Y"] + self.list_names:
            assert col in list(syn_map.columns), f"{col} not in {list(syn_map.columns)}"

        syn_map = syn_map.sort_values("idx")
        syn_map = syn_map.set_index(["X", "Y"])

        syn_map.loc[:, self.list_names] = scaler.from_lin_to_scaled(
            syn_map.loc[:, self.list_names].values
        )

        # * generate observations for specified lines
        self.Theta_true_scaled = syn_map.loc[:, self.list_names].values
        self.Theta_true_scaled = self.Theta_true_scaled[:, self.list_idx_sampling]

        syn_map.loc[:, self.list_lines_fit] = forward_map.evaluate(
            self.Theta_true_scaled,
        )

        # * save observations
        syn_map_to_csv = syn_map.copy()
        syn_map_to_csv.loc[:, self.list_names] = scaler.from_scaled_to_lin(
            syn_map_to_csv.loc[:, self.list_names].values,
        )
        syn_map_to_csv.to_csv(
            f"{self.path_data_csv_in}/true_params_and_emissions_maps.csv"
        )
        self.map_shaper = MapShaper(syn_map_to_csv)

        # * generation of noisy observation
        rng = np.random.default_rng(2023)
        eps_a = rng.normal(loc=0.0, scale=sigma_a)
        eps_m = rng.lognormal(
            mean=-(sigma_m**2) / 2,
            sigma=sigma_m,
        )
        assert eps_a.shape == (self.N, self.L), f"{(self.N, self.L)}"
        assert eps_m.shape == (self.N, self.L), f"{(self.N, self.L)}"
        f_Theta = forward_map.evaluate(self.Theta_true_scaled)
        y = np.maximum(omega, eps_m * f_Theta + eps_a)

        # rv_a = norm(loc=0, scale=sigma_a)
        # rv_m = lognorm(scale=np.exp(-(sigma_m ** 2) / 2), s=sigma_m)

        # y0 =forward_map.evaluate(self.Theta_true_scaled)

        # eps_a = rv_a.rvs(random_state=42)
        # eps_m = rv_m.rvs(random_state=42)
        # assert eps_a.shape == (self.N, self.L)
        # assert eps_m.shape == (self.N, self.L)

        # y = np.maximum(omega, eps_m * y0 + eps_a)  # (N,L)

        # * save observation
        df_observation = syn_map.copy()
        df_observation = df_observation.drop(columns=self.list_names)
        df_observation.loc[:, self.list_lines_fit] = y * 1
        df_observation.to_csv(f"{self.path_data_csv_in}/observation_maps.csv")

        return syn_map, y
