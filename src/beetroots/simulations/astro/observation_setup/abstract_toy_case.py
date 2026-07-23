import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.modelling.forward_maps.abstract_base import ForwardMap
from beetroots.simulations.astro.observation_setup.abstract_observation import (
    SimulationObservation,
)
from beetroots.space_transform.transform import MyScaler


class SimulationToyCase(SimulationObservation):
    def _load_csv_array(self, path_csv: str, key: str) -> np.ndarray:
        df_var_loaded = pd.read_csv(path_csv)
        df_var_loaded = df_var_loaded.sort_values(["n", "d"])
        if key not in df_var_loaded.columns:
            raise ValueError(
                f"Expected column '{key}' in the CSV file, got: {df_var_loaded.columns}"
            )
        N_ = df_var_loaded["n"].max() + 1
        D_ = df_var_loaded["d"].max() + 1
        if N_ != self.N:
            raise ValueError(
                f"Loaded {key} has shape {N_} pixels, but expected {self.N}."
            )
        if key == "Theta" and D_ != self.D_sampling:
            raise ValueError(
                f"Loaded {key} has shape ({N_}, {D_}), but expected ({self.N}, {self.D_sampling})."
            )
        if key == "Aux" and D_ != self.L:
            raise ValueError(
                f"Loaded {key} has shape ({N_}, {D_}), but expected ({self.N}, {self.L})."
            )
        var_loaded = df_var_loaded[key].values
        return np.asarray(var_loaded).reshape((self.N, D_))

    def setup_observation(
        self,
        scaler: MyScaler,
        forward_map: ForwardMap,
        sigma_a: np.ndarray,
        sigma_m: np.ndarray,
        omega: np.ndarray,
        return_true_scaled_params: bool = False,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        data_int_path = f"{self.DATA_PATH}/{self.cloud_name}"

        assert os.path.isfile(f"{data_int_path}.pkl") or os.path.isfile(
            f"{data_int_path}.csv"
        ), f"There is no csv or pkl file called {self.DATA_PATH}/{self.cloud_name}."

        if os.path.isfile(f"{data_int_path}.pkl"):
            syn_map = pd.read_pickle(f"{data_int_path}.pkl")
        else:
            syn_map = pd.read_csv(f"{data_int_path}.csv")
            warnings.warn(
                "For faster loading, consider using the `.pkl` format instead `.csv`."
            )

        for col in ["idx", "X", "Y"] + self.list_names:
            assert col in list(syn_map.columns), f"{col} not in {list(syn_map.columns)}"

        syn_map = syn_map.sort_values("idx")
        syn_map = syn_map.set_index(["X", "Y"])

        syn_map.loc[:, self.list_names] = scaler.from_lin_to_scaled(
            syn_map.loc[:, self.list_names].values
        )

        # * generate observations for specified lines
        self.Theta_true_scaled = syn_map.loc[:, self.list_names].values
        self.Theta_true_scaled = self.Theta_true_scaled[
            :, self.list_idx_sampling_param_interest
        ]
        self.Theta_true_scaled_scaler = scaler

        syn_map.loc[:, self.list_lines_fit] = forward_map.evaluate(
            self.Theta_true_scaled
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
        mult_f_Theta = eps_m * f_Theta
        y = np.maximum(omega, mult_f_Theta + eps_a)
        censored = y == omega

        # Create a new DataFrame with "X", "Y", "idx" and columns from mult_f_Theta
        df_mult_f_Theta = syn_map_to_csv.reset_index()[["X", "Y", "idx"]].copy()
        for i in range(mult_f_Theta.shape[1]):
            df_mult_f_Theta[f"mult_f_Theta_l{i}"] = mult_f_Theta[:, i]
        df_mult_f_Theta.to_csv(
            f"{self.path_data_csv_in}/mult_f_Theta_maps.csv", index=False
        )

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

        if return_true_scaled_params:
            return (
                syn_map,
                y,
                mult_f_Theta,
                self.Theta_true_scaled,
                censored,
                forward_map,
            )
        else:
            return syn_map, y, mult_f_Theta
