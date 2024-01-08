import os
from typing import Tuple

import numpy as np
import pandas as pd

from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.sampler.my_sampler import MySampler
from beetroots.sampler.utils.psgldparams import PSGLDParams
from beetroots.simulations.astro.real_data.real_data_nn_direct_posterior import (
    SimulationRealDataNNDirectPosterior,
)


class SimulationAstroHorsehead(SimulationRealDataNNDirectPosterior):
    def setup_observation(
        self,
        data_int_path: str,
        data_err_path: str,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:

        # * read observations
        df_int = pd.read_pickle(data_int_path)
        df_err = pd.read_pickle(data_err_path)
        assert list(df_int.index.names) == ["X", "Y"]
        assert list(df_err.index.names) == ["X", "Y"]
        df_int = df_int.sort_index()
        df_err = df_err.sort_index()

        df_int = df_int.reindex(sorted(df_int.columns), axis=1)
        df_err = df_err.reindex(sorted(df_err.columns), axis=1)

        self.N = len(df_int)

        if "idx" not in list(df_err.columns):
            df_int["idx"] = np.arange(self.N)
            df_err["idx"] = np.arange(self.N)

        assert df_int.shape == df_err.shape
        assert list(df_int.columns) == list(df_err.columns)

        for line in self.list_lines_fit:
            assert line in list(df_int.columns)

        self.list_lines_valid = list(
            set(list(df_int.columns)) - set(self.list_lines_fit)
        )
        self.list_lines_valid.remove("idx")

        # * select lines
        df_int_fit = df_int.loc[:, self.list_lines_fit + ["idx"]]
        df_err_fit = df_err.loc[:, self.list_lines_fit + ["idx"]]

        df_int_valid = df_int.loc[:, self.list_lines_valid + ["idx"]]
        df_err_valid = df_err.loc[:, self.list_lines_valid + ["idx"]]

        # eliminate absurd values
        for col in self.list_lines_fit:
            df_int_fit[col] = df_int_fit[col].apply(
                lambda x: x if x <= 1e3 else np.inf,
            )
            df_err_fit[col] = df_err_fit[col].apply(
                lambda x: x if x <= 1e3 else np.inf,
            )
        for col in self.list_lines_valid:
            df_int_valid[col] = df_int_valid[col].apply(
                lambda x: x if x <= 1e3 else np.inf,
            )
            df_err_valid[col] = df_err_valid[col].apply(
                lambda x: x if x <= 1e3 else np.inf,
            )

        # * correct values (censoring / errors / etc.)
        df_int_fit[np.isnan(df_int_fit) | np.isinf(df_int_fit)] = 0
        df_err_fit[np.isnan(df_err_fit) | np.isinf(df_err_fit) | (df_err_fit < 0)] = 1e0

        df_int_valid[np.isnan(df_int_valid) | np.isinf(df_int_valid)] = 0
        df_err_valid[
            np.isnan(df_err_valid) | np.isinf(df_err_valid) | (df_err_valid < 0)
        ] = 1e0

        df_censor_fit = df_err_fit.copy()
        df_censor_fit.loc[:, self.list_lines_fit] = -np.infty  # no censoring

        df_censor_valid = df_err_valid.copy()
        df_censor_valid.loc[:, self.list_lines_valid] = -np.infty  # no censoring

        # *
        y_fit = np.nan_to_num(df_int_fit.drop("idx", 1).values)
        sigma_a_fit = np.nan_to_num(df_err_fit.drop("idx", 1).values, nan=1)
        omega_fit = df_censor_fit.drop("idx", 1).values

        y_valid = np.nan_to_num(df_int_valid.drop("idx", 1).values)
        sigma_a_valid = np.nan_to_num(df_err_valid.drop("idx", 1).values, nan=1)
        omega_valid = df_censor_valid.drop("idx", 1).values
        # self.sigma_a = 1e-9 * np.ones((self.N, self.L))
        # self.omega = 3 * self.sigma_a

        y_fit = np.where(y_fit < omega_fit, omega_fit, y_fit)
        y_valid = np.where(y_valid < omega_valid, omega_valid, y_valid)

        self.X_true_scaled = None
        self.map_shaper = MapShaper(df_int)

        # * save observation
        df_int_fit.to_pickle(f"{self.path_data_csv_in}/observation_maps.pkl")
        df_err_fit.to_pickle(f"{self.path_data_csv_in}/additive_std.pkl")
        df_censor_fit.to_pickle(f"{self.path_data_csv_in}/censor_threshold.pkl")

        return (
            df_int_fit,
            y_fit,
            sigma_a_fit,
            omega_fit,
            y_valid,
            sigma_a_valid,
            omega_valid,
        )


if __name__ == "__main__":
    path_data_cloud = f"{os.path.dirname(os.path.abspath(__file__))}"
    path_data_cloud += "/../../../../../data/horsehead"

    params = SimulationRealDataNNDirectPosterior.load_params(path_data_cloud)

    simulation = SimulationRealDataNNDirectPosterior(
        **params["simu_init"], params=params
    )

    simulation.main(params=params, path_data_cloud=path_data_cloud)
