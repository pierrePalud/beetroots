from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.simulations.astro.observation.abstract_observation import (
    SimulationObservation,
)


class SimulationRealData(SimulationObservation):
    r"""abstract class that reads the observation data for real observations"""

    def setup_observation(
        self,
        data_int_path: str,
        data_err_path: str,
        save_obs: bool = True,
    ) -> Tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        r"""reads the observation data for real observations

        Parameters
        ----------
        data_int_path : str
            path to the ``.pkl`` file that contains the observation maps
        data_err_path : str
            path to the ``.pkl`` file that contains the maps of additive noise standard deviation
        save_obs : bool, optional
            by default True

        Returns
        -------
        df_int_fit : pd.DataFrame
            DataFrame containing the observations used for inference
        y_fit : np.ndarray of shape (N, L)
            observations to be used for inference
        sigma_a_fit : np.ndarray of shape (N, L)
            additive noise standard deviations associated with the observations used for inference
        omega_fit : np.ndarray of shape (N, L)
            censor threshold associated with the observations used for inference
        y_valid : np.ndarray of shape (N, L_valid)
            observations that are not to be used for inference
        sigma_a_valid : np.ndarray of shape (N, L_valid)
            additive noise standard deviations associated with the observations not used for inference
        omega_valid : np.ndarray of shape (N, L_valid)
            censor threshold associated with the observations not used for inference
        """
        # * read observations
        df_int: pd.DataFrame = pd.read_pickle(data_int_path)
        df_err: pd.DataFrame = pd.read_pickle(data_err_path)
        assert list(df_int.index.names) == ["X", "Y"]
        assert list(df_err.index.names) == ["X", "Y"]
        assert len(df_int) == len(df_err)

        df_int = df_int.sort_index()
        df_err = df_err.sort_index()

        if "idx" not in list(df_err.columns):
            df_int["idx"] = np.arange(len(df_int))
            df_err["idx"] = np.arange(len(df_err))

        assert df_int.shape == df_err.shape
        assert list(df_int.columns) == list(df_err.columns)

        for line in self.list_lines_fit:
            assert line in list(df_int.columns)

        self.list_lines_valid = list(
            set(list(df_int.columns)) - set(self.list_lines_fit)
        )
        r"""List[str]: list of observed lines not used for inference, that can be used for validation"""
        self.list_lines_valid.remove("idx")
        self.list_lines_valid.sort()

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

        self.N = len(df_int_fit)
        r"""int: number of pixels / components in the observation"""

        # * correct values (censoring / errors / etc.)
        df_censor_fit = df_err_fit.copy()
        # df_censor.iloc[:, :-1] *= 3 # with potential censorship
        df_censor_fit.iloc[:, :-1] = 1e-60  # no censorship

        df_censor_valid = df_err_valid.copy()
        # df_censor.iloc[:, :-1] *= 3 # with potential censorship
        df_censor_valid.iloc[:, :-1] = 1e-60  # no censorship

        y_fit = np.nan_to_num(df_int_fit.drop(columns="idx").values, nan=1e-15)
        sigma_a_fit = np.nan_to_num(df_err_fit.drop(columns="idx").values, nan=1)
        omega_fit = df_censor_fit.drop(columns="idx").values

        y_valid = np.nan_to_num(df_int_valid.drop(columns="idx").values, nan=1e-15)
        sigma_a_valid = np.nan_to_num(
            df_err_valid.drop(columns="idx").values,
            nan=1,
        )
        omega_valid = df_censor_valid.drop(columns="idx").values

        self.Theta_true_scaled = None
        r"""Optional[np.ndarray]: true values of the physical parameters. Always ``None`` for real applications."""
        self.map_shaper = MapShaper(df_int)
        r"""MapShaper: defines the transformation from vectors to 2D maps"""

        # * save observation
        if save_obs:
            df_int_fit.to_pickle(
                f"{self.path_data_csv_in}/observation_maps.pkl",
            )
            df_err_fit.to_pickle(f"{self.path_data_csv_in}/additive_std.pkl")
            df_censor_fit.to_pickle(
                f"{self.path_data_csv_in}/censor_threshold.pkl",
            )

        return (
            df_int_fit,
            y_fit,
            sigma_a_fit,
            omega_fit,
            y_valid,
            sigma_a_valid,
            omega_valid,
        )
