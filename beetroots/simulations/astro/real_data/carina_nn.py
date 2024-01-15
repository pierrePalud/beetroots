import os
from typing import Tuple

import numpy as np
import pandas as pd

from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.simulations.astro import data_validation
from beetroots.simulations.astro.real_data.real_data_nn import SimulationRealDataNN


def read_point_challenger(
    path_data_cloud: str,
    filename: str,
    point_name: str,
) -> dict:
    df_pt_challenger = pd.read_pickle(f"{path_data_cloud}/{filename}")
    point_challenger = {"name": point_name, "value": df_pt_challenger.values}
    return point_challenger


def apply_data_transformation_3cases_rule(
    df_int: pd.DataFrame,
    df_err: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_int_inter = df_int.drop("idx", 1)
    df_err_inter = df_err.drop("idx", 1)
    df_censor_inter = df_err_inter.copy()

    # case 1 : sigma = + infty : omega = u and sigma = 1/3 * u
    mask = np.isinf(df_err)
    df_err_inter[mask] = 1 / 3 * df_int[mask]
    df_censor_inter[mask] = 1 * df_int[mask]

    # case 2 : u < sigma < + infty : omega = 3 * sigma and u = omega
    mask = (df_int < df_err) & ~np.isinf(df_err) & ~np.isnan(df_err)
    df_censor_inter[mask] = 3 * df_err[mask]
    df_int_inter[mask] = 1 * df_censor_inter[mask]

    # case 3 : sigma < u < + infty : omega = sigma
    mask = (df_err < df_int) & ~np.isinf(df_err) & ~np.isnan(df_err)
    df_censor_inter[mask] = 1 * df_err[mask]

    return df_int_inter, df_err_inter, df_censor_inter


def apply_data_transformation_3cases_rule_alternative(
    df_int: pd.DataFrame,
    df_err: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_int_inter = df_int.drop("idx", 1)
    df_err_inter = df_err.drop("idx", 1)
    df_censor_inter = df_err_inter.copy()

    # first : never consider observations to be censored
    df_censor_inter = 1e-60

    # if sigma = + infty : put very high noise std
    mask = np.isinf(df_err)
    df_err_inter[mask] = 1.0

    return df_int_inter, df_err_inter, df_censor_inter


class SimulationAstroCarina(SimulationRealDataNN):
    def setup_observation(
        self,
        data_int_path: str,
        data_err_path: str,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        df_int = pd.read_pickle(data_int_path)
        df_err = pd.read_pickle(data_err_path)
        assert list(df_int.index.names) == ["X", "Y"]
        assert list(df_err.index.names) == ["X", "Y"]
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

        self.N = len(df_int)

        # * correct values (censoring / errors / etc.)
        df_censor_fit = df_err_fit.copy()
        df_censor_fit.loc[:, self.list_lines_fit] *= 3  # omega = 3 sigma_a

        df_censor_valid = df_err_valid.copy()
        df_censor_valid.loc[:, self.list_lines_valid] *= 3  # omega = 3 sigma_a

        # * apply data transformation 3-cases rule
        (
            df_int_fit_inter,
            df_err_fit_inter,
            df_censor_fit_inter,
        ) = apply_data_transformation_3cases_rule_alternative(
            df_int_fit,
            df_err_fit,
        )
        (
            df_int_valid_inter,
            df_err_valid_inter,
            df_censor_valid_inter,
        ) = apply_data_transformation_3cases_rule_alternative(
            df_int_valid,
            df_err_valid,
        )

        df_int_fit.loc[:, self.list_lines_fit] = df_int_fit_inter
        df_err_fit.loc[:, self.list_lines_fit] = df_err_fit_inter
        df_censor_fit.loc[:, self.list_lines_fit] = df_censor_fit_inter

        df_int_valid.loc[:, self.list_lines_valid] = df_int_valid_inter
        df_err_valid.loc[:, self.list_lines_valid] = df_err_valid_inter
        df_censor_valid.loc[:, self.list_lines_valid] = df_censor_valid_inter

        # *
        y_fit = np.nan_to_num(df_int_fit.drop("idx", 1).values)
        sigma_a_fit = np.nan_to_num(df_err_fit.drop("idx", 1).values, nan=1)
        omega_fit = df_censor_fit.drop("idx", 1).values

        y_valid = np.nan_to_num(df_int_valid.drop("idx", 1).values)
        sigma_a_valid = np.nan_to_num(df_err_valid.drop("idx", 1).values, nan=1)
        omega_valid = df_censor_valid.drop("idx", 1).values

        self.X_true_scaled = None
        self.map_shaper = MapShaper(df_int)

        # * save observation
        df_int_fit.to_pickle(f"{self.path_data_csv_in}/observation_maps.pkl")
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


if __name__ == "__main__":
    # * setup path to data
    path_data_cloud = f"{os.path.dirname(os.path.abspath(__file__))}"
    path_data_cloud += "/../../../../../data/carina"

    # * import Wu et al. 2018 values
    point_challenger = read_point_challenger(
        path_data_cloud,
        filename="values_wu_et_al_2018.pkl",
        point_name="Wu et al., 2018",
    )

    params = SimulationRealDataNN.load_params(path_data_cloud)
    SimulationRealDataNN.check_input_params_file(
        params,
        data_validation.schema,
    )

    simulation = SimulationRealDataNN(**params["simu_init"], params=params)

    simulation.main(
        params=params,
        path_data_cloud=path_data_cloud,
        point_challenger=point_challenger,
    )
