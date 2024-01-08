from typing import Optional, Tuple

import numpy as np
import pandas as pd

from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams

# def create_prior_params(
#     with_spatial_prior: bool,
#     initial_regu_weights: Optional[np.ndarray] = None,
# ) -> Tuple[SpatialPriorParams, str]:
#     # spatial prior
#     if with_spatial_prior:
#         spatial_prior_params = SpatialPriorParams(
#             name="L2-laplacian",
#             use_next_nearest_neighbours=False,
#             initial_regu_weights=initial_regu_weights,
#             use_clustering=False,
#             n_clusters=None,  # None,
#             cluster_algo=None,  # None,
#         )
#         with_spatial_prior_str = "with_spatial_regu"
#     else:
#         spatial_prior_params = None
#         with_spatial_prior_str = "no_spatial_regu"

#     # indicator prior
#     indicator_margin_scale = 1e-1
#     lower_bounds_lin = np.array([1e-1, 1e5, 1e0, 1e0])
#     upper_bounds_lin = np.array([1e1, 1e9, 1e5, 4e1])

#     return (
#         spatial_prior_params,
#         with_spatial_prior_str,
#         indicator_margin_scale,
#         lower_bounds_lin,
#         upper_bounds_lin,
#     )


def choose_forward_model_version(
    grid_version: bool, angle: Optional[float]
) -> Tuple[str, str]:
    if grid_version == "1p7":
        assert angle is not None
        forward_model_name = "meudon_pdr_model_dense"
    elif grid_version == "1p5p4_PAH":
        assert angle is None
        angle = 30.0  # actually sets to 0 (due to explicit normalization)
        forward_model_name = "meudon_pdr_154PAH_model_dense"
    else:
        raise ValueError("grid_version should be either 1p7 or 1p5p4_PAH")

    return forward_model_name, angle


def read_point_challenger(
    path_data_cloud: str,
    filename: str,
    point_name: str,
) -> dict:
    df_pt_challenger = pd.read_pickle(f"{path_data_cloud}/{filename}")
    point_challenger = {"name": point_name, "value": df_pt_challenger.values}
    return point_challenger
