import os

import numpy as np

from beetroots.approx_optim.nn_bo import ApproxParamsOptimNNBO
from beetroots.simulations.astro.observation.abstract_real_data import (
    SimulationRealData,
)


class ReadDataRealData(SimulationRealData):
    def __init__(self, list_lines):
        """overwrite SimulationRealData __init__function"""
        self.list_lines_fit = list_lines


if __name__ == "__main__":
    path_data_cloud = f"{os.path.dirname(os.path.abspath(__file__))}"
    path_data_cloud += "/../../../data/orionbar"
    path_data_cloud = os.path.abspath(path_data_cloud)

    list_lines = [
        "co_v0_j11__v0_j10",  # 'CO_(11-10)',
        "co_v0_j12__v0_j11",  # 'CO_(12-11)',
        "co_v0_j13__v0_j12",  # 'CO_(13-12)',
        "co_v0_j14__v0_j13",  # 'CO_(14-13)',
        "co_v0_j15__v0_j14",  # 'CO_(15-14)',
        "co_v0_j16__v0_j15",  # 'CO_(16-15)',
        "co_v0_j17__v0_j16",  # 'CO_(17-16)',
        "co_v0_j18__v0_j17",  # 'CO_(18-17)',
        "co_v0_j19__v0_j18",  # 'CO_(19-18)',
        "co_v0_j20__v0_j19",  # 'CO_(20-19)',
        "co_v0_j21__v0_j20",  # 'CO_(21-20)',
        "co_v0_j23__v0_j22",  # 'CO_(23-22)',
        #
        # H2 lines
        "h2_v0_j2__v0_j0",  # 'H2_0-0_S(0)',
        "h2_v0_j3__v0_j1",  # 'H2_0-0_S(1)',
        "h2_v0_j4__v0_j2",  # 'H2_0-0_S(2)',
        "h2_v0_j5__v0_j3",  # 'H2_0-0_S(3)',
        "h2_v0_j6__v0_j4",  # 'H2_0-0_S(4)',
        "h2_v0_j7__v0_j5",  # 'H2_0-0_S(5)',
        #
        # CH+ lines
        "chp_j1__j0",  # 'CH+_(1-0)',
        "chp_j2__j1",  # 'CH+_(2-1)',
        "chp_j3__j2",  # 'CH+_(3-2)',
        "chp_j4__j3",  # 'CH+_(4-3)',
        "chp_j5__j4",  # 'CH+_(5-4)',
        "chp_j6__j5",  # 'CH+_(6-5)'
    ]

    # for orionbar, the additive noise variance changes from line to line
    (
        _,  # df_int_fit
        _,  # y_fit
        sigma_a,  # _fit
        _,  # omega_fit
        _,  # y_valid
        _,  # sigma_a_valid
        _,  # omega_valid
    ) = ReadDataRealData(list_lines).setup_observation(
        data_int_path=f"{path_data_cloud}/OrionBar_Joblin_Int.pkl",
        data_err_path=f"{path_data_cloud}/OrionBar_Joblin_Err.pkl",
        save_obs=False,
    )

    sigma_m = np.log(1.3)

    approx_optim = ApproxParamsOptimNNBO(
        list_lines,
        name="bo_nn_orionbar",
        D=4,
        D_no_kappa=3,
        K=30,
        log10_f_grid_size=100,
        N_samples_y=200_000,  # 250_000
        max_workers=20,
        sigma_a=sigma_a,
        sigma_m=sigma_m,
    )
    approx_optim.main(
        forward_model_name="meudon_pdr_model_dense",
        angle=60.0,
        lower_bounds_lin=np.array([1e-1, 1e5, 1e0, 1e0]),
        upper_bounds_lin=np.array([1e1, 1e9, 1e5, 4e1]),
        n_iter=40,
    )
