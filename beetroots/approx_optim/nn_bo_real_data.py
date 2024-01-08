import os
import sys

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
    if len(sys.argv) < 2:
        print("Please provide the name of the YAML file as an argument.")
        sys.exit(1)

    path_yaml_file = sys.argv[1]

    path_data_cloud, yaml_file = os.path.split(path_yaml_file)

    with open(os.path.abspath(f"{path_data_cloud}/{yaml_file}")) as f:
        params = yaml.safe_load(f)

    # for orionbar, the additive noise variance changes from line to line
    (
        _,  # df_int_fit
        _,  # y_fit
        sigma_a,  # _fit
        _,  # omega_fit
        _,  # y_valid
        _,  # sigma_a_valid
        _,  # omega_valid
    ) = ReadDataRealData(params["list_lines"]).setup_observation(
        data_int_path=f"{path_data_cloud}/{params['filename_int']}",
        data_err_path=f"{path_data_cloud}/{params['filename_err']}",
        save_obs=False,
    )

    approx_optim = ApproxParamsOptimNNBO(
        list_lines=params["list_lines"],
        sigma_a=sigma_a,
        sigma_m=np.log(params["sigma_m_float_linscale"]),
        **params["simu_init"],
    )
    approx_optim.main(**params["main_params"])
