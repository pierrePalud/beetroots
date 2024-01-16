import os
import sys
from typing import List

import numpy as np
import yaml

from beetroots.approx_optim.nn_bo import ApproxParamsOptimNNBO
from beetroots.simulations.astro.observation.abstract_real_data import (
    SimulationRealData,
)


class ReadDataRealData(SimulationRealData):
    r"""implements an ``__init__`` method for :class:`.SimulationRealData`"""

    def __init__(self, list_lines: List[str]) -> None:
        """

        Parameters
        ----------
        list_lines : List[str]
            observables for which the likelihood approximation parameters are to be adjusted
        """
        self.list_lines_fit = list_lines


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the name of the YAML file as an argument.")
        sys.exit(1)

    path_yaml_file = sys.argv[1]
    path_data_cloud, yaml_file = os.path.split(path_yaml_file)

    # step 1: read ``.yaml`` input file
    with open(os.path.abspath(f"{path_data_cloud}/{yaml_file}")) as f:
        params = yaml.safe_load(f)

    # step 2: read the additive noise standard deviations for the specified
    # lines
    data_reader = ReadDataRealData(params["list_lines"])
    (_, _, sigma_a, _, _, _, _) = data_reader.setup_observation(
        data_int_path=f"{path_data_cloud}/{params['filename_int']}",
        data_err_path=f"{path_data_cloud}/{params['filename_err']}",
        save_obs=False,
    )

    # step 3: run the Bayesian optimization
    approx_optim = ApproxParamsOptimNNBO(
        list_lines=params["list_lines"],
        sigma_a=sigma_a,
        sigma_m=np.log(params["sigma_m_float_linscale"]),
        **params["simu_init"],
    )
    approx_optim.main(**params["main_params"])
