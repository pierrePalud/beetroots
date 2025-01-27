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
    yaml_file, path_data, path_models, path_outputs = ApproxParamsOptimNNBO.parse_args()

    # step 1: read ``.yaml`` input file
    with open(os.path.abspath(f"{path_data}/{yaml_file}")) as f:
        params = yaml.safe_load(f)

    # step 2: read the additive noise standard deviations for the specified
    # lines
    data_reader = ReadDataRealData(params["list_lines"])

    # giving a path to an observations file is required for code compatibility,
    # but one can give anything since the observations are not used.
    # So we give the error file again.
    (_, _, sigma_a_raw, _, _, _, _) = data_reader.setup_observation(
        data_int_path=f"{path_data}/{params['filename_err']}",
        data_err_path=f"{path_data}/{params['filename_err']}",
        save_obs=False,
    )

    # step 3: run the Bayesian optimization
    approx_optim = ApproxParamsOptimNNBO(
        list_lines=params["list_lines"],
        sigma_a_raw=sigma_a_raw,
        sigma_m=np.log(params["sigma_m_float_linscale"]),  # float
        **params["simu_init"],
        path_outputs=path_outputs,
        path_models=path_models,
    )

    approx_optim.main(**params["main_params"])

    approx_optim.save_results_in_data_folder(path_data, params["filename_err"])
