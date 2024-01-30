import os
import sys

import numpy as np
import yaml

from beetroots.approx_optim.nn_bo import ApproxParamsOptimNNBO

if __name__ == "__main__":
    yaml_file, path_data, path_models, path_outputs = ApproxParamsOptimNNBO.parse_args()

    # step 1: read ``.yaml`` input file
    with open(os.path.abspath(f"{path_data}/{yaml_file}")) as f:
        params = yaml.safe_load(f)

    # step 2: run the Bayesian optimization
    approx_optim = ApproxParamsOptimNNBO(
        params["list_lines"],
        sigma_a=params["sigma_a"],
        sigma_m=np.log(params["sigma_m_float_linscale"]),
        **params["simu_init"],
        path_models=path_models,
        path_outputs=path_outputs,
    )
    approx_optim.main(**params["main_params"])
