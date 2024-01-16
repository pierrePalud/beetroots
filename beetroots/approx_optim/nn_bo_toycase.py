import os
import sys

import numpy as np
import yaml

from beetroots.approx_optim.nn_bo import ApproxParamsOptimNNBO

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide the name of the YAML file as an argument.")
        sys.exit(1)

    path_yaml_file = sys.argv[1]
    path_data_cloud, yaml_file = os.path.split(path_yaml_file)

    # step 1: read ``.yaml`` input file
    with open(os.path.abspath(f"{path_data_cloud}/{yaml_file}")) as f:
        params = yaml.safe_load(f)

    # step 2: run the Bayesian optimization
    approx_optim = ApproxParamsOptimNNBO(
        params["list_lines"],
        sigma_a=params["sigma_a"],
        sigma_m=np.log(params["sigma_m_float_linscale"]),
        **params["simu_init"],
    )
    approx_optim.main(**params["main_params"])
