import os

import numpy as np

from beetroots.simulations.astro import data_validation

# from beetroots.simulations.astro import utils
from beetroots.simulations.astro.real_data.real_data_nn import SimulationRealDataNN

if __name__ == "__main__":
    path_data_cloud = os.path.abspath(
        f"{os.path.dirname(os.path.abspath(__file__))}/../../data/ngc7023"
    )

    # note : G0 (front of cloud) = 1.2786 * radm / 2
    G0_joblin = 2.6e3
    radm_joblin = 2 * G0_joblin / 1.2786

    point_challenger = {
        "name": "Joblin et al., 2018",
        "value": np.array([[0.7, 1e8, radm_joblin, 1e1, 0.0]]),
    }

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
