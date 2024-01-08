import os

import numpy as np

from beetroots.simulations.astro.real_data.real_data_nn_direct_posterior import (
    SimulationRealDataNNDirectPosterior,
)

if __name__ == "__main__":
    path_data_cloud = f"{os.path.dirname(os.path.abspath(__file__))}"
    path_data_cloud += "/../../../../../data/orionbar"

    # note : G0 (front of cloud) = 1.2786 * radm / 2
    G0_joblin = 2e4
    radm_joblin = 2 * G0_joblin / 1.2786

    point_challenger = {
        "name": "Joblin, 2018",
        "value": np.array([[1.3, 2.8e8, radm_joblin, 1e1, 60.0]]),
    }

    params = SimulationRealDataNNDirectPosterior.load_params(path_data_cloud)

    simulation = SimulationRealDataNNDirectPosterior(
        **params["simu_init"], params=params
    )

    simulation.main(
        params=params,
        path_data_cloud=path_data_cloud,
        point_challenger=point_challenger,
    )
