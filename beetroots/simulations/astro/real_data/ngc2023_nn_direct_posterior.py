import os

from beetroots.simulations.astro.real_data.real_data_nn_direct_posterior import (
    SimulationRealDataNNDirectPosterior,
)

if __name__ == "__main__":
    path_data_cloud = f"{os.path.dirname(os.path.abspath(__file__))}"
    path_data_cloud += "/../../../../../data/ngc2023"

    params = SimulationRealDataNNDirectPosterior.load_params(path_data_cloud)

    simulation = SimulationRealDataNNDirectPosterior(**params["simu_init"])

    simulation.main(params=params, path_data_cloud=path_data_cloud)
