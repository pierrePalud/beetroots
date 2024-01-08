import os

# from beetroots.simulations.astro import utils
from beetroots.simulations.astro.real_data.omc1_nn_direct_posterior import (
    SimulationAstroOMC1,
)

if __name__ == "__main__":
    # * setup path to data
    path_data_cloud = f"{os.path.dirname(os.path.abspath(__file__))}"
    path_data_cloud += "/../../../../../data/omc1-orionbar"

    params = SimulationAstroOMC1.load_params(path_data_cloud)

    simulation = SimulationAstroOMC1(**params["simu_init"], params=params)

    simulation.main(params=params, path_data_cloud=path_data_cloud)
