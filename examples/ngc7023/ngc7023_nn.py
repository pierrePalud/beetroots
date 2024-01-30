import numpy as np

from beetroots.simulations.astro import data_validation

# from beetroots.simulations.astro import utils
from beetroots.simulations.astro.real_data.real_data_nn import SimulationRealDataNN

if __name__ == "__main__":
    yaml_file, path_data, path_models, path_outputs = SimulationRealDataNN.parse_args()

    # note : G0 (front of cloud) = 1.2786 * radm / 2
    G0_joblin = 2.6e3
    radm_joblin = 2 * G0_joblin / 1.2786

    point_challenger = {
        "name": "Joblin et al., 2018",
        "value": np.array([[0.7, 1e8, radm_joblin, 1e1, 0.0]]),
    }

    params = SimulationRealDataNN.load_params(path_data, yaml_file)

    SimulationRealDataNN.check_input_params_file(
        params,
        data_validation.schema,
    )

    pixels_of_interest = {}
    if "pixels_of_interest" in params.keys():
        pixels_of_interest = params["pixels_of_interest"]

    simulation = SimulationRealDataNN(
        **params["simu_init"],
        yaml_file=yaml_file,
        path_data=path_data,
        path_outputs=path_outputs,
        path_models=path_models,
        forward_model_fixed_params=params["forward_model"]["fixed_params"],
        pixels_of_interest=pixels_of_interest,
    )

    simulation.main(
        params=params,
        path_data_cloud=path_data,
        point_challenger=point_challenger,
    )
