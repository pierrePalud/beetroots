# import json
import os
import pickle
import warnings
from typing import Dict, Tuple

import numpy as np

from beetroots.modelling.forward_maps.neural_network_approx import NeuralNetworkApprox
from beetroots.simulations.astro.forward_map.abstract_forward_map import (
    SimulationForwardMap,
)
from beetroots.space_transform.transform import MyScaler

warnings.filterwarnings("ignore")


class SimulationNN(SimulationForwardMap):
    def setup_forward_map(
        self,
        forward_model_name: str,
        force_use_cpu: bool,
        dict_fixed_params: Dict[str, float],
        dict_is_log_scale_params: Dict[str, bool],
    ) -> Tuple[MyScaler, NeuralNetworkApprox]:
        models_path = f"{os.path.dirname(os.path.abspath(__file__))}"
        models_path += "/../../../../../data/models"

        print(f"fixed values: {dict_fixed_params}")

        with open(
            f"{models_path}/{forward_model_name}/scaler.pickle",
            "rb",
        ) as file_:
            scaler_sklearn = pickle.load(file_)

        scaler = MyScaler(
            mean_=scaler_sklearn.mean_.flatten(),
            std_=scaler_sklearn.scale_.flatten(),
            list_is_log=list(dict_is_log_scale_params.values()),
        )

        # transformation from linear scale (in degrees) to scaled

        # angle_scaled = (angle - 30.0) / 20.0

        # eg {"kappa":None, "Pth":None, "G0":None, "AV":None, "angle":0.}
        arr_fixed_values = np.ones((1, len(dict_fixed_params)))
        for i, name in enumerate(dict_fixed_params.keys()):
            if dict_fixed_params[name] is not None:
                arr_fixed_values[0, i] = dict_fixed_params[name] * 1

        arr_fixed_values_scaled = scaler.from_lin_to_scaled(arr_fixed_values)

        dict_fixed_params_scaled = {
            name: arr_fixed_values_scaled[0, i]
            if dict_fixed_params[name] is not None
            else None
            for i, name in enumerate(dict_fixed_params.keys())
        }

        # load forward model
        forward_map = NeuralNetworkApprox(
            forward_model_name,
            dict_fixed_params_scaled,
            device="cpu" if force_use_cpu else None,
        )
        forward_map.restrict_to_output_subset(self.list_lines_fit)
        return scaler, forward_map
