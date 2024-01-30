# import json
import os
import pickle
import warnings
from typing import Dict, Optional, Tuple

import numpy as np

from beetroots.modelling.forward_maps.neural_network_approx import NeuralNetworkApprox
from beetroots.simulations.astro.forward_map.abstract_forward_map import (
    SimulationForwardMap,
)
from beetroots.space_transform.transform import MyScaler

warnings.filterwarnings("ignore")


class SimulationNN(SimulationForwardMap):
    r"""abstract class for to set up the forward map to an already defined neural network for an inversion of astrophysical data"""

    def setup_forward_map(
        self,
        forward_model_name: str,
        force_use_cpu: bool,
        dict_fixed_params: Dict[str, Optional[float]],
        dict_is_log_scale_params: Dict[str, bool],
    ) -> Tuple[MyScaler, NeuralNetworkApprox]:
        print(f"fixed values: {dict_fixed_params}")

        with open(
            f"{self.MODELS_PATH}/{forward_model_name}/scaler.pickle",
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
        dict_fixed_params_scaled = self.scale_dict_fixed_params(
            scaler, dict_fixed_params
        )

        # load forward model
        forward_map = NeuralNetworkApprox(
            self.MODELS_PATH,
            forward_model_name,
            dict_fixed_params_scaled,
            device="cpu" if force_use_cpu else None,
        )
        forward_map.restrict_to_output_subset(self.list_lines_fit)
        return scaler, forward_map
