# import json
import os
import pickle
import warnings
from typing import Tuple

from beetroots.modelling.forward_maps.regression_poly import PolynomialApprox
from beetroots.simulations.astro.forward_map.abstract_forward_map import (
    SimulationForwardMap,
)
from beetroots.space_transform.transform import MyScaler

warnings.filterwarnings("ignore")


class SimulationPolynomialReg(SimulationForwardMap):
    def setup_forward_map(
        self,
        forward_model_name: str,
        angle: float,
    ) -> Tuple[MyScaler, PolynomialApprox]:
        models_path = f"{os.path.dirname(os.path.abspath(__file__))}"
        models_path += "/../../../../../data/models"

        with open(
            f"{models_path}/{forward_model_name}/scaler.pickle",
            "rb",
        ) as file_:
            scaler_sklearn = pickle.load(file_)

        scaler = MyScaler(
            mean_=scaler_sklearn.mean_[:-1].flatten(),
            std_=scaler_sklearn.scale_[:-1].flatten(),
        )

        # transformation from linear scale (in degrees) to scaled
        angle_scaled = (angle - 30.0) / 20.0

        # load forward model
        forward_map = PolynomialApprox(
            forward_model_name,
            angle_scaled,
        )
        forward_map.restrict_to_output_subset(self.list_lines_fit)
        return scaler, forward_map
