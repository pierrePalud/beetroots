# import json
import os
import pickle
import warnings
from typing import Dict, Optional, Tuple

from beetroots.modelling.forward_maps.regression_poly import PolynomialApprox
from beetroots.simulations.astro.forward_map.abstract_forward_map import (
    SimulationForwardMap,
)
from beetroots.space_transform.transform import MyScaler

warnings.filterwarnings("ignore")


class SimulationPolynomialReg(SimulationForwardMap):
    r"""abstract class for to set up the forward map to an already defined polynomial for an inversion of astrophysical data"""

    def setup_forward_map(
        self,
        forward_model_name: str,
        dict_fixed_params: Dict[str, Optional[float]],
        dict_is_log_scale_params: Dict[str, bool],
    ) -> Tuple[MyScaler, PolynomialApprox]:

        with open(
            f"{self.MODELS_PATH}/{forward_model_name}/scaler.pickle",
            "rb",
        ) as file_:
            scaler_sklearn = pickle.load(file_)

        scaler = MyScaler(
            mean_=scaler_sklearn.mean_[:-1].flatten(),
            std_=scaler_sklearn.scale_[:-1].flatten(),
            list_is_log=list(dict_is_log_scale_params.values()),
        )

        # transformation from linear scale (in degrees) to scaled
        angle = dict_fixed_params["angle"]
        assert angle is not None
        angle_scaled = (angle - 30.0) / 20.0

        dict_fixed_params_scaled = self.scale_dict_fixed_params(
            scaler, dict_fixed_params
        )

        # load forward model
        forward_map = PolynomialApprox(
            self.MODELS_PATH,
            forward_model_name,
            dict_fixed_params_scaled,
            angle_scaled,
        )
        forward_map.restrict_to_output_subset(self.list_lines_fit)
        return scaler, forward_map
