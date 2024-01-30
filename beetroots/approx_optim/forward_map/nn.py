import pickle
from typing import Dict, Optional, Tuple

import numpy as np

from beetroots.approx_optim.forward_map.abstract_forward_map import (
    ApproxOptimForwardMap,
)
from beetroots.modelling.forward_maps.neural_network_approx import NeuralNetworkApprox
from beetroots.space_transform.transform import MyScaler


class ApproxOptimNN(ApproxOptimForwardMap):
    r"""handles the generation of a dataset of :math:`\log_{10} f_{\ell}(\theta)` values for a neural network forward map"""

    def setup_forward_map(
        self,
        forward_model_name: str,
        force_use_cpu: bool,
        fixed_params: Dict[str, Optional[float]],
        is_log_scale_params: Dict[str, bool],
    ) -> Tuple[MyScaler, NeuralNetworkApprox]:
        print(f"fixed values: {fixed_params}")

        with open(
            f"{self.MODELS_PATH}/{forward_model_name}/scaler.pickle",
            "rb",
        ) as file_:
            scaler_sklearn = pickle.load(file_)

        scaler = MyScaler(
            mean_=scaler_sklearn.mean_.flatten(),
            std_=scaler_sklearn.scale_.flatten(),
            list_is_log=list(is_log_scale_params.values()),
        )

        # transformation from linear scale (in degrees) to scaled

        # angle_scaled = (angle - 30.0) / 20.0

        # eg {"kappa":None, "Pth":None, "G0":None, "AV":None, "angle":0.}
        fixed_params_scaled = self.scale_dict_fixed_params(scaler, fixed_params)

        # load forward model
        forward_map = NeuralNetworkApprox(
            self.MODELS_PATH,
            forward_model_name,
            fixed_params_scaled,
            device="cpu" if force_use_cpu else None,
        )
        forward_map.restrict_to_output_subset(self.list_lines)
        return scaler, forward_map

    def compute_log10_f_Theta(
        self,
        dict_forward_model: dict,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
    ) -> np.ndarray:

        scaler, forward_map = self.setup_forward_map(**dict_forward_model)

        lower_bounds = scaler.from_lin_to_scaled(
            lower_bounds_lin.reshape((1, self.D)),
        ).flatten()  # (D,)
        upper_bounds = scaler.from_lin_to_scaled(
            upper_bounds_lin.reshape((1, self.D)),
        ).flatten()  # (D,)

        # restrict bounds to the unfixed parameters
        lower_bounds = lower_bounds[self.list_idx_sampling] * 1
        upper_bounds = upper_bounds[self.list_idx_sampling] * 1

        fixed_params_scaled = self.scale_dict_fixed_params(
            scaler, dict_forward_model["fixed_params"]
        )
        print(fixed_params_scaled)

        Theta = np.zeros((self.K**self.D_sampling, self.D))
        Theta[:, self.list_idx_sampling] = self.sample_theta(lower_bounds, upper_bounds)

        for i, v in enumerate(fixed_params_scaled.values()):
            if v is not None:
                assert np.allclose(Theta[:, i], 0.0), Theta[:10]
                Theta[:, i] = v * 1

        print(f"Theta.shape = {Theta.shape}")

        # the division is to get log in base 10
        print("starting computation of set of log10 f(theta) values")

        log10_f_Theta = forward_map.evaluate_log(Theta) / np.log(10)
        assert log10_f_Theta.shape == (
            self.N_samples_theta,
            self.L,
        ), f"log10_f_Theta.shape = {log10_f_Theta.shape}, (self.N_samples_theta, self.L) = {(self.N_samples_theta, self.L)}"

        print(f"log10_f_Theta.shape = {log10_f_Theta.shape}")
        print("computation of set of log10 f(theta) values done")

        return log10_f_Theta
