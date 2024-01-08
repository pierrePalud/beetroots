import numpy as np

from beetroots.approx_optim.forward_map.abstract_forward_map import (
    ApproxOptimForwardMap,
)
from beetroots.simulations.astro.forward_map.abstract_nn import SimulationNN


class ApproxOptimNN(ApproxOptimForwardMap):
    def compute_log10_f_Theta(
        self,
        dict_forward_model: dict,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
    ) -> np.ndarray:
        simulation = SimulationNN()
        simulation.list_lines_fit = self.list_lines * 1
        scaler, forward_map = simulation.setup_forward_map(
            forward_model_name=dict_forward_model["forward_model_name"],
            force_use_cpu=dict_forward_model["force_use_cpu"],
            dict_fixed_params=dict_forward_model["fixed_params"],
            dict_is_log_scale_params=dict_forward_model["is_log_scale_params"],
        )

        lower_bounds = scaler.from_lin_to_scaled(
            lower_bounds_lin.reshape((1, self.D)),
        ).flatten()
        upper_bounds = scaler.from_lin_to_scaled(
            upper_bounds_lin.reshape((1, self.D)),
        ).flatten()
        x = self.sample_theta(lower_bounds, upper_bounds)

        # the division is to get log in base 10
        log10_f_Theta = forward_map.evaluate_log(x) / np.log(10)
        assert log10_f_Theta.shape == (self.N_samples_theta, self.L)
        return log10_f_Theta
