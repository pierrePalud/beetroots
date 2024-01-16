import numpy as np

from beetroots.approx_optim.forward_map.abstract_forward_map import (
    ApproxOptimForwardMap,
)
from beetroots.simulations.astro.forward_map.abstract_nn import SimulationNN


class ApproxOptimNN(ApproxOptimForwardMap):
    r"""handles the generation of a dataset of :math:`\log_{10} f_{\ell}(\theta)` values for a neural network forward map"""

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
        ).flatten()  # (D,)
        upper_bounds = scaler.from_lin_to_scaled(
            upper_bounds_lin.reshape((1, self.D)),
        ).flatten()  # (D,)

        # restrict bounds to the unfixed parameters
        lower_bounds = lower_bounds[self.list_idx_sampling] * 1
        upper_bounds = upper_bounds[self.list_idx_sampling] * 1

        dict_fixed_params_scaled = simulation.scale_dict_fixed_params(
            scaler, dict_forward_model["fixed_params"]
        )
        print(dict_fixed_params_scaled)

        Theta = np.zeros((self.K**self.D_sampling, self.D))
        Theta[:, self.list_idx_sampling] = self.sample_theta(lower_bounds, upper_bounds)

        for i, v in enumerate(dict_fixed_params_scaled.values()):
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
