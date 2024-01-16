import numpy as np

from beetroots.approx_optim.forward_map.abstract_forward_map import (
    ApproxOptimForwardMap,
)
from beetroots.simulations.astro.forward_map.abstract_poly_reg import (
    SimulationPolynomialReg,
)


class ApproxOptimPolynomialReg(ApproxOptimForwardMap):
    r"""handles the generation of a dataset of :math:`\log_{10} f_{\ell}(\theta)` values  for a polynomial forward map

    .. warning::

            Unfinished class.
            Needs to be updated for D_sampling to remove the angle parameter.
            Needs work on the :class:`.PolynomialApprox` class
    """

    def compute_log10_f_Theta(
        self,
        forward_model_name: str,
        angle: float,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
    ):
        r""".. warning::

        Unfinished method.
        Needs to be updated for D_sampling to remove the angle parameter.
        Needs work on the :class:`.PolynomialApprox` class
        """
        simulation = SimulationPolynomialReg()
        simulation.list_lines_fit = self.list_lines * 1
        scaler, forward_map = simulation.setup_forward_map(
            forward_model_name=forward_model_name,
            force_use_cpu=False,
            angle=angle,
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
