import pickle
from typing import Dict, Optional, Tuple

import numpy as np

from beetroots.approx_optim.forward_map.abstract_forward_map import (
    ApproxOptimForwardMap,
)
from beetroots.modelling.forward_maps.regression_poly import PolynomialApprox
from beetroots.space_transform.transform import MyScaler


class ApproxOptimPolynomialReg(ApproxOptimForwardMap):
    r"""handles the generation of a dataset of :math:`\log_{10} f_{\ell}(\theta)` values  for a polynomial forward map

    .. warning::

        Unfinished class.
        Needs to be updated for D_sampling to remove the angle parameter.
        Needs work on the :class:`.PolynomialApprox` class
    """

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
        forward_map.restrict_to_output_subset(self.list_lines)
        return scaler, forward_map

    def compute_log10_f_Theta(
        self,
        dict_forward_model: dict,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
    ):
        r""".. warning::

        Unfinished method.
        Needs to be updated for D_sampling to remove the angle parameter.
        Needs work on the :class:`.PolynomialApprox` class
        """
        scaler, forward_map = self.setup_forward_map(**dict_forward_model)

        lower_bounds = scaler.from_lin_to_scaled(
            lower_bounds_lin.reshape((1, self.D)),
        ).flatten()
        upper_bounds = scaler.from_lin_to_scaled(
            upper_bounds_lin.reshape((1, self.D)),
        ).flatten()
        Theta = self.sample_theta(lower_bounds, upper_bounds)

        # the division is to get log in base 10
        log10_f_Theta = forward_map.evaluate_log(Theta) / np.log(10)
        assert log10_f_Theta.shape == (self.N_samples_theta, self.L)

        return log10_f_Theta
