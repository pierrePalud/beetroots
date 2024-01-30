import abc
from typing import Dict, Optional, Tuple

import numpy as np

from beetroots.space_transform.abstract_transform import Scaler


class ApproxOptimForwardMap(abc.ABC):
    r"""abstract class that handles the generation of a dataset of :math:`\log_{10} f_{\ell}(\theta)` values"""

    @abc.abstractmethod
    def compute_log10_f_Theta(self):
        r"""Evaluates :math:`\log_{10} f` on a list of ``self.N_samples_theta`` :math:`\theta` values, each randomly generated in the hypercube defined by its lower and upper bounds.

        Parameters
        ----------
        forward_model_name : str
            name of the forward model to load (i.e., of the corresponding folder)
        angle : float
            angle at which the cloud is observed
        lower_bounds_lin : np.ndarray of shape (D,)
            array of lower bounds in linear scale
        upper_bounds_lin : np.ndarray of shape (D,)
            array of upper bounds in linear scale

        Returns
        -------
        np.ndarray of shape (self.N_samples_theta, L)
            evaluations of the ``self.N_samples_theta`` :math:`\theta` values for the L considered lines.
        """
        pass

    def scale_dict_fixed_params(
        self, scaler: Scaler, dict_fixed_params: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        r"""transforms the fixed parameters from their value in their natural spae to their value in the space in which they are to be sampled.

        Note
        ----
        since the scaler does not accept None or np.NaN, this transformation really needed a dedicated (though simple) method

        Parameters
        ----------
        scaler : Scaler
            contains the transformation of the physical parameters values from their natural space to their scaled space (in which the sampling happens) and its inverse
        dict_fixed_params : Dict[str, Optional[float]]
            contains the value of the fixed parameters in the physical parameter natural space. For example, {"kappa":None, "Pth":None, "G0":None, "AV":None, "angle":0.} (the values with None indicate that the parameter is not fixed, so here, only the angle is set, to 0 degree).

        Returns
        -------
        Dict[str, Optional[float]]
            contains the value of the fixed parameters in the space in which the physical parameter are to be sampled
        """
        arr_fixed_values = np.ones((1, len(dict_fixed_params)))
        for i, name in enumerate(dict_fixed_params.keys()):
            if dict_fixed_params[name] is not None:
                arr_fixed_values[0, i] = dict_fixed_params[name]

        arr_fixed_values_scaled = scaler.from_lin_to_scaled(arr_fixed_values)

        dict_fixed_params_scaled = {
            name: arr_fixed_values_scaled[0, i]
            if dict_fixed_params[name] is not None
            else None
            for i, name in enumerate(dict_fixed_params.keys())
        }
        return dict_fixed_params_scaled
