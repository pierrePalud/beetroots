import abc
import os
from typing import Dict, Optional, Tuple

import numpy as np

from beetroots.modelling.forward_maps.abstract_base import ForwardMap
from beetroots.simulations.astro.abstract_astro_simulation import AstroSimulation
from beetroots.space_transform.abstract_transform import Scaler


class SimulationForwardMap(AstroSimulation, abc.ABC):
    r"""abstract class for to set up the forward map for an inversion of astrophysical data"""

    @abc.abstractmethod
    def setup_forward_map(self, **kwargs) -> Tuple[Scaler, ForwardMap]:
        """sets up the forward map and the scaler

        Returns
        -------
        Scaler
            contains the transformation of the Theta values from their natural space to their scaled space (used by the forward map and in which the sampling happens) and its inverse
        ForwardMap
            forward map to be used in the inversion
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
