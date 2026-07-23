import abc
from typing import Dict, Optional, Tuple

from beetroots.modelling.forward_maps.abstract_base import ForwardMap
from beetroots.simulations.astro.abstract_astro_simulation import AstroSimulation
from beetroots.simulations.astro.forward_map_setup.utils import _scale_dict_fixed_params
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
        r"""transforms the fixed parameters from their value in their natural space to their value in the space in which they are to be sampled.

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
        return _scale_dict_fixed_params(scaler, dict_fixed_params)
