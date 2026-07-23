import os
from typing import Dict, List, Optional, Union

import numpy as np

from beetroots.modelling.likelihoods.approx_censored_add_mult import (
    MixingModelsLikelihood,
)
from beetroots.modelling.likelihoods.gaussian_censored import CensoredGaussianLikelihood
from beetroots.modelling.priors.l22_laplacian_prior import L22LaplacianSpatialPrior
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.modelling.target_distribution.posterior import Posterior
from beetroots.sampler.utils.sampler_params import MySamplerParams
from beetroots.simulations.astro.forward_map_setup.abstract_nn import SimulationNN
from beetroots.simulations.astro.observation_setup.abstract_real_data import (
    SimulationRealData,
)
from beetroots.simulations.astro.sampler_setup.abstract_sampler_hierarchical import (
    SimulationMyGibbsSampler,
)


class SimulationRealDataNN(SimulationNN, SimulationRealData, SimulationMyGibbsSampler):
    pass  # TODO: Implement the class with the hierarchical model code structure.
