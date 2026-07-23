import abc
import warnings
from sys import byteorder
from typing import Optional, Union

import numpy as np

from beetroots.modelling.component_distribution import ComponentDistribution
from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.saver.abstract_saver import Saver
from beetroots.sampler.utils.mml import EBayesMMLE


class Sampler(abc.ABC):
    r"""abstract class designed to sample from a distribution defined with the ``TargetDistribution`` class."""

    ESS_OPTIM = 1_000
    r"""number of random reproduced observations to draw to evaluate the model checking p-value for optimization procedures"""

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        rng: np.random.Generator = np.random.default_rng(42),
    ):
        r"""

        Parameters
        ----------
        D : int
            total number of physical parameters to reconstruct
        L : int
            number of observables per component :math:`n`
        N : int
            total number of pixels to reconstruct
        rng : np.random.Generator
            numpy random generator (for sampling reproducibility), by default np.random.default_rng(42)
        """
        self.D = D
        r"""int: total number of physical parameters to reconstruct"""
        self.L = L
        r"""int: number of observables per component :math:`n`"""
        self.N = N
        r"""int: total number of pixels to reconstruct"""

        self.rng = rng
        r"""np.random.Generator: numpy random generator (for sampling reproducibility)"""

    def get_rng_state(self):
        r"""extracts the current state and inc of the random generator

        Returns
        -------
        rng_state_array : np.array of ints
            current state of the random generator
        rng_inc_array : np.array of ints
            current inc of the random generator

        Note
        ----
        1. state and inc are very large integers, and thus need to be converted to hex format (later to an array of ints) to be saved in a .h5 file https://docs.python.org/3/library/stdtypes.html#int.to_bytes
        2. need 32 bytes in length: otherwise, the inverse operation int.from_bytes(state_array,byteorder) does not coincide with the original int value
        """
        rng_state_array = np.array(
            bytearray(
                self.rng.bit_generator.__getstate__()[0]["state"]["state"].to_bytes(
                    32, byteorder
                ),
            ),
            dtype=np.uint8,
        )
        rng_inc_array = np.array(
            bytearray(
                self.rng.bit_generator.__getstate__()[0]["state"]["inc"].to_bytes(
                    32, byteorder
                ),
            ),
            dtype=np.uint8,
        )
        return rng_state_array, rng_inc_array

    def set_rng_state(self, state_bytes, inc_bytes):
        r"""sets the state and inc of the random generator to a specific value. Enables to re-launch a sampling when the rng state is regularly saved (see ``MySaver`` class).

        Parameters
        ----------
        state_bytes : bytes
            state to be set
        inc_bytes : bytes
            inc to be set
        """
        # step 1 : convert bytes to int
        loaded_state = int.from_bytes(state_bytes, byteorder)
        loaded_inc = int.from_bytes(inc_bytes, byteorder)

        # step 2 : set rng state
        state = self.rng.bit_generator.__getstate__()[0]
        state["state"]["state"] = loaded_state
        state["state"]["inc"] = loaded_inc
        self.rng.__setstate__(state)

    # def generate_random_start_Var(
    #     self,
    #     target_distribution: TargetDistribution,
    #     type_init: Union[str, None] = None,
    # ) -> np.ndarray:
    #     r"""generates a random element of the hypercube defined by the lower and upper bounds with uniform distribution

    #     Parameters
    #     ----------
    #     target_distribution : TargetDistribution
    #         contains the lower and upper bounds of the hypercube if there is an indicator prior. Otherwise it uses the 'type_init' parameter.
    #     type_init : Union[str, None], optional
    #         type of distribution to sample from, by default None uses a standard normal distribution. If a positivity constraint is needed, use 'half-gaussian'.

    #     Returns
    #     -------
    #     Var : np.array of shape (N, D)
    #         random element of the hypercube defined by the lower and upper bounds with uniform distribution
    #     """
    #     has_indicator = False
    #     prior_indicator = None
    #     for (
    #         key,
    #         component_distribution,
    #     ) in target_distribution.distribution_components.items():
    #         assert isinstance(component_distribution, ComponentDistribution)
    #         if isinstance(component_distribution, SmoothIndicatorPrior):
    #             has_indicator = True
    #             prior_indicator = component_distribution

    #     if has_indicator:
    #         Var = (
    #             self.rng.uniform(0, 1, size=target_distribution.var_shape)
    #             * (
    #                 prior_indicator.upper_bounds[None, :]
    #                 - prior_indicator.lower_bounds[None, :]
    #             )
    #             + prior_indicator.lower_bounds[None, :]
    #         )
    #     else:
    #         Var = self.rng.standard_normal(size=target_distribution.var_shape)

    #     if isinstance(type_init, str) and type_init.lower() == "half-gaussian":
    #         Var = np.abs(Var)
    #     elif type_init is not None:
    #         warnings.warn(
    #             f"'{type_init}' is not an unknown type of initialization. The initialization will be done with a standard normal distribution. The only type of initialization distribution that is implemented for the moment is 'half-gaussian'. It you want to use a standard normal distribution, you can remove the 'type_init' parameter as it is the default behavior when no indicator prior is available."
    #         )

    #     return Var

    # @abc.abstractmethod # TODO: check if there is a problem if we remove it (not necessary in MyGibbsSampler)
    # def generate_random_start_Theta_1pix(
    #     self, Theta: np.ndarray, posterior: Posterior, idx_pix: np.ndarray
    # ) -> np.ndarray:
    #     r"""draws a random vectors for components :math:`n` (e.g., a pixel :math:`\theta_n`).

    #     Parameters
    #     ----------
    #     Theta : np.ndarray
    #         current iterate
    #     posterior : Posterior
    #         contains the lower and upper bounds of the hypercube
    #     idx_pix : np.ndarray
    #         indices of the pixels

    #     Returns
    #     -------
    #     np.array of shape (n_pix, self.k_mtm, D)
    #         random element of the hypercube defined by the lower and upper bounds with uniform distribution
    #     """
    #     pass

    @abc.abstractmethod
    def sample(
        self,
        target_distributions: Union[TargetDistribution, dict[str, TargetDistribution]],
        saver: Saver,
        max_iter: int,
        Theta_0: Optional[np.ndarray],
        T_BI: Optional[int],
        **kwargs,
    ) -> None:
        """main method of the class, runs the sampler on the input target distribution

        Parameters
        ----------
        target_distribution : TargetDistribution
            probability distribution to be sampled
        saver : Saver
            object responsible for progressively saving the Markov chain data during the run
        max_iter : int
            total duration of a Markov chain
        Theta_0 : Optional[np.ndarray], optional
            starting point
        T_BI : int, optional
            duration of the `Burn-in` phase
        """
        pass

    def sample_regu_hyperparams(
        self,
        spatial_prior: SpatialPrior,
        regu_weights_optimizer: EBayesMMLE,
        t: int,
        current_Theta: np.ndarray,
        is_scalar: bool,
    ) -> np.ndarray:
        r"""updates the spatial regularization weight vector (denoted :math:`\tau` in :cite:t:`paludEfficientSamplingNon2023`) using the marginal likelihood optimizer defined in :cite:t:`vidalMaximumLikelihoodEstimation2020`

        Parameters
        ----------
        spatial_prior : SpatialPrior
            spatial prior to be used
        regu_weights_optimizer : EBayesMMLE
            marginal likelihood optimizer
        t : int
            time step
        current_Theta : np.ndarray
            current iterate

        Returns
        -------
        np.ndarray
            updated spatial regularization weight vector
        """
        assert self.N > 1
        assert spatial_prior is not None

        tau_tm1 = spatial_prior.weights * 1

        neglog_prior = spatial_prior.neglog_pdf(
            with_weights=False, paramwise=True, chromatic_gibbs=False
        )  # (D,)

        if is_scalar:
            new_tau = regu_weights_optimizer.update(
                tau_tm1.mean(), t, neglog_prior.sum()
            )
            return new_tau * np.ones(self.D)
        else:
            new_tau = regu_weights_optimizer.update(tau_tm1, t, neglog_prior)
            return new_tau
