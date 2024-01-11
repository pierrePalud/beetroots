import abc
from sys import byteorder
from typing import Optional

import numpy as np

from beetroots.modelling.posterior import Posterior
from beetroots.sampler.saver.abstract_saver import Saver
from beetroots.sampler.utils.mml import EBayesMMLE


class Sampler(abc.ABC):
    r"""abstract class designed to sample from a distribution defined with the ``Posterior`` class."""

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
                self.rng.__getstate__()["state"]["state"].to_bytes(32, byteorder),
            )
        )
        rng_inc_array = np.array(
            bytearray(
                self.rng.__getstate__()["state"]["inc"].to_bytes(32, byteorder),
            )
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
        state = self.rng.__getstate__()
        state["state"]["state"] = loaded_state
        state["state"]["inc"] = loaded_inc
        self.rng.__setstate__(state)

    def generate_random_start_Theta(self, posterior: Posterior):
        r"""generates a random element of the hypercube defined by the lower and upper bounds with uniform distribution

        Parameters
        ----------
        posterior : Posterior
            contains the lower and upper bounds of the hypercube

        Returns
        -------
        Theta : np.array of shape (N, D)
            random element of the hypercube defined by the lower and upper bounds with uniform distribution
        """
        if posterior.prior_indicator is not None:
            Theta = (
                self.rng.uniform(0, 1, size=(posterior.N, posterior.D))
                * (
                    posterior.prior_indicator.upper_bounds[None, :]
                    - posterior.prior_indicator.lower_bounds[None, :]
                )
                + posterior.prior_indicator.lower_bounds[None, :]
            )

        else:
            Theta = self.rng.standard_normal(size=(posterior.N, posterior.D))

        return Theta

    @abc.abstractmethod
    def generate_random_start_Theta_1pix(
        self, Theta: np.ndarray, posterior: Posterior, idx_pix: np.ndarray
    ) -> np.ndarray:
        r"""draws a random vectors for components :math:`n` (e.g., a pixel :math:`\theta_n`).

        Parameters
        ----------
        Theta : np.ndarray
            current iterate
        posterior : Posterior
            contains the lower and upper bounds of the hypercube
        idx_pix : np.ndarray
            indices of the pixels

        Returns
        -------
        np.array of shape (n_pix, self.k_mtm, D)
            random element of the hypercube defined by the lower and upper bounds with uniform distribution
        """
        pass

    @abc.abstractmethod
    def sample(
        self,
        posterior: Posterior,
        saver: Saver,
        max_iter: int,
        Theta_0: Optional[np.ndarray],
        T_BI: Optional[int],
        **kwargs,
    ) -> None:
        """main method of the class, runs the sampler on the input posterior distribution

        Parameters
        ----------
        posterior : Posterior
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
        posterior: Posterior,
        regu_weights_optimizer: EBayesMMLE,
        t: int,
        current_Theta: np.ndarray,
    ) -> np.ndarray:
        r"""updates the spatial regularization weight vector (denoted :math:`\tau` in :cite:t:`paludEfficientSamplingNon2023`) using the marginal likelihood optimizer defined in :cite:t:`vidalMaximumLikelihoodEstimation2020`

        Parameters
        ----------
        posterior : Posterior
            posterior distribution
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
        assert posterior.prior_spatial is not None

        tau_tm1 = posterior.prior_spatial.weights * 1

        neglog_prior = posterior.prior_spatial.neglog_pdf(
            current_Theta,
            with_weights=False,
        )  # (D,)

        new_tau = regu_weights_optimizer.update(tau_tm1, t, neglog_prior)
        # new_tau[3] = 20  # fix tau of A_v
        return new_tau
