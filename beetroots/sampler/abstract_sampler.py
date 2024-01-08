import abc
from sys import byteorder

import numpy as np

from beetroots.modelling.posterior import Posterior
from beetroots.sampler.utils.mml import EBayesMMLE


class Sampler(abc.ABC):
    def __init__(
        self,
        rng: np.random.Generator = np.random.default_rng(42),
    ):
        self.rng = rng

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
        r"""sets the state and inc of the random generator to a specific value. Enables to re-launch a sampling.

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

        # step 2 :
        state = self.rng.__getstate__()
        state["state"]["state"] = loaded_state
        state["state"]["inc"] = loaded_inc
        self.rng.__setstate__(state)

    def generate_random_start_Theta(self, posterior: Posterior):
        """generates a random element of the hypercube defined by the lower and upper bounds with uniform distribution

        Parameters
        ----------
        posterior : Posterior
            contains the lower and upper bounds of the hypercube

        Returns
        -------
        x : np.array of shape (N, D)
            random element of the hypercube defined by the lower and upper bounds with uniform distribution
        """
        if posterior.prior_indicator is not None:
            x = (
                self.rng.uniform(0, 1, size=(posterior.N, posterior.D))
                * (
                    posterior.prior_indicator.upper_bounds[None, :]
                    - posterior.prior_indicator.lower_bounds[None, :]
                )
                + posterior.prior_indicator.lower_bounds[None, :]
            )

        else:
            x = self.rng.standard_normal(size=(posterior.N, posterior.D))

        return x

    @abc.abstractmethod
    def generate_random_start_Theta_1pix(
        self, x: np.ndarray, posterior: Posterior, idx_pix: np.ndarray
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def sample(self, posterior: Posterior):
        pass

    def sample_regu_hyperparams(
        self,
        posterior: Posterior,
        regu_weights_optimizer: EBayesMMLE,
        t: int,
        current_Theta: np.ndarray,
    ) -> np.ndarray:
        r"""
        Updates the spatial regularization hyperparameters
        """
        assert self.N > 1
        tau_tm1 = posterior.prior_spatial.weights * 1

        neglog_prior = posterior.prior_spatial.neglog_pdf(
            current_Theta,
            with_weights=False,
        )  # (D,)

        new_tau = regu_weights_optimizer.update(tau_tm1, t, neglog_prior)
        # new_tau[3] = 20  # fix tau of A_v
        return new_tau

        # assert self.N > 1
        # # hyperparameters
        # a = np.array([1] + [1 for d in range(self.D_no_kappa)])
        # # b = np.array([100] + [100 for d in range(self.D_no_kappa)]) # limit to ~1
        # # b = np.array([10] + [10 for d in range(self.D_no_kappa)]) # limit to ~10
        # b = np.array([1] + [1 for d in range(self.D_no_kappa)])  # limit to ~100
        # # b = np.array([0.1] + [0.1 for d in range(self.D_no_kappa)])  # limit to ~1000

        # h_Theta = conditional.prior_spatial.neglog_pdf(
        #     self.current["Theta"],
        #     with_weights=False,
        # )

        # tau = np.zeros((self.D,))
        # for d in range(self.D):
        #     tau[d] = self.rng.gamma(
        #         shape=a[d] + self.N,
        #         scale=1 / (b[d] + h_Theta[d]),
        #     )

        # return tau
