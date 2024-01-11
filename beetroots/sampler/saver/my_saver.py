from typing import Optional

import numpy as np

from beetroots.sampler.saver.abstract_saver import Saver


class MySaver(Saver):
    def initialize_memory(
        self,
        T_MC: int,
        t: int,
        Theta: np.ndarray,
        forward_map_evals: dict = dict(),
        nll_utils: dict = dict(),
        dict_objective: dict = dict(),
        additional_sampling_log: dict = dict(),
    ) -> None:
        r"""initializes the memory with the correct shapes

        Parameters
        ----------
        T_MC : int
            size of the Markov chain to be sampled / optimization procedure
        t : int
            current iteration index
        Theta : np.ndarray
            current iterate in the Markov chain / optimization run
        forward_map_evals : dict, optional
            evaluations of the forward map and potentially derivatives, by default dict()
        nll_utils : dict, optional
            evaluation of utilitary values of the Likelihood class, by default dict()
        dict_objective : dict, optional
            contains the negative log posterior value and detailed components, by default dict()
        additional_sampling_log : dict, optional
            additional data on the sampling / optimization run, by default dict()
        """
        if self.batch_size is None:
            self.batch_size = T_MC

        self.t_last_init = t * 1
        self.next_batch_size = min(self.batch_size, (T_MC - t + 1) // self.freq_save)
        # print(t, self.next_batch_size)
        self.final_next_batch_size = self.next_batch_size

        self.memory["list_Theta"] = np.zeros(
            (self.final_next_batch_size, self.N, self.D_sampling)
        )

        if self.save_forward_map_evals:
            for k, v in forward_map_evals.items():
                if np.all(["grad" not in k, "hess_diag" not in k]):
                    self.memory[f"list_{k}"] = np.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )

        for k, v in nll_utils.items():
            if np.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                self.memory[f"list_{k}"] = np.zeros(
                    (self.final_next_batch_size,) + v.shape
                )

        for k, v in dict_objective.items():
            self.memory[f"list_{k}"] = np.zeros((self.final_next_batch_size,) + v.shape)

        for k, v in additional_sampling_log.items():
            if isinstance(v, np.ndarray):
                self.memory[f"list_{k}"] = np.zeros(
                    (self.final_next_batch_size,) + v.shape
                )
            else:
                self.memory[f"list_{k}"] = np.zeros(
                    (self.final_next_batch_size,),
                )

        self.memory["list_rng_state"] = np.zeros(
            (self.final_next_batch_size, 32),
            dtype=np.uint8,
        )
        self.memory["list_rng_inc"] = np.zeros(
            (self.final_next_batch_size, 32),
            dtype=np.uint8,
        )

    def update_memory(
        self,
        t: int,
        Theta: np.ndarray,
        forward_map_evals: dict = dict(),
        nll_utils: dict = dict(),
        dict_objective: dict = dict(),
        additional_sampling_log: dict = dict(),
        rng_state_array: Optional[np.ndarray] = None,
        rng_inc_array: Optional[np.ndarray] = None,
    ) -> None:
        r"""updates the memory with new information. All of the potential entries are optional except for the current iterate.

        Parameters
        ----------
        t : int
            current iteration index
        Theta : np.ndarray
            current iterate in the Markov chain / optimization run
        forward_map_evals : dict, optional
            evaluations of the forward map and potentially derivatives, by default dict()
        nll_utils : dict, optional
            evaluation of utilitary values of the Likelihood class, by default dict()
        dict_objective : dict, optional
            contains the negative log posterior value and detailed components, by default dict()
        additional_sampling_log : dict, optional
            additional data on the sampling / optimization run, by default dict()
        rng_state_array : Optional[np.ndarray], optional
            current state of the random generator (saved for sampling reproducibility), by default None
        rng_inc_array : Optional[np.ndarray], optional
            current inc of the random generator (saved for sampling reproducibility), by default None
        """
        t_save = (t - self.t_last_init) // self.freq_save

        Theta_full = np.zeros((Theta.shape[0], self.D))
        for i, idx in enumerate(self.list_idx_sampling):
            Theta_full[:, idx] = Theta[:, i]
        lin_Theta_full = self.scaler.from_scaled_to_lin(Theta_full)
        lin_Theta_full = lin_Theta_full[:, self.list_idx_sampling]
        self.memory["list_Theta"][t_save, :, :] = lin_Theta_full

        if self.save_forward_map_evals:
            for k, v in forward_map_evals.items():
                if np.all(["grad" not in k, "hess_diag" not in k]):
                    self.memory[f"list_{k}"][t_save] = v

        for k, v in nll_utils.items():
            if np.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                self.memory[f"list_{k}"][t_save] = v

        for k, v in dict_objective.items():
            if k not in ["m_a", "s_a", "m_m", "s_m"]:
                self.memory[f"list_{k}"][t_save] = v

        for k, v in additional_sampling_log.items():
            self.memory[f"list_{k}"][t_save] = v

        if (rng_state_array is not None) and (rng_inc_array is not None):
            self.memory["list_rng_state"][t_save] = rng_state_array
            self.memory["list_rng_inc"][t_save] = rng_inc_array
