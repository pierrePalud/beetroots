from typing import Optional

import numpy as np

from beetroots.sampler.saver.abstract_saver import Saver


class HierarchicalSaver(Saver):
    def initialize_memory(
        self,
        T_MC: int,
        t: int,
        x: np.ndarray,
        u: np.ndarray,
        forward_map_evals: dict = dict(),
        forward_map_evals_u: dict = dict(),
        nll_utils: dict = dict(),
        nll_utils_u: dict = dict(),
        dict_objective: dict = dict(),
        dict_objective_u: dict = dict(),
        additional_sampling_log: dict = dict(),
        additional_sampling_log_u: dict = dict(),
    ) -> None:
        """initializes the memory with the correct shapes

        Parameters
        ----------
        T_MC : int
            size of the markov chain to be sampled
        t : int
            current iteration index
        """
        if self.batch_size is None:
            self.batch_size = T_MC

        self.t_last_init = t * 1
        self.next_batch_size = min(self.batch_size, (T_MC - t + 1) // self.freq_save)
        # print(t, self.next_batch_size)
        self.final_next_batch_size = self.next_batch_size

        self.memory["list_Theta"] = np.zeros((self.final_next_batch_size, *x.shape))

        self.memory["list_U"] = np.zeros((self.final_next_batch_size, *u.shape))

        if self.save_forward_map_evals:
            for k, v in forward_map_evals.items():
                if np.all(["grad" not in k, "hess_diag" not in k]):
                    self.memory[f"list_{k}"] = np.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )

            for k, v in forward_map_evals_u.items():
                if np.all(["grad" not in k, "hess_diag" not in k]):
                    self.memory[f"list_u_{k}"] = np.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )

        for k, v in nll_utils.items():
            if np.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                self.memory[f"list_{k}"] = np.zeros(
                    (self.final_next_batch_size,) + v.shape
                )

        for k, v in nll_utils_u.items():
            if np.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                self.memory[f"list_u_{k}"] = np.zeros(
                    (self.final_next_batch_size,) + v.shape
                )

        for k, v in dict_objective.items():
            self.memory[f"list_{k}"] = np.zeros((self.final_next_batch_size,) + v.shape)

        for k, v in dict_objective_u.items():
            self.memory[f"list_u_{k}"] = np.zeros(
                (self.final_next_batch_size,) + v.shape
            )

        for k, v in additional_sampling_log.items():
            if isinstance(v, np.ndarray):
                self.memory[f"list_{k}"] = np.zeros(
                    (self.final_next_batch_size,) + v.shape
                )
            else:
                self.memory[f"list_{k}"] = np.zeros((self.final_next_batch_size,))

        for k, v in additional_sampling_log_u.items():
            if isinstance(v, np.ndarray):
                self.memory[f"list_u_{k}"] = np.zeros(
                    (self.final_next_batch_size,) + v.shape
                )
            else:
                self.memory[f"list_u_{k}"] = np.zeros((self.final_next_batch_size,))

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
        x: np.ndarray,
        u: np.ndarray,
        forward_map_evals: dict = dict(),
        forward_map_evals_u: dict = dict(),
        nll_utils: dict = dict(),
        nll_utils_u: dict = dict(),
        dict_objective: dict = dict(),
        dict_objective_u: dict = dict(),
        additional_sampling_log: dict = dict(),
        additional_sampling_log_u: dict = dict(),
        rng_state_array: Optional[np.ndarray] = None,
        rng_inc_array: Optional[np.ndarray] = None,
    ) -> None:
        """updates the memory with new information. All of the potential entries are optional except for the current iterate."""
        t_save = (t - self.t_last_init) // self.freq_save

        self.memory["list_Theta"][t_save, :, :] = self.scaler.from_scaled_to_lin(x)
        self.memory["list_U"][t_save, :, :] = u

        if self.save_forward_map_evals:
            for k, v in forward_map_evals.items():
                if np.all(["grad" not in k, "hess_diag" not in k]):
                    self.memory[f"list_{k}"][t_save] = v

            for k, v in forward_map_evals_u.items():
                if np.all(["grad" not in k, "hess_diag" not in k]):
                    self.memory[f"list_{k}"][t_save] = v

        for k, v in nll_utils.items():
            if np.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                self.memory[f"list_{k}"][t_save] = v

        for k, v in nll_utils_u.items():
            if np.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                self.memory[f"list_u_{k}"][t_save] = v

        for k, v in dict_objective.items():
            if k not in ["m_a", "s_a", "m_m", "s_m"]:
                self.memory[f"list_{k}"][t_save] = v

        for k, v in dict_objective_u.items():
            if k not in ["m_a", "s_a", "m_m", "s_m"]:
                self.memory[f"list_u_{k}"][t_save] = v

        for k, v in additional_sampling_log.items():
            self.memory[f"list_{k}"][t_save] = v

        for k, v in additional_sampling_log_u.items():
            self.memory[f"list_u_{k}"][t_save] = v

        if (rng_state_array is not None) and (rng_inc_array is not None):
            self.memory["list_rng_state"][t_save] = rng_state_array
            self.memory["list_rng_inc"][t_save] = rng_inc_array
