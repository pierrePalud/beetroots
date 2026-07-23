from typing import Optional

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from beetroots.sampler.saver.abstract_saver import Saver


class MyGibbsSaver(Saver):
    def initialize_memory(
        self,
        T_MC: int,
        t: int,
        sample_variables: dict[str, bool],
        current: dict = dict(),
        forward_map_evals: Optional[dict] = None,
        nlpdf_utils: dict = dict(),
        dict_objective: dict = dict(),
        additional_sampling_log: dict = dict(),
        forward_map_evals_keys: Optional[dict] = None,
        kernels_list: Optional[list] = None,
        dict_model_check: Optional[dict] = None,
    ) -> None:
        """initializes the memory with the correct shapes

        Parameters
        ----------
        T_MC : int
            size of the markov chain to be sampled
        t : int
            current iteration index
        kernels_list : Optional[list], optional
            list of kernels from which to save hyperparameters, by default None
        dict_model_check : Optional[dict], optional
            model checking statistics dictionary for checkpoint saving, by default None
        """
        if self.batch_size is None:
            self.batch_size = T_MC

        self.t_last_init = t * 1
        self.next_batch_size = min(self.batch_size, (T_MC - t + 1) // self.freq_save)
        # print(t, self.next_batch_size)
        self.final_next_batch_size = self.next_batch_size

        # Initialize kernels group for checkpoint data
        self.memory["kernels"] = {}
        if kernels_list is not None:
            for kernel in kernels_list:
                kernel_name = kernel.plot_name
                kernel_params = kernel._dict_parameters_saving()
                if kernel_params:  # Only save if there are parameters to save
                    self.memory["kernels"][kernel_name] = {}
                    for param_name, param_value in kernel_params.items():
                        # Store as single value (not history)
                        if isinstance(param_value, xp.ndarray):
                            self.memory["kernels"][kernel_name][param_name] = (
                                param_value.copy()
                            )
                        else:
                            self.memory["kernels"][kernel_name][param_name] = (
                                param_value
                            )

        # Initialize model_check group for checkpoint data
        self.memory["model_check"] = {}
        if dict_model_check is not None:
            for key, value in dict_model_check.items():
                if isinstance(value, xp.ndarray):
                    self.memory["model_check"][key] = value.copy()
                else:
                    self.memory["model_check"][key] = value

        for key in current.keys():
            self.memory[key] = {
                "list_var": xp.zeros(
                    (self.final_next_batch_size, *current[key]["var"].shape)
                )
            }

            if self.save_forward_map_evals and forward_map_evals is not None:
                for k, v in forward_map_evals.items():
                    if xp.all(["grad" not in k, "hess_diag" not in k]):
                        self.memory[f"list_{k}"] = xp.zeros(
                            (self.final_next_batch_size,) + v.shape
                        )

            for k, v in nlpdf_utils[key].items():
                if xp.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                    self.memory[key][f"list_{k}"] = xp.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )

            for k, v in dict_objective[key].items():
                if isinstance(v, float):
                    v = xp.asarray(v)
                self.memory[key][f"list_{k}"] = xp.zeros(
                    (self.final_next_batch_size,) + v.shape
                )

            for k, v in additional_sampling_log[key].items():
                if isinstance(v, xp.ndarray) and sample_variables[key]:
                    self.memory[key][f"list_{k}"] = xp.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )
                elif isinstance(v, str):
                    self.memory[key][f"list_{k}"] = xp.zeros(
                        (self.final_next_batch_size,), dtype="U100"
                    )
                else:
                    self.memory[key][f"list_{k}"] = xp.zeros(
                        (self.final_next_batch_size,)
                    )
        for k in set(additional_sampling_log.keys()).difference(
            set(sample_variables.keys())
        ):
            self.memory[f"list_{k}"] = xp.zeros(
                (self.final_next_batch_size,) + additional_sampling_log[k].shape
            )

        self.memory["list_rng_state"] = xp.zeros(
            (self.final_next_batch_size, 32),
            dtype=xp.uint8,
        )
        self.memory["list_rng_inc"] = xp.zeros(
            (self.final_next_batch_size, 32),
            dtype=xp.uint8,
        )

        self.memory["list_objective_global"] = xp.zeros(
            (self.final_next_batch_size,), dtype=xp.float64
        )

    def update_memory(
        self,
        t: int,
        current: dict,
        sample_variables: dict[str, bool],
        nlpdf_utils: dict = dict(),
        dict_objective: dict = dict(),
        additional_sampling_log: dict = dict(),
        rng_state_array: Optional[xp.ndarray] = None,
        rng_inc_array: Optional[xp.ndarray] = None,
        forward_map_evals: Optional[dict] = None,
        kernels_list: Optional[list] = None,
        dict_model_check: Optional[dict] = None,
    ) -> None:
        """updates the memory with new information. All of the potential entries are optional except for the current iterate.

        Parameters
        ----------
        kernels_list : Optional[list], optional
            list of kernels from which to save hyperparameters, by default None
        dict_model_check : Optional[dict], optional
            model checking statistics dictionary for checkpoint saving, by default None
        """
        # Update kernel hyperparameters (overwriting, not appending)
        if kernels_list is not None and "kernels" in self.memory:
            for kernel in kernels_list:
                kernel_name = kernel.plot_name
                kernel_params = kernel._dict_parameters_saving()
                if kernel_params:  # Only update if there are parameters to save
                    if kernel_name not in self.memory["kernels"]:
                        self.memory["kernels"][kernel_name] = {}
                    for param_name, param_value in kernel_params.items():
                        # Overwrite with latest values
                        if isinstance(param_value, xp.ndarray):
                            self.memory["kernels"][kernel_name][param_name] = (
                                param_value.copy()
                            )
                        else:
                            self.memory["kernels"][kernel_name][param_name] = (
                                param_value
                            )

        # Update model_check values (overwriting, not appending)
        if dict_model_check is not None and "model_check" in self.memory:
            for key, value in dict_model_check.items():
                if isinstance(value, xp.ndarray):
                    self.memory["model_check"][key] = value.copy()
                else:
                    self.memory["model_check"][key] = value

        t_save = (t - self.t_last_init) // self.freq_save

        for key in current.keys():
            Var = current[key]["var"]
            if hasattr(
                self.scaler[key], "D"
            ):  # For the variable of interest. We might not have sampled every component.
                Var_full = xp.zeros((*Var.shape[:-1], self.D))
                for i, idx in enumerate(self.list_idx_sampling):
                    Var_full[:, idx] = Var[:, i]
                self.memory[key]["list_var"][t_save] = self.scaler[
                    key
                ].from_scaled_to_lin(Var_full)[..., self.list_idx_sampling]
            else:
                self.memory[key]["list_var"][t_save] = self.scaler[
                    key
                ].from_scaled_to_lin(Var)

            if self.save_forward_map_evals and forward_map_evals is not None:
                for k, v in forward_map_evals.items():
                    if xp.all(["grad" not in k, "hess_diag" not in k]):
                        self.memory[f"list_{k}"][t_save] = v

            for k, v in nlpdf_utils[key].items():
                if xp.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                    self.memory[key][f"list_{k}"][t_save] = v

            for k, v in dict_objective[key].items():
                if k not in ["m_a", "s_a", "m_m", "s_m"]:
                    self.memory[key][f"list_{k}"][t_save] = v

            for k, v in additional_sampling_log[key].items():
                if sample_variables[key]:
                    self.memory[key][f"list_{k}"][t_save] = v

        for k in set(additional_sampling_log.keys()).difference(
            set(sample_variables.keys())
        ):
            self.memory[f"list_{k}"][t_save] = additional_sampling_log[k]

        if (rng_state_array is not None) and (rng_inc_array is not None):
            self.memory["list_rng_state"][t_save] = rng_state_array
            self.memory["list_rng_inc"][t_save] = rng_inc_array
        self.memory["list_objective_global"][t_save] = dict_objective[
            "objective_global"
        ]
