import abc
import os
from typing import List, Optional

import h5py
import numpy as np

from beetroots.space_transform.abstract_transform import Scaler


class Saver:
    """enable to regularly save the progression of the markov chain to a `.hdf5` file"""

    __slots__ = (
        "N",
        "D",
        "L",
        "results_path",
        "batch_size",
        "freq_save",
        "scaler",
        "t_last_save",
        "next_batch_size",
        "final_next_batch_size",
        "memory",
        "save_forward_map_evals",
    )

    def __init__(
        self,
        N: int,
        D: int,
        D_sampling: int,
        L: int,
        scaler: Scaler,
        results_path: Optional[str] = "",
        batch_size: Optional[int] = None,
        freq_save: int = 1,
        save_forward_map_evals: bool = False,
        list_idx_sampling: Optional[List[int]] = None,
    ):
        """

        Parameters
        ----------
        N : int
            total number of pixels to reconstruct
        D : int
            total nunmber of physical parameters to reconstruct
        L : int
            number of observed lines
        scaler : Scaler
            permets transformations from sampling space to interpretable space
        results_path : str
            path towards the hdf5 output file, by default ""
        batch_size : int, optional
            number of iterations between two saves on file, by default None
        freq_save : int, optional
            save one sample in every (freq_save). Used to save disk space., by default 1
        save_forward_map_evals: bool, optional
            wether or not to save the forward model evaluations and gradients, by default False
        """
        self.N = N
        self.D = D
        self.D_sampling = D_sampling
        self.L = L

        if list_idx_sampling is not None:
            self.list_idx_sampling = list_idx_sampling
        else:
            self.list_idx_sampling = np.arange(self.D)

        self.results_path = results_path
        if len(results_path) > 0 and not (os.path.isdir(results_path)):
            os.mkdir(results_path)

        self.batch_size = batch_size
        self.freq_save = freq_save
        self.scaler = scaler

        # these two attributes are initialized by initialize_memory
        # updated during sampling
        self.t_last_save = 0
        self.next_batch_size = 0
        self.final_next_batch_size = 0
        self.memory = dict()

        self.save_forward_map_evals = save_forward_map_evals

    def set_results_path(self, results_path: str) -> None:
        self.results_path = results_path
        if len(results_path) > 0 and not (os.path.isdir(results_path)):
            os.mkdir(results_path)
        return

    def check_need_to_save(self, t: int) -> bool:
        """checks wether or not the memory should be saved to a `.hdf5` file

        Parameters
        ----------
        t : int
            current iteration index

        Returns
        -------
        bool
            wether or not to save to disk now
        """
        current_t_in_batch = t - self.t_last_init + 1
        return current_t_in_batch == self.next_batch_size * self.freq_save

    def check_need_to_update_memory(self, t: int) -> bool:
        """checks wether or not the memory should be updated

        Parameters
        ----------
        t : int
            current iteration index

        Returns
        -------
        bool
            wether or not to save to update the memory
        """
        return t % self.freq_save == 0

    @abc.abstractmethod
    def initialize_memory(
        self,
        T_MC: int,
        t: int,
        x: np.ndarray,
        forward_map_evals: dict = dict(),
        nll_utils: dict = dict(),
        dict_objective: dict = dict(),
        additional_sampling_log: dict = dict(),
    ) -> None:
        """initializes the memory with the correct shapes"""
        pass

    @abc.abstractmethod
    def update_memory(
        self,
        t: int,
        x: np.ndarray,
        forward_map_evals: dict = dict(),
        nll_utils: dict = dict(),
        dict_objective: dict = dict(),
        additional_sampling_log: dict = dict(),
        rng_state_array: Optional[np.ndarray] = None,
        rng_inc_array: Optional[np.ndarray] = None,
    ) -> None:
        """updates the memory with new information. All of the potential entries are optional except for the current iterate."""
        pass

    def save_to_file(self):
        """Saves the current memory content to a `.hdf5` file"""

        if self.t_last_init == 1:  # if first writing
            with h5py.File(
                os.path.join(self.results_path, "mc_chains.hdf5"),
                "w",
            ) as f:
                for k, v in self.memory.items():
                    f.create_dataset(k, data=v, maxshape=(None,) + v.shape[1:])

        else:  # append data to already created file
            with h5py.File(
                os.path.join(self.results_path, "mc_chains.hdf5"),
                "a",
            ) as f:
                for k, v in self.memory.items():
                    f[k].resize(
                        f[k].shape[0] + self.final_next_batch_size,
                        axis=0,
                    )
                    f[k][-self.final_next_batch_size :] = v

        self.memory = dict()

    def save_additional(
        self,
        list_arrays: List[np.ndarray],
        list_names: List[str],
    ) -> None:
        """Saves additional content to a `.hdf5` file"""

        with h5py.File(
            os.path.join(self.results_path, "mc_chains.hdf5"),
            "a",
        ) as f:
            for name, array_ in zip(list_names, list_arrays):
                f.create_dataset(name, data=array_)

        return
