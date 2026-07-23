import abc
import os
from typing import List, Optional

import h5py
import numpy as np

from beetroots.space_transform.abstract_transform import Scaler


def save_dict_to_hdf5(file, d, parent_key=""):
    for k, v in d.items():
        if k in ["kernels", "model_check"] and parent_key == "":
            create_single_dict_to_hdf5(file, v, parent_key + "/" + k)
        elif isinstance(v, dict):
            save_dict_to_hdf5(file, v, parent_key + "/" + k)
        else:
            # print("{}, dtype: {}".format(parent_key + "/" + k, v.dtype))

            # NOTE: handling array of strings with h5py, which does not support numpy 'U' types
            # https://numpy.org/doc/stable/reference/arrays.dtypes.html
            # https://docs.h5py.org/en/stable/strings.html#what-about-numpy-s-u-type
            # https://docs.h5py.org/en/stable/strings.html
            if isinstance(v, np.ndarray):
                if v.dtype.kind in {"U", "S"}:
                    dtype = h5py.string_dtype()

                    file.create_dataset(
                        parent_key + "/" + k,
                        data=[
                            *v,
                        ],
                        maxshape=(None,) + v.shape[1:],
                        dtype=dtype,
                    )
                else:
                    dtype = v.dtype

                    file.create_dataset(
                        parent_key + "/" + k,
                        data=v,
                        maxshape=(None,) + v.shape[1:],
                        dtype=dtype,
                    )
            else:
                # No maxshape = it's a fixed-size scalar dataset
                file.create_dataset(
                    parent_key + "/" + k,
                    data=v,
                    dtype=np.asarray(v).dtype,
                )


def create_single_dict_to_hdf5(file, d, parent_key=""):
    """Create new datasets (used for kernels group overwrite)"""
    for k, v in d.items():
        if isinstance(v, dict):
            create_single_dict_to_hdf5(file, v, parent_key + "/" + k)
        else:
            if isinstance(v, np.ndarray) and v.dtype.kind in {"U", "S"}:
                dtype = h5py.string_dtype()
                file.create_dataset(
                    parent_key + "/" + k,
                    data=[*v],
                    dtype=dtype,
                )
            else:
                file.create_dataset(
                    parent_key + "/" + k,
                    data=v,
                    dtype=np.asarray(v).dtype,
                )


def overwrite_dict_to_hdf5(file, d, parent_key=""):
    """Overwrite existing datasets (used for kernels group overwrite)"""
    for k, v in d.items():
        current_key = parent_key + "/" + k
        if isinstance(v, dict):
            overwrite_dict_to_hdf5(file, v, current_key)
        else:
            if isinstance(v, np.ndarray) and v.dtype.kind in {"U", "S"}:
                file[current_key][...] = [*v]
            else:
                file[current_key][...] = v


class Saver:
    r"""enable to regularly save the progression of the Markov chain to a ``.hdf5`` file"""

    __slots__ = (
        "N",
        "D",
        "D_sampling",
        "L",
        "results_path",
        "batch_size",
        "freq_save",
        "scaler",
        "t_last_save",
        "t_last_init",
        "next_batch_size",
        "final_next_batch_size",
        "memory",
        "save_forward_map_evals",
        "list_idx_sampling",
    )

    def __init__(
        self,
        N: int,
        D: int,
        D_sampling: int,
        L: int,
        scaler: Scaler,
        results_path: str = "",
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
            total number of physical parameters
        D_sampling : int
            number of physical parameters that are optimized / sampled
        L : int
            number of observed lines
        scaler : Scaler
            contains the transformation of the Theta values from their scaled space (in which the sampling happens) to their natural space
        results_path : str
            path towards the ``.hdf5`` output file, by default ""
        batch_size : int, optional
            number of iterations between two saves on file, by default None
        freq_save : int, optional
            save one sample in every (freq_save). Used to save disk space., by default 1
        save_forward_map_evals: bool, optional
            wether to save the forward model evaluations and gradients, by default False
        list_idx_sampling : Optional[List[int]], optional
            contains the indices of the physical parameters to be sampled
        """
        self.N = N
        r"""int: total number of pixels to reconstruct"""
        self.D = D
        r"""int: total number of physical parameters"""
        self.D_sampling = D_sampling
        r"""int: number of physical parameters that are optimized / sampled"""
        self.L = L
        r"""int: number of observed lines"""

        if list_idx_sampling is None:
            list_idx_sampling = np.arange(self.D)

        self.list_idx_sampling = list_idx_sampling
        r"""1D np.ndarray: contains the indices of the physical parameters to be sampled"""

        self.results_path = results_path
        r"""str: path towards the ``.hdf5`` output file"""
        if len(results_path) > 0 and not (os.path.isdir(results_path)):
            os.mkdir(results_path)

        self.batch_size = batch_size
        r"""int: frequency of saves, i.e., "every ``batch_size`` new iterates to be saved, the memory is saved to an ``.hdf5`` file and re-initialized"""

        self.freq_save = freq_save
        r"""int: frequency of saved iterates during the run (1 means that every iteration is saved)"""

        self.scaler = scaler
        r"""Scaler: contains the transformation of the Theta values from their natural space to their scaled space (in which the sampling happens)"""

        # these two attributes are initialized by initialize_memory
        # updated during sampling
        self.t_last_save = 0
        r"""int: time index of the last save of the memory to ``.hdf5`` file"""
        self.t_last_init = 0
        r"""int: time index of the last memory initialization"""
        self.next_batch_size = 0
        r"""int: number of iterates to be stored in the next batch, i.e., until next save to file"""
        self.final_next_batch_size = 0
        r"""int: """
        self.memory = dict()
        """dict[str, Union[float, np.ndarray]]: stores the values before saving them to file"""

        self.save_forward_map_evals = save_forward_map_evals
        r"""bool: wether to save the forward model evaluations and gradients"""

    def set_results_path(self, results_path: str) -> None:
        r"""sets the path of the ``.hdf5`` file to a new value

        Parameters
        ----------
        results_path : str
            path towards the ``.hdf5`` output file
        """
        self.results_path = results_path
        if len(results_path) > 0 and not (os.path.isdir(results_path)):
            os.mkdir(results_path)
        return

    def check_need_to_save(self, t: int) -> bool:
        """checks wether or not the memory should be saved to a ``.hdf5`` file

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
        Theta: np.ndarray,
        forward_map_evals: dict = dict(),
        nll_utils: dict = dict(),
        dict_objective: dict = dict(),
        additional_sampling_log: dict = dict(),
    ) -> None:
        r"""initializes the memory with the correct shapes"""
        pass

    @abc.abstractmethod
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
        """updates the memory with new information. All of the potential entries are optional except for the current iterate."""
        pass

    def save_to_file(self):
        """Saves the current memory content to a ``.hdf5`` file"""
        h5_file = os.path.join(self.results_path, "mc_chains.hdf5")

        if self.t_last_init == 1:  # if first writing
            with h5py.File(
                h5_file,
                "w",
            ) as f:
                save_dict_to_hdf5(f, self.memory)

        else:  # append data to already created file
            with h5py.File(
                h5_file,
                "a",
            ) as f:

                def append_dict_to_hdf5(file, d, parent_key=""):
                    for k, v in d.items():
                        current_key = parent_key + "/" + k
                        # Special handling for kernels and model_check groups: overwrite instead of append
                        if k in ["kernels", "model_check"] and parent_key == "":
                            overwrite_dict_to_hdf5(file, v, current_key)
                        elif isinstance(v, dict):
                            append_dict_to_hdf5(file, v, current_key)
                        else:
                            file[current_key].resize(
                                file[current_key].shape[0] + self.final_next_batch_size,
                                axis=0,
                            )
                            file[current_key][-self.final_next_batch_size :] = v

                append_dict_to_hdf5(f, self.memory)

        self.memory = dict()

    def save_additional(
        self,
        list_arrays: List[np.ndarray],
        list_names: List[str],
    ) -> None:
        r"""saves additional content to a ``.hdf5`` file

        Parameters
        ----------
        list_arrays : List[np.ndarray]
            list of the arrays to be saved
        list_names : List[str]
            list of names for the arrays to be saved in the ``.hdf5`` file
        """
        assert len(list_names) == len(list_arrays)

        with h5py.File(
            os.path.join(self.results_path, "mc_chains.hdf5"),
            "a",
        ) as f:
            for name, array_ in zip(list_names, list_arrays):
                f.create_dataset(name, data=array_)

        return
