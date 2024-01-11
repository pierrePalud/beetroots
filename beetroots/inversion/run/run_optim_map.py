import copy
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

from beetroots.inversion.results.results_optim_map import ResultsExtractorOptimMAP
from beetroots.inversion.results.results_optim_mle import ResultsExtractorOptimMLE
from beetroots.inversion.run.abstract_run import Run
from beetroots.modelling.posterior import Posterior
from beetroots.sampler.abstract_sampler import Sampler
from beetroots.sampler.saver.abstract_saver import Saver
from beetroots.space_transform.abstract_transform import Scaler


class RunOptimMAP(Run):
    r"""class that runs inversions using an optimization approach, considering all pixels / components at once"""

    __slots__ = ("path_data_csv_out", "max_workers")

    def __init__(self, path_data_csv_out: str, max_workers: int):
        r"""

        Parameters
        ----------
        path_data_csv_out : str
            path to the folder where results are to be saved
        max_workers : int
            max number of workers that can be used to run the inversion
        """
        self.path_data_csv_out = path_data_csv_out
        r"""str: path to the folder where results are to be saved"""

        self.max_workers = max_workers
        r"""int: max number of workers that can be used to run the inversion"""

    def prepare_run(
        self,
        dict_posteriors: dict[str, Posterior],
        path_raw: str,
        N_runs: int,
        scaler: Scaler,
        start_from: Optional[str] = None,
        path_csv_mle: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        r"""prepares the run in two ways :

        * step 1 : creates empty folders to save the run results
        * step 2 : reads ``Theta_0`` if specified (as the MLE)

        Parameters
        ----------
        dict_posteriors : dict[str, Posterior]
            dictionary of posterior distributions
        path_raw : str
            path to the folders where the ``.hdf5`` files are to be stored
        N_runs : int
            number of independent optimization runs to run per posterior distribution
        scaler : Scaler
            contains the transformation of the Theta values from their natural space to their scaled space (in which the sampling happens)
        start_from : Optional[str]
            point at which the inversion will start, must be in [None, "MLE"]. For None, a random value is drawn uniformly in the scaled hypercube.
        path_csv_mle : Optional[str]
            path to the csv file containing the already estimated MLE

        Returns
        -------
        Optional[np.ndarray]
            starting point of the  (in scaled space) inversion, ``Theta_0``, if specified. Otherwise ``None``.
        """
        # create empty folders to save the run results
        for seed in range(N_runs):
            for model_name in list(dict_posteriors.keys()):
                folder_path = f"{path_raw}/{model_name}/optim_MAP_{seed}"

                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)

        # read Theta_0 if needed
        assert start_from in ["MLE", "MAP", None]

        if start_from == "MLE":
            result_extractor = ResultsExtractorOptimMLE(path_csv_mle)
            Theta_0, _ = result_extractor.read_estimator(model_name)
            Theta_0 = scaler.from_lin_to_scaled(Theta_0)

        else:
            Theta_0 = None

        return Theta_0

    def run(
        self,
        dict_posteriors: dict[str, Posterior],
        sampler_: Sampler,
        saver_: Saver,
        N_runs: int,
        max_iter: int,
        path_raw: str,
        Theta_0: Optional[np.ndarray] = None,
        freq_save: int = 1,
        can_run_in_parallel: bool = True,
    ) -> None:
        r"""runs the inversion

        Parameters
        ----------
        dict_posteriors : dict[str, Posterior]
            dictionary of posterior distributions
        sampler_ : Sampler
            optimizer
        saver_ : Saver
            object responsible for progressively saving the optimization run data during the run
        N_runs : int
            number of independent optimization runs to run per posterior distribution
        max_iter : int
            total duration of an optimization run
        path_raw : str
            path to the folders where the ``.hdf5`` files are to be stored
        Theta_0 : Optional[np.ndarray], optional
            starting point, by default None
        freq_save : int, optional
            frequency of saved iterates during the run (1 means that every iteration is saved), by default 1
        can_run_in_parallel : bool, optional
            wether the inversion can be run in parallel (may cause difficulties for forward maps based on neural networks run on GPU), by default True
        """
        global _run_one_simulation_optim_map_all_pixels

        def _run_one_simulation_optim_map_all_pixels(dict_input: dict) -> dict:
            model_name = dict_input["model_name"]
            seed = dict_input["seed"]

            folder_path = f"{path_raw}/{model_name}/optim_MAP_{seed}"
            saver_seed = copy.deepcopy(saver_)
            saver_seed.set_results_path(folder_path)

            sampler_seed = copy.deepcopy(sampler_)

            tps0 = time.time()

            sampler_seed.sample(
                dict_posteriors[model_name],
                saver=saver_seed,
                max_iter=max_iter,
                Theta_0=Theta_0,
            )
            # return input dict with duration information
            dict_output = {
                "seed": seed,
                "model_name": model_name,
                "total_duration": time.time() - tps0,
            }
            return dict_output

        # * global function
        print("starting optimization MAP")
        list_params = [
            {"seed": seed, "model_name": model_name}
            for seed in range(N_runs)
            for model_name in list(dict_posteriors.keys())
        ]
        if can_run_in_parallel:
            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=mp.get_context("fork")
            ) as p:
                list_simulations_durations = list(
                    p.map(_run_one_simulation_optim_map_all_pixels, list_params)
                )
        else:
            # * non parallel version
            list_simulations_durations = []
            for params in list_params:
                duration = _run_one_simulation_optim_map_all_pixels(params)
                list_simulations_durations.append(duration)

        df_results_sampling = pd.DataFrame(list_simulations_durations)
        filename = f"{self.path_data_csv_out}/durations_optim_MAP.csv"
        df_results_sampling.to_csv(filename)
        print("optimization MAP done\n")
        return

    def main(
        self,
        dict_posteriors: dict[str, Posterior],
        sampler_: Sampler,
        saver_: Saver,
        scaler: Scaler,
        N_runs: int,
        max_iter: int,
        path_raw: str,
        path_csv_mle: Optional[str] = None,
        start_from: Optional[str] = None,
        freq_save: int = 1,
        can_run_in_parallel: bool = True,
    ) -> None:
        r"""sequentially calls ``prepare_run`` and ``run``

        Parameters
        ----------
        dict_posteriors : dict[str, Posterior]
            dictionary of posterior distributions
        sampler_ : Sampler
            optimizer
        saver_ : Saver
            object responsible for progressively saving the optimization run data during the run
        scaler : Scaler
            contains the transformation of the Theta values from their natural space to their scaled space (in which the sampling happens)
        N_runs : int
            number of independent optimization runs to run per posterior distribution
        max_iter : int
            total duration of an optimization run
        path_raw : str
            path to the folders where the ``.hdf5`` files are to be stored
        path_csv_mle : Optional[str]
            path to the csv file containing the already estimated MLE
        start_from : Optional[str], optional
            _description_, by default None
        freq_save : int, optional
            frequency of saved iterates during the run (1 means that every iteration is saved), by default 1
        can_run_in_parallel : bool, optional
            wether the inversion can be run in parallel (may cause difficulties for forward maps based on neural networks run on GPU), by default True
        """
        Theta_0 = self.prepare_run(
            dict_posteriors,
            path_raw,
            N_runs,
            scaler,
            path_csv_mle,
            start_from,
        )
        self.run(
            dict_posteriors,
            sampler_,
            saver_,
            N_runs,
            max_iter,
            path_raw,
            Theta_0,
            freq_save,
            can_run_in_parallel,
        )
        return
