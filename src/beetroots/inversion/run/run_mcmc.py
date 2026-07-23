import copy
import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from beetroots.inversion.results.results_optim_map_mygibbssampler import (
    ResultsExtractorOptimMAP,
)
from beetroots.inversion.results.results_optim_mle_mygibbssampler import (
    ResultsExtractorOptimMLE,
)
from beetroots.inversion.run.abstract_run import Run
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.modelling.target_distribution.posterior import Posterior
from beetroots.sampler.abstract_sampler import Sampler
from beetroots.sampler.saver.abstract_saver import Saver
from beetroots.space_transform.abstract_transform import Scaler


class RunMCMC(Run):
    r"""class that runs inversions using a sampling approach"""

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

    def run(
        self,
        dict_posteriors: dict[str, Posterior],
        samplers_: List[Sampler],
        saver_: Saver,
        N_runs: int,
        max_iter: int,
        T_BI: int,
        path_raw: str,
        Vars_0: dict[int, dict[str, np.ndarray]],
        # sample_regu_weights: bool = True,
        # T_BI_reguweights: Optional[int] = None,
        #
        regu_spatial_N0: Union[int, float] = np.inf,
        regu_spatial_scale: Optional[float] = 1.0,
        regu_spatial_vmin: Optional[float] = 1e-8,
        regu_spatial_vmax: Optional[float] = 1e8,
        #
        freq_save: int = 1,
        can_run_in_parallel: bool = True,
    ) -> None:
        r"""runs the inversion

        Parameters
        ----------
        dict_posteriors : dict[str, Posterior]
            dictionary of posterior distributions
        sampler_ : Sampler
            sampler to be used to generate the Markov chain(s)
        saver_ : Saver
            object responsible for progressively saving the Markov chain data during the run
        N_runs : int
            number of independent Markov chains to run per posterior distribution
        max_iter : int
            total duration of a Markov chain
        T_BI : int
            duration of the `Burn-in` phase
        path_raw : str
            path to the folders where the ``.hdf5`` files are to be stored
        Vars_0 : dict[int, dict[str, np.ndarray]]
            starting point. Dict with keys being the seed, and values being dict with keys being the variable names and values being the corresponding arrays.
        regu_spatial_N0 : Union[int, float], optional
            number of iterations defining the initial update phase (for spatial regularization weight optimization). np.inf means that the optimization phase never starts, and that the weight optimization is not applied. by default np.inf
        regu_spatial_scale : Optional[float], optional
            scale parameter involved in the definition of the projected gradient
            step size (for spatial regularization weight optimization). by default 1.0
        regu_spatial_vmin : Optional[float], optional
            lower limit of the admissible interval (for spatial regularization weight optimization), by default 1e-8
        regu_spatial_vmax : Optional[float], optional
            upper limit of the admissible interval (for spatial regularization weight optimization), by default 1e8
        freq_save : int, optional
            frequency of saved iterates during the run (1 means that every iteration is saved), by default 1
        can_run_in_parallel : bool, optional
            wether the inversion can be run in parallel (may cause difficulties for forward maps based on neural networks run on GPU), by default True
        """
        global _run_one_simulation_mcmc_all_pixels

        def _run_one_simulation_mcmc_all_pixels(dict_input: dict) -> dict:
            model_name = dict_input["model_name"]
            seed = dict_input["seed"]

            folder_path = f"{path_raw}/{model_name}/mcmc_{seed}"
            saver_seed = copy.deepcopy(saver_)
            saver_seed.set_results_path(folder_path)

            sampler_seed = dict_input["sampler"]
            Vars_0_seed = dict_input["Vars_0"]

            tps0 = time.time()

            sampler_seed.sample(
                dict_posteriors[model_name],
                saver=saver_seed,
                max_iter=max_iter,
                Vars_0=Vars_0_seed,
                #
                # sample_regu_weights=sample_regu_weights,
                # T_BI_reguweights=T_BI_reguweights,
                regu_spatial_N0=regu_spatial_N0,
                regu_spatial_scale=regu_spatial_scale,
                regu_spatial_vmin=regu_spatial_vmin,
                regu_spatial_vmax=regu_spatial_vmax,
                #
                T_BI=T_BI,
            )

            # return input dict with duration information
            total_duration = time.time() - tps0

            dict_output = {
                "seed": seed,
                "model_name": model_name,
                "total_duration": total_duration,
            }
            return dict_output

        # *
        print("starting sampling")
        list_params = [
            {
                "seed": seed,
                "model_name": model_name,
                "sampler": samplers_[seed],
                "Vars_0": Vars_0[seed],
            }
            for seed in range(N_runs)
            for model_name in list(dict_posteriors.keys())
        ]
        if can_run_in_parallel:
            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=mp.get_context("fork")
            ) as p:
                list_simulations_durations = list(
                    p.map(_run_one_simulation_mcmc_all_pixels, list_params)
                )
        else:
            list_simulations_durations = [
                _run_one_simulation_mcmc_all_pixels(params) for params in list_params
            ]

        # save run durations
        df_results_sampling = pd.DataFrame(list_simulations_durations)
        filename = f"{self.path_data_csv_out}/durations_MCMC.csv"
        df_results_sampling.to_csv(filename)
        print("sampling done\n")
        return

    def main(
        self,
        dict_posteriors: dict[str, Posterior],
        samplers_: List[Sampler],
        saver_: Saver,
        scaler: Scaler,
        N_runs: int,
        max_iter: int,
        T_BI: int,
        path_raw: str,
        Vars_0: dict,
        # sample_regu_weights: bool = True,
        # T_BI_reguweights: Optional[int] = None,
        #
        regu_spatial_N0: Union[int, float] = np.inf,
        regu_spatial_scale: Optional[float] = 1.0,
        regu_spatial_vmin: Optional[float] = 1e-8,
        regu_spatial_vmax: Optional[float] = 1e8,
        #
        freq_save: int = 1,
        can_run_in_parallel: bool = True,
    ) -> None:
        r"""sequentially calls ``prepare_run`` and ``run``

        Parameters
        ----------
        dict_posteriors : dict[str, Posterior]
            dictionary of posterior distributions
        sampler_ : Sampler
            sampler to be used to generate the Markov chain(s)
        saver_ : Saver
            object responsible for progressively saving the Markov chain data during the run
        scaler : Scaler
            contains the transformation of the Theta values from their natural space to their scaled space (in which the sampling happens)
        N_runs : int
            number of independent Markov chains to run per posterior distribution
        max_iter : int
            total duration of a Markov chain
        T_BI : int
            duration of the `Burn-in` phase
        path_raw : str
            path to the folders where the ``.hdf5`` files are to be stored
        path_csv_mle : Optional[str]
            path to the csv file containing the already estimated MLE
        path_csv_map : Optional[str]
            path to the csv file containing the already estimated MAP
        Vars_0 : dict[int, dict[str, np.ndarray]]
            point at which the inversion will start. Dict with keys being the seed, and values being dict with keys being the variable names and values being the corresponding arrays.
        regu_spatial_N0 : Union[int, float], optional
            number of iterations defining the initial update phase (for spatial regularization weight optimization). np.inf means that the optimization phase never starts, and that the weight optimization is not applied. By default np.inf
        regu_spatial_scale : Optional[float], optional
            scale parameter involved in the definition of the projected gradient
            step size (for spatial regularization weight optimization). by default 1.0
        regu_spatial_vmin : Optional[float], optional
            lower limit of the admissible interval (for spatial regularization weight optimization), by default 1e-8
        regu_spatial_vmax : Optional[float], optional
            upper limit of the admissible interval (for spatial regularization weight optimization), by default 1e8
        freq_save : int, optional
            frequency of saved iterates during the run (1 means that every iteration is saved), by default 1
        can_run_in_parallel : bool, optional
            wether the inversion can be run in parallel (may cause difficulties for forward maps based on neural networks run on GPU), by default True
        """

        self.run(
            dict_posteriors,
            samplers_,
            saver_,
            N_runs=N_runs,
            max_iter=max_iter,
            T_BI=T_BI,
            path_raw=path_raw,
            Vars_0=Vars_0,
            #
            regu_spatial_N0=regu_spatial_N0,
            regu_spatial_scale=regu_spatial_scale,
            regu_spatial_vmin=regu_spatial_vmin,
            regu_spatial_vmax=regu_spatial_vmax,
            #
            freq_save=freq_save,
            can_run_in_parallel=can_run_in_parallel,
        )
        return
