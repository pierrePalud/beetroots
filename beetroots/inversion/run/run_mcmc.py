import copy
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Union

import numpy as np
import pandas as pd

from beetroots.inversion.results.results_optim_map import ResultsExtractorOptimMAP
from beetroots.inversion.results.results_optim_mle import ResultsExtractorOptimMLE
from beetroots.inversion.run.abstract_run import Run
from beetroots.sampler.abstract_sampler import Sampler
from beetroots.sampler.saver.abstract_saver import Saver
from beetroots.space_transform.abstract_transform import Scaler


class RunMCMC(Run):

    __slots__ = ("path_data_csv_out", "max_workers")

    def __init__(self, path_data_csv_out: str, max_workers: int):
        self.path_data_csv_out = path_data_csv_out
        self.max_workers = max_workers

    def prepare_run(
        self,
        dict_posteriors: dict,
        path_raw: str,
        N_runs: int,
        scaler: Scaler,
        start_from: Optional[str],
        path_csv_mle: Optional[str],
        path_csv_map: Optional[str],
    ) -> Optional[np.ndarray]:
        # create empty folders to save the run results
        for seed in range(N_runs):
            for model_name in list(dict_posteriors.keys()):
                folder_path = f"{path_raw}/{model_name}/mcmc_{seed}"

                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)

        # read x0 if needed
        assert start_from in ["MLE", "MAP", None]

        if start_from == "MLE":
            x0, _ = ResultsExtractorOptimMLE.read_estimator(
                path_csv_mle,
                model_name,
            )
            x0 = scaler.from_lin_to_scaled(x0)

        elif start_from == "MAP":
            x0, _ = ResultsExtractorOptimMAP.read_estimator(
                path_csv_map,
                model_name,
            )
            x0 = scaler.from_lin_to_scaled(x0)

        else:
            x0 = None

        return x0

    def run(
        self,
        dict_posteriors: dict,
        sampler_: Sampler,
        saver_: Saver,
        N_runs: int,
        max_iter: int,
        T_BI: int,
        path_raw: str,
        x0: Optional[np.ndarray] = None,
        # sample_regu_weights: bool = True,
        # T_BI_reguweights: Optional[int] = None,
        #
        regu_spatial_N0: Union[int, float] = np.infty,
        regu_spatial_scale: Optional[float] = 1.0,
        regu_spatial_vmin: Optional[float] = 1e-8,
        regu_spatial_vmax: Optional[float] = 1e8,
        #
        freq_save: int = 1,
        can_run_in_parallel: bool = True,
    ) -> None:
        global _run_one_simulation_mcmc_all_pixels

        def _run_one_simulation_mcmc_all_pixels(dict_input: dict) -> dict:
            model_name = dict_input["model_name"]
            seed = dict_input["seed"]

            folder_path = f"{path_raw}/{model_name}/mcmc_{seed}"
            saver_seed = copy.deepcopy(saver_)
            saver_seed.set_results_path(folder_path)

            sampler_seed = copy.deepcopy(sampler_)

            tps0 = time.time()

            sampler_seed.sample(
                dict_posteriors[model_name],
                saver=saver_seed,
                max_iter=max_iter,
                x0=x0,
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
            {"seed": seed, "model_name": model_name}
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
        dict_posteriors: dict,
        sampler_: Sampler,
        saver_: Saver,
        scaler: Scaler,
        N_runs: int,
        max_iter: int,
        T_BI: int,
        path_raw: str,
        path_csv_mle: Optional[str],
        path_csv_map: Optional[str],
        start_from: Optional[str],
        # sample_regu_weights: bool = True,
        # T_BI_reguweights: Optional[int] = None,
        #
        regu_spatial_N0: Union[int, float] = np.infty,
        regu_spatial_scale: Optional[float] = 1.0,
        regu_spatial_vmin: Optional[float] = 1e-8,
        regu_spatial_vmax: Optional[float] = 1e8,
        #
        freq_save: int = 1,
        can_run_in_parallel: bool = True,
    ) -> None:
        x0 = self.prepare_run(
            dict_posteriors,
            path_raw,
            N_runs,
            scaler,
            start_from=start_from,
            path_csv_mle=path_csv_mle,
            path_csv_map=path_csv_map,
        )
        self.run(
            dict_posteriors,
            sampler_,
            saver_,
            N_runs=N_runs,
            max_iter=max_iter,
            T_BI=T_BI,
            path_raw=path_raw,
            x0=x0,
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
