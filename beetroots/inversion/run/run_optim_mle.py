import copy
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

from beetroots.inversion.run.abstract_run import Run
from beetroots.modelling.posterior import Posterior
from beetroots.sampler.abstract_sampler import Sampler
from beetroots.sampler.saver.abstract_saver import Saver
from beetroots.space_transform.abstract_transform import Scaler

# TODO : finish this class


class RunOptimMLE(Run):
    r"""class that runs inversions using an optimization approach, considering all pixels independently.

    .. warning::
        Unfinished class.
    """

    def __init__(self, path_data_csv_out: str, max_workers: int, N: int):
        r"""

        Parameters
        ----------
        path_data_csv_out : str
            path to the folder where results are to be saved
        max_workers : int
            max number of workers that can be used to run the inversion
        N : int
            number of pixels, i.e., of optimization problems to solve
        """
        self.path_data_csv_out = path_data_csv_out
        r"""str: path to the folder where results are to be saved"""

        self.max_workers = max_workers
        r"""int: max number of workers that can be used to run the inversion"""

        self.N = N
        r"""int: number of pixels, i.e., of optimization problems to solve"""

    def prepare_run(
        self,
        dict_posteriors: dict[str, Posterior],
        path_raw: str,
        N_runs: int,
    ) -> None:
        r"""prepares the run by creating empty folders to save the run results

        Parameters
        ----------
        dict_posteriors : dict[str, Posterior]
            dictionary of posterior distributions
        path_raw : str
            path to the folders where the ``.hdf5`` files are to be stored
        N_runs : int
            number of independent optimization runs to run per component :math:`n` / pixel
        """
        for model_name in list(dict_posteriors.keys()):
            for seed in range(N_runs):
                folder_path_inter = f"{path_raw}/{model_name}/"
                folder_path_inter += f"optim_MLE_{seed}"

                for n in range(self.N):
                    folder_path = f"{folder_path_inter}/pixel_{n}"
                    for path_ in [folder_path_inter, folder_path]:
                        if not os.path.isdir(path_):
                            os.mkdir(path_)

        return

    def run(
        self,
        dict_posteriors: dict[str, Posterior],
        sampler_: Sampler,
        saver_: Saver,
        N_runs: int,
        max_iter: int,
        path_raw: str,
        freq_save: int = 1,
    ) -> None:
        r"""runs the ``N`` inversions.

        .. warning::
            Unfinished method.

        Parameters
        ----------
        dict_posteriors : dict[str, Posterior]
            _dictionary of posterior distributions
        sampler_ : Sampler
            sampler to be used to generate the Markov chain(s)
        saver_ : Saver
            object responsible for progressively saving the Markov chain data during the run
        N_runs : int
            number of independent Markov chains to run per posterior distribution
        max_iter : int
            total duration of a Markov chain
        path_raw : str
            path to the folders where the ``.hdf5`` files are to be stored
        freq_save : int, optional
            frequency of saved iterates during the run (1 means that every iteration is saved), by default 1
        """
        # TODO: change this method using the old method run_optimization_MLE
        global _run_one_simulation_optim_mle_all_pixels

        def _run_one_simulation_optim_mle_all_pixels(dict_input: dict) -> dict:
            model_name = dict_input["model_name"]
            seed = dict_input["seed"]

            folder_path = f"{path_raw}/{model_name}/optim_MLE_{seed}"
            saver_seed = copy.deepcopy(saver_)
            saver_seed.set_results_path(folder_path)

            sampler_seed = copy.deepcopy(sampler_)

            tps0 = time.time()

            sampler_seed.sample(
                dict_posteriors[model_name],
                saver=saver_seed,
                max_iter=max_iter,
            )
            # return input dict with duration information
            dict_output = {
                "seed": seed,
                "model_name": model_name,
                "total_duration": time.time() - tps0,
            }
            return dict_output

        # * global function
        print("starting optimization MLE")
        list_params = [
            {"seed": seed, "model_name": model_name}
            for seed in range(N_runs)
            for model_name in list(dict_posteriors.keys())
        ]
        # TODO: multiprocessing and cuda?
        # with ProcessPoolExecutor(
        #     max_workers=self.max_workers, mp_context=mp.get_context("fork")
        # ) as p:
        #     list_simulations_durations = list(
        #         p.map(_run_one_simulation_optim_map_all_pixels, list_params)
        #     )

        # * non parallel version
        list_simulations_durations = []
        for params in list_params:
            duration = _run_one_simulation_optim_map_all_pixels(params)
            list_simulations_durations.append(duration)

        df_results_sampling = pd.DataFrame(list_simulations_durations)
        df_results_sampling.to_csv(f"{self.path_data_csv_out}/durations_optim_MLE.csv")
        print("optimization MLE done\n")
        return

    def main(
        self,
        dict_posteriors: dict[str, Posterior],
        sampler_: Sampler,
        saver_: Saver,
        N_runs: int,
        max_iter: int,
        path_raw: str,
        freq_save: int = 1,
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
        N_runs : int
            number of independent optimization runs to run per posterior distribution
        max_iter : int
            total duration of an optimization run
        path_raw : str
            path to the folders where the ``.hdf5`` files are to be stored
        freq_save : int, optional
            frequency of saved iterates during the run (1 means that every iteration is saved), by default 1
        """
        self.prepare_run(dict_posteriors, path_raw, N_runs)
        self.run(
            dict_posteriors,
            sampler_,
            saver_,
            N_runs,
            max_iter,
            path_raw,
            freq_save,
        )
        return

    #! Old method
    # def run_optimization_MLE(
    #     self,
    #     my_sampler_params,
    #     freq_save: int = 1,
    # ) -> None:

    #     global _run_one_simulation_one_pixel

    #     my_sampler_params.save_to_file(self.path_data_csv_in, "algo_params_optim_MLE.csv")

    #     def _run_one_simulation_one_pixel(dict_input: dict) -> dict:
    #         n = dict_input["n"]
    #         #! change seed between pixels
    #         #! interest: avoid starting at the same point for all n
    #         seed_folder = dict_input["seed"]
    #         seed_run = (dict_input["seed"] + 1) * (n + 1)
    #         model_name = dict_input["model_name"]

    #         idx_model = list(self.dict_posteriors.keys()).index(model_name)
    #         params = (self.list_mixing_model_params + self.list_gaussian_approx_params)[
    #             idx_model
    #         ]

    #         folder_path_inter = f"{path_raw}/{model_name}/opti_MLE_{seed_folder}"
    #         folder_path = f"{folder_path_inter}/pixel_{n}"

    #         saver_ = MySaver(
    #             1,
    #             self.D,
    #             self.L,
    #             folder_path,
    #             self.batch_size,
    #             freq_save=freq_save_mle,
    #         )
    #         sampler = MySampler(
    #             my_sampler_params,
    #             self.D,
    #             self.D_no_kappa,
    #             self.L,
    #             1,
    #             np.random.default_rng(seed_run),
    #         )
    #         y_pix = self.y[n, :].reshape((1, self.L))
    #         omega_pix = self.omega[n, :].reshape((1, self.L))

    #         if "mixing" in model_name:
    #             sigma_a_pix = self.sigma_a[n, :].reshape((1, self.L))
    #             sigma_m_pix = self.sigma_m[n, :].reshape((1, self.L))

    #             likelihood_1pix = MixingModelsLikelihood(
    #                 self.forward_map,
    #                 self.D,
    #                 self.L,
    #                 1,
    #                 y_pix,
    #                 sigma_a_pix,
    #                 sigma_m_pix,
    #                 omega_pix,
    #                 path_transition_params=params["path_transition_params"],
    #             )
    #         elif "gaussian_approx" in model_name:
    #             is_raw = params * 1
    #             if is_raw:
    #                 m_a_pix = 0
    #                 s_a_pix = self.sigma_a[n, :].reshape((1, self.L))
    #             else:
    #                 m_a = (np.exp(self.sigma_m ** 2 / 2) - 1) * self.y
    #                 s_a = np.sqrt(
    #                     self.y ** 2
    #                     * np.exp(self.sigma_m ** 2)
    #                     * (np.exp(self.sigma_m ** 2) - 1)
    #                     + self.sigma_a ** 2
    #                 )

    #                 m_a_pix = m_a[n, :].reshape((1, self.L))
    #                 s_a_pix = s_a[n, :].reshape((1, self.L))

    #             likelihood_1pix = CensoredGaussianLikelihood(
    #                 self.forward_map,
    #                 self.D,
    #                 self.L,
    #                 1,
    #                 y_pix,
    #                 s_a_pix,
    #                 omega_pix,
    #                 bias=m_a_pix,
    #             )
    #         else:
    #             raise ValueError(f"invalid model name : {model_name}")

    #         posterior_1pix = Posterior(
    #             self.D,
    #             self.L,
    #             1,
    #             likelihood=likelihood_1pix,
    #             prior_spatial=None,
    #             prior_indicator=self.prior_indicator_1pix,
    #         )

    #         tps0 = time.time()
    #         sampler.sample(
    #             posterior_1pix,
    #             saver=saver_,
    #             max_iter=self.T_OPTI_MLE,
    #             # sample_regu_weights=False,
    #             disable_progress_bar=True,
    #         )

    #         # return input dict with duration information
    #         total_duration = time.time() - tps0

    #         dict_output = {
    #             "n": n,
    #             "seed": seed_folder,
    #             "seed_run": seed_run,
    #             "model_name": model_name,
    #             "total_duration": total_duration,
    #         }
    #         return dict_output

    #     # * core function
    #     print("starting optimization MLE")
    #     list_params = [
    #         {"n": n, "seed": seed, "model_name": model_name}
    #         for n in range(self.N)
    #         for seed in range(N_runs)
    #         for model_name in list(self.dict_posteriors.keys())
    #     ]

    #     n_models = len(self.list_gaussian_approx_params)
    #     n_models += len(self.list_mixing_model_params)

    #     with ProcessPoolExecutor(
    #         max_workers=self.max_workers, mp_context=mp.get_context("fork")
    #     ) as p:
    #         list_simulations_durations = list(
    #             tqdm(
    #                 p.map(
    #                     _run_one_simulation_one_pixel,
    #                     list_params,
    #                 ),
    #                 total=self.N * N_runs * n_models,
    #             )
    #         )
    #     df_results_optim_mle = pd.DataFrame(list_simulations_durations)
    #     df_results_optim_mle.to_csv(
    #         f"{self.path_data_csv_out}/durations_optim_MLE.csv"
    #     )
    #     print("optimization MLE done")
    #     return
