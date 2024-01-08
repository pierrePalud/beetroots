import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.inversion.results.utils.mc_utils import ess, histograms
from beetroots.space_transform.abstract_transform import Scaler


class ResultsMC(ResultsUtil):

    __slots__ = (
        "model_name",
        "chain_type",
        "path_img",
        "path_data_csv_out_mcmc",
        "max_workers",
        "N_MCMC",
        "T_MC",
        "T_BI",
        "freq_save",
        "effective_T_BI",
        "lower_bounds_lin",
        "upper_bounds_lin",
        "N",
        "D",
        "list_names",
    )

    def __init__(
        self,
        model_name: str,
        chain_type: str,
        path_img: str,
        path_data_csv_out_mcmc: str,
        max_workers: int,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        freq_save: int,
        N: int,
        list_idx_sampling: List,
        list_fixed_values_scaled: List,
        lower_bounds_lin: Union[np.ndarray, List[float]],
        upper_bounds_lin: Union[np.ndarray, List[float]],
        list_names: List[str],
    ):
        assert chain_type in ["mcmc", "optim_map", "optim_mle"]

        self.model_name = model_name
        self.chain_type = chain_type
        self.path_img = path_img
        self.path_data_csv_out_mcmc = path_data_csv_out_mcmc
        self.max_workers = max_workers

        self.N_MCMC = N_MCMC
        self.T_MC = T_MC
        self.T_BI = T_BI
        self.freq_save = freq_save
        self.effective_T_BI = T_BI // freq_save

        if isinstance(lower_bounds_lin, list):
            lower_bounds_lin = np.array(lower_bounds_lin)
        if isinstance(upper_bounds_lin, list):
            upper_bounds_lin = np.array(upper_bounds_lin)

        self.lower_bounds_lin = lower_bounds_lin
        self.upper_bounds_lin = upper_bounds_lin

        self.list_idx_sampling = list_idx_sampling
        self.list_fixed_values_scaled = list_fixed_values_scaled

        self.N = N
        self.D = upper_bounds_lin.size
        self.D_sampling = len(self.list_idx_sampling)

        self.list_names = list_names

    def read_data(self):
        pass

    def create_folders(self) -> Tuple[str]:
        folder_path_mc = f"{self.path_img}/mc"

        folder_path_1D = f"{folder_path_mc}/{self.model_name}_1D"
        folder_path_1D_chain = f"{folder_path_1D}/chains"
        folder_path_1D_hist = f"{folder_path_1D}/hist"

        folder_path_2D = f"{folder_path_mc}/{self.model_name}_2D"
        folder_path_2D_chain = f"{folder_path_2D}/chains"
        folder_path_2D_hist = f"{folder_path_2D}/hist"
        folder_path_2D_proba = f"{folder_path_2D}/proba_contours"

        for path_ in [
            folder_path_mc,
            folder_path_1D,
            folder_path_1D_chain,
            folder_path_1D_hist,
            folder_path_2D,
            folder_path_2D_chain,
            folder_path_2D_hist,
            folder_path_2D_proba,
        ]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return (
            folder_path_1D_chain,
            folder_path_1D_hist,
            folder_path_2D_chain,
            folder_path_2D_hist,
            folder_path_2D_proba,
        )

    def full_mc_analysis(
        self,
        scaler: Scaler,
        Theta_true_scaled_full: np.ndarray,
        list_mcmc_folders: List[str],
        plot_ESS: bool,
        plot_1D_chains: bool,
        plot_2D_chains: bool,
        plot_comparisons_yspace: bool,
        #
        folder_path_1D_chain: str,
        folder_path_1D_hist: str,
        folder_path_2D_chain: str,
        folder_path_2D_hist: str,
        folder_path_2D_proba: str,
        #
        point_challenger: Dict = {},
        list_CI: List[int] = [],
    ) -> None:
        global _one_pixel_mmse_ic_extraction

        list_fixed_values_lin = np.zeros((1, self.D))
        for d, value in enumerate(self.list_fixed_values_scaled):
            if value is not None:
                list_fixed_values_lin[0, d] = value * 1

        list_fixed_values_lin = scaler.from_scaled_to_lin(
            list_fixed_values_lin,
        ).flatten()  # (D,)

        # list_fixed_values_lin = [
        #     v if d not in self.list_idx_sampling else None
        #     for d, v in enumerate(list_fixed_values_lin)
        # ]

        len_mc = (self.T_MC - self.T_BI) // self.freq_save

        def _one_pixel_mmse_ic_extraction(dict_input: dict):
            """for one pixel n, performs:
            - MMSE and credibility interval extraction
            - ESS computation
            - plot 1D histograms
            - plot 2D histograms
            """
            n = dict_input["n"]
            Theta_n_true = dict_input["Theta_n_true"]

            # read data
            list_Theta_n_lin_full = np.zeros((self.N_MCMC, len_mc, self.D))
            for d in range(self.D):
                if d not in self.list_idx_sampling:
                    list_Theta_n_lin_full[:, :, d] = list_fixed_values_lin[d] * 1

            for seed, mc_path in enumerate(list_mcmc_folders):
                with h5py.File(mc_path, "r") as f:
                    list_Theta_n_lin_sub = np.array(
                        f["list_Theta"][self.effective_T_BI :, n, :]
                    )

                for idx, d in enumerate(self.list_idx_sampling):
                    list_Theta_n_lin_full[seed, :, d] = list_Theta_n_lin_sub[:, idx]

            # * MMSE and IC estimators
            list_Theta_n_lin_full_flatter = list_Theta_n_lin_full.reshape(
                (self.N_MCMC * (len_mc), self.D)
            )

            # compute percentiles
            dict_ci_per = {
                ci: {"lower": (100 - ci) / 2, "upper": 100 - (100 - ci) / 2}
                for ci in list_CI
            }
            list_per = list([dict_ci_per[ci]["lower"] for ci in list_CI])
            list_per += list([dict_ci_per[ci]["upper"] for ci in list_CI])
            list_per.sort()
            dict_per = {
                per: np.percentile(list_Theta_n_lin_full_flatter, per, axis=0)  # (D,)
                for per in list_per
            }

            # compute MMSE
            list_Theta_n_scaled_full_flatter = scaler.from_lin_to_scaled(
                list_Theta_n_lin_full_flatter,
            )

            # for d in range(self.D):
            #     if self.list_fixed_values_scaled[d] is not None:
            #         list_Theta_n_scaled_flatter[:, d] += self.list_fixed_values_scaled[d]

            Theta_n_MMSE_scaled = np.mean(
                list_Theta_n_scaled_full_flatter, axis=0
            )  # (D,)
            Theta_n_MMSE_lin = scaler.from_scaled_to_lin(
                Theta_n_MMSE_scaled.reshape((1, self.D))
            ).flatten()  # (D,)

            assert Theta_n_MMSE_lin.shape == (
                self.D,
            ), f"shape {Theta_n_MMSE_lin.shape}, should be {(self.D,)}"

            # create and save dataset of MMSE and IC
            df_estim = pd.DataFrame()
            df_estim["n"] = n * np.ones((self.D,), dtype=np.int32)
            df_estim["d"] = np.arange(self.D)
            df_estim["Theta_MMSE"] = Theta_n_MMSE_lin * 1
            for per in list_per:
                df_estim[f"per_{per:.1f}".replace(".", "p")] = dict_per[per]

            # in order to avoid re-writing multiple times the header because of
            # parallel writing, force a delay to favor n = 0 to be written
            # first with header
            path_overall_results = f"{self.path_data_csv_out_mcmc}/"
            path_overall_results += f"estimation_Theta_{self.model_name}.csv"

            if n == 0:
                df_estim.to_csv(path_overall_results, mode="w")
            else:
                while not (os.path.exists(path_overall_results)):
                    time.sleep(0.5)

                df_estim.to_csv(
                    path_overall_results,
                    mode="a",
                    header=not (os.path.exists(path_overall_results)),
                )

            del df_estim

            # *  index of first element st true val btw [MC first val, elt] or
            # * [elt, MC first val]
            if Theta_n_true is not None:
                first_elt_arr = -np.ones((self.N_MCMC, self.D_sampling))
                for seed in range(self.N_MCMC):
                    for idx_d, d in enumerate(self.list_idx_sampling):
                        if list_Theta_n_lin_full[seed, 0, d] < Theta_n_true[d]:
                            (idx,) = np.where(
                                list_Theta_n_lin_full[seed, :, d] >= Theta_n_true[d],
                            )
                        else:
                            (idx,) = np.where(
                                list_Theta_n_lin_full[seed, :, d] <= Theta_n_true[d],
                            )

                        if idx.size > 0:
                            first_elt_arr[seed, idx_d] = idx[0]

                list_dict = [
                    {
                        "seed": seed,
                        "n": n,
                        "d": d,
                        "first_elt_valid_mc": int(first_elt_arr[seed, idx_d]),
                    }
                    for seed in range(self.N_MCMC)
                    for idx_d, d in enumerate(self.list_idx_sampling)
                ]
                df_first_elt_valid_mc = pd.DataFrame.from_records(list_dict)

                path_file = f"{self.path_data_csv_out_mcmc}/"
                path_file += f"first_elt_valid_mc_{self.model_name}.csv"
                if n == 0:
                    df_first_elt_valid_mc.to_csv(
                        path_file,
                        mode="w",
                    )
                else:
                    while not (os.path.exists(path_file)):
                        time.sleep(0.5)

                    df_first_elt_valid_mc.to_csv(
                        path_file,
                        mode="a",
                        header=not (os.path.exists(path_file)),
                    )

                del df_first_elt_valid_mc

            # * ESS
            if plot_ESS:
                list_Theta_n_scaled_full = list_Theta_n_scaled_full_flatter.reshape(
                    (self.N_MCMC, len_mc, self.D)
                )
                list_dict_output = []
                for d in self.list_idx_sampling:
                    ess_ = ess.compute_ess(list_Theta_n_scaled_full[:, :, d])
                    list_dict_output.append(
                        {
                            "n": n,
                            "d": d,
                            "seed": "overall",
                            "model_name": self.model_name,
                            "ess": ess_,
                        }
                    )

                df_ess_nd = pd.DataFrame.from_records(list_dict_output)

                path_file = f"{self.path_data_csv_out_mcmc}/"
                path_file += f"estimation_ESS_{self.model_name}.csv"
                if n == 0:
                    df_ess_nd.to_csv(path_file, mode="w")

                else:
                    while not (os.path.exists(path_file)):
                        time.sleep(0.5)

                    df_ess_nd.to_csv(
                        path_file,
                        mode="a",
                        header=not (os.path.exists(path_file)),
                    )

                del df_ess_nd

            # * 1D histograms
            if plot_1D_chains:
                for d in self.list_idx_sampling:
                    true_val = Theta_n_true[d] if Theta_n_true is not None else None

                    histograms.plot_1D_chain(
                        list_Theta_n_lin_full_flatter[:, d],
                        n,
                        d,
                        folder_path_1D_chain,
                        f"MC component {self.list_names[d]} of pixel {n}",
                        self.lower_bounds_lin,
                        self.upper_bounds_lin,
                        self.N_MCMC,
                        self.T_MC,
                        self.T_BI,
                        true_val,
                    )

                    histograms.plot_1D_hist(
                        list_Theta_n_lin_full_flatter[:, d],
                        n,
                        d,
                        folder_path_1D_hist,
                        title=f"hist. of {self.list_names[d]} of pixel {n}",
                        lower_bounds_lin=self.lower_bounds_lin,
                        upper_bounds_lin=self.upper_bounds_lin,
                        seed=None,
                        estimator=Theta_n_MMSE_lin[d],
                        true_val=true_val,
                    )

            # * 2D histograms
            if plot_2D_chains and self.D > 1:
                for idx_d1, d1 in enumerate(self.list_idx_sampling):
                    for d2 in self.list_idx_sampling[idx_d1 + 1 :]:
                        if Theta_n_true is not None:
                            true_val = Theta_n_true[[d1, d2]] * 1
                        else:
                            true_val = None

                        histograms.plot_2D_hist(
                            list_Theta_n_lin_full_flatter[:, [d1, d2]],
                            n,
                            d1,
                            d2,
                            self.model_name,
                            folder_path_2D_hist,
                            self.list_names,
                            self.lower_bounds_lin,
                            self.upper_bounds_lin,
                            Theta_MMSE=Theta_n_MMSE_lin[[d1, d2]],
                            true_val=true_val,
                            point_challenger=point_challenger,
                        )

                        try:
                            histograms.plot_2D_proba_contours(
                                list_Theta_n_lin_full_flatter[:, [d1, d2]],
                                n,
                                d1,
                                d2,
                                self.model_name,
                                folder_path_2D_proba,
                                self.list_names,
                                self.lower_bounds_lin,
                                self.upper_bounds_lin,
                                Theta_MMSE=Theta_n_MMSE_lin[[d1, d2]],
                                true_val=true_val,
                                point_challenger=point_challenger,
                            )
                        except:
                            msg = "Issue with proba contour plot for (n, d1, d2) = "
                            msg += f"({n}, {d1}, {d2})"
                            print(msg)

            return

        # * global part of the function
        if Theta_true_scaled_full is not None:
            Theta_true_lin = scaler.from_scaled_to_lin(Theta_true_scaled_full)
        else:
            Theta_true_lin = None

        list_params = [
            {
                "n": n,
                "Theta_n_true": Theta_true_lin[n]
                if Theta_true_lin is not None
                else None,
            }
            for n in range(self.N)
        ]

        # ? The parallel execution may fail on mac, even with the mp_context
        # ? argument. As I can't correct the error, in case of fail, I perform
        # ? the extraction in series, which is much slower.
        try:
            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=mp.get_context("fork")
            ) as p:
                _ = list(
                    tqdm(
                        p.map(_one_pixel_mmse_ic_extraction, list_params),
                        total=self.N,
                    )
                )
        except:
            warnings.warn(
                "The parallel pixel-wise result extraction failed. Extracting in series instead."
            )
            for params in tqdm(list_params):
                _one_pixel_mmse_ic_extraction(params)

        return

    def main(
        self,
        scaler: Scaler,
        Theta_true_scaled_full: np.ndarray,
        list_mcmc_folders: List[str],
        plot_ESS: bool,
        plot_1D_chains: bool,
        plot_2D_chains: bool,
        plot_comparisons_yspace: bool,
        point_challenger: Dict = {},
        list_CI: List[int] = [],
    ):
        (
            folder_path_1D_chain,
            folder_path_1D_hist,
            folder_path_2D_chain,
            folder_path_2D_hist,
            folder_path_2D_proba,
        ) = self.create_folders()

        self.full_mc_analysis(
            scaler,
            Theta_true_scaled_full,
            list_mcmc_folders,
            plot_ESS,
            plot_1D_chains,
            plot_2D_chains,
            plot_comparisons_yspace,
            #
            folder_path_1D_chain,
            folder_path_1D_hist,
            folder_path_2D_chain,
            folder_path_2D_hist,
            folder_path_2D_proba,
            #
            point_challenger=point_challenger,
            list_CI=list_CI,
        )
        return
