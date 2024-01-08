import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.inversion.results.utils.mc_utils import ess, histograms
from beetroots.space_transform.abstract_transform import Scaler


class ResultsMCHierarchical(ResultsUtil):

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
        "L",
        "list_names",
        "list_lines",
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
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
        list_names: List[str],
        list_lines: List[str],
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

        self.lower_bounds_lin = lower_bounds_lin
        self.upper_bounds_lin = upper_bounds_lin

        self.list_names = list_names
        self.list_lines = list_lines

        self.N = N
        self.D = upper_bounds_lin.size
        self.L = len(list_lines)

    def read_data(self):
        pass

    def create_folders(self):
        folder_path_mc = f"{self.path_img}/mc"

        folder_path_1D = f"{folder_path_mc}/{self.model_name}_1D"
        folder_path_1D_chain = f"{folder_path_1D}/chains"
        folder_path_1D_hist = f"{folder_path_1D}/hist"

        folder_path_2D = f"{folder_path_mc}/{self.model_name}_2D"
        folder_path_2D_chain = f"{folder_path_2D}/chains"
        folder_path_2D_hist = f"{folder_path_2D}/hist"

        for path_ in [
            folder_path_mc,
            folder_path_1D,
            folder_path_1D_chain,
            folder_path_1D_hist,
            folder_path_2D,
            folder_path_2D_chain,
            folder_path_2D_hist,
        ]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return (
            folder_path_1D_chain,
            folder_path_1D_hist,
            folder_path_2D_chain,
            folder_path_2D_hist,
        )

    def full_mc_analysis(
        self,
        scaler: Scaler,
        Theta_true_scaled: Optional[np.ndarray],
        u_true: Optional[np.ndarray],
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
    ) -> None:
        global _one_pixel_mmse_ic_extraction

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
            u_n_true = dict_input["u_n_true"]

            # read data
            list_Theta_n_lin = np.zeros((self.N_MCMC, len_mc, self.D))
            list_u_n_lin = np.zeros((self.N_MCMC, len_mc, self.L))
            for seed, mc_path in enumerate(list_mcmc_folders):
                with h5py.File(mc_path, "r") as f:
                    list_Theta_n_lin[seed] = np.array(
                        f["list_Theta"][self.effective_T_BI :, n, :]
                    )
                    list_u_n_lin[seed] = np.array(
                        f["list_U"][self.effective_T_BI :, n, :]
                    )

            # * MMSE and IC estimators
            list_Theta_n_lin_flatter = list_Theta_n_lin.reshape(
                (self.N_MCMC * (len_mc), self.D)
            )
            list_u_n_lin_flatter = list_u_n_lin.reshape(
                (self.N_MCMC * (len_mc), self.L)
            )

            # compute percentiles, all (D,)
            per_0p5 = np.percentile(list_Theta_n_lin_flatter, 0.5, axis=0)
            per_2p5 = np.percentile(list_Theta_n_lin_flatter, 2.5, axis=0)
            per_5 = np.percentile(list_Theta_n_lin_flatter, 5, axis=0)
            per_95 = np.percentile(list_Theta_n_lin_flatter, 95, axis=0)
            per_97p5 = np.percentile(list_Theta_n_lin_flatter, 97.5, axis=0)
            per_99p5 = np.percentile(list_Theta_n_lin_flatter, 99.5, axis=0)

            # compute percentiles, all (L,)
            per_0p5_u = np.percentile(list_u_n_lin_flatter, 0.5, axis=0)
            per_2p5_u = np.percentile(list_u_n_lin_flatter, 2.5, axis=0)
            per_5_u = np.percentile(list_u_n_lin_flatter, 5, axis=0)
            per_95_u = np.percentile(list_u_n_lin_flatter, 95, axis=0)
            per_97p5_u = np.percentile(list_u_n_lin_flatter, 97.5, axis=0)
            per_99p5_u = np.percentile(list_u_n_lin_flatter, 99.5, axis=0)

            # compute MMSE
            list_Theta_n_scaled_flatter = scaler.from_lin_to_scaled(
                list_Theta_n_lin_flatter,
            )
            Theta_n_MMSE_scaled = np.mean(list_Theta_n_scaled_flatter, axis=0)  # (D,)
            Theta_n_MMSE_lin = scaler.from_scaled_to_lin(
                Theta_n_MMSE_scaled.reshape((1, self.D))
            ).flatten()  # (D,)

            u_n_MMSE_lin = np.mean(list_u_n_lin_flatter, axis=0)  # (L,)

            assert per_0p5.shape == (
                self.D,
            ), f"shape {per_0p5.shape}, should be {(self.D,)}"
            assert Theta_n_MMSE_lin.shape == (
                self.D,
            ), f"shape {Theta_n_MMSE_lin.shape}, should be {(self.D,)}"

            # create and save dataset of MMSE and IC
            df_estim = pd.DataFrame()
            df_estim["n"] = n * np.ones((self.D,), dtype=np.int32)
            df_estim["d"] = np.arange(self.D)
            df_estim["Theta_MMSE"] = Theta_n_MMSE_lin * 1
            df_estim["per_0p5"] = per_0p5 * 1
            df_estim["per_2p5"] = per_2p5 * 1
            df_estim["per_5"] = per_5 * 1
            df_estim["per_95"] = per_95 * 1
            df_estim["per_97p5"] = per_97p5 * 1
            df_estim["per_99p5"] = per_99p5 * 1

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

            df_estim = pd.DataFrame()
            df_estim["n"] = n * np.ones((self.L,), dtype=np.int32)
            df_estim["ell"] = np.arange(self.L)
            df_estim["u_MMSE"] = u_n_MMSE_lin * 1
            df_estim["per_0p5"] = per_0p5_u * 1
            df_estim["per_2p5"] = per_2p5_u * 1
            df_estim["per_5"] = per_5_u * 1
            df_estim["per_95"] = per_95_u * 1
            df_estim["per_97p5"] = per_97p5_u * 1
            df_estim["per_99p5"] = per_99p5_u * 1

            # in order to avoid re-writing multiple times the header because of
            # parallel writing, force a delay to favor n = 0 to be written
            # first with header
            path_overall_results = f"{self.path_data_csv_out_mcmc}/"
            path_overall_results += f"estimation_u_{self.model_name}.csv"

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
                first_elt_arr = -np.ones((self.N_MCMC, self.D))
                for seed in range(self.N_MCMC):
                    for d in range(self.D):
                        if list_Theta_n_lin[seed, 0, d] < Theta_n_true[d]:
                            (idx,) = np.where(
                                list_Theta_n_lin[seed, :, d] >= Theta_n_true[d],
                            )
                        else:
                            (idx,) = np.where(
                                list_Theta_n_lin[seed, :, d] <= Theta_n_true[d],
                            )

                        if idx.size > 0:
                            first_elt_arr[seed, d] = idx[0]

                list_dict = [
                    {
                        "seed": seed,
                        "n": n,
                        "d": d,
                        "first_elt_valid_mc": int(first_elt_arr[seed, d]),
                    }
                    for seed in range(self.N_MCMC)
                    for d in range(self.D)
                ]
                df_first_elt_valid_mc = pd.DataFrame.from_records(list_dict)

                path_file = f"{self.path_data_csv_out_mcmc}/"
                path_file += f"first_elt_valid_mc_{self.model_name}.csv"
                if n == 0:
                    df_first_elt_valid_mc.to_csv(
                        path_file,
                        mode="w",
                        header=not (os.path.exists(path_file)),
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

            # * ESS (on theta only)
            if plot_ESS:
                list_Theta_n_scaled = list_Theta_n_scaled_flatter.reshape(
                    (self.N_MCMC, len_mc, self.D)
                )
                list_dict_output = []
                for d in range(self.D):
                    ess_ = ess.compute_ess(list_Theta_n_scaled[:, :, d])
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
                for d in range(self.D):
                    true_val = Theta_n_true[d] if Theta_n_true is not None else None
                    true_val_u = u_n_true[d] if u_n_true is not None else None

                    histograms.plot_1D_chain(
                        list_Theta_n_lin_flatter[:, d],
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
                        list_Theta_n_lin_flatter[:, d],
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

                lower_bounds_lin_u = np.min(list_u_n_lin_flatter, axis=0) / 1.1
                upper_bounds_lin_u = np.max(list_u_n_lin_flatter, axis=0) * 1.1
                for ell in range(len(self.list_lines)):
                    histograms.plot_1D_chain(
                        list_u_n_lin_flatter[:, ell],
                        n,
                        ell,
                        folder_path_1D_chain,
                        title=f"MC component of line {self.list_lines[ell]} of pixel {n}",
                        lower_bounds_lin=lower_bounds_lin_u,
                        upper_bounds_lin=upper_bounds_lin_u,
                        N_MCMC=self.N_MCMC,
                        T_MC=self.T_MC,
                        T_BI=self.T_BI,
                        true_val=true_val_u,
                        is_u=True,
                    )

            # * 2D histograms
            if plot_2D_chains and self.D > 1:
                for d1 in range(self.D):
                    for d2 in range(d1 + 1, self.D):
                        if Theta_n_true is not None:
                            true_val = Theta_n_true[[d1, d2]] * 1
                        else:
                            true_val = None

                        try:
                            histograms.plot_2D_hist(
                                list_Theta_n_lin_flatter[:, [d1, d2]],
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
                            )
                        except:
                            print(f"2D hist failed for n={n}, d1={d1}, d2={d2}")
                        # no 2D plot for u

            return

        # * global part of the function
        if Theta_true_scaled is not None:
            Theta_true_lin = scaler.from_scaled_to_lin(Theta_true_scaled)
        else:
            Theta_true_lin = None

        list_params = [
            {
                "n": n,
                "Theta_n_true": Theta_true_lin[n]
                if Theta_true_lin is not None
                else None,
                "u_n_true": u_true[n] if u_true is not None else None,
            }
            for n in range(self.N)
        ]
        # for params in tqdm(list_params):
        #     _one_pixel_mmse_ic_extraction(params)

        with ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=mp.get_context("fork")
        ) as p:
            _ = list(
                tqdm(
                    p.map(_one_pixel_mmse_ic_extraction, list_params),
                    total=self.N,
                )
            )

        return

    def main(
        self,
        scaler: Scaler,
        Theta_true_scaled: Optional[np.ndarray],
        u_true: Optional[np.ndarray],
        list_mcmc_folders: List[str],
        plot_ESS: bool,
        plot_1D_chains: bool,
        plot_2D_chains: bool,
        plot_comparisons_yspace: bool,
    ):
        (
            folder_path_1D_chain,
            folder_path_1D_hist,
            folder_path_2D_chain,
            folder_path_2D_hist,
        ) = self.create_folders()

        self.full_mc_analysis(
            scaler,
            Theta_true_scaled,
            u_true,
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
        )
        return
