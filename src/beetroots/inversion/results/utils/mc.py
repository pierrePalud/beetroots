import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.inversion.results.utils.mc_utils import ess, histograms
from beetroots.space_transform.abstract_transform import Scaler


def _one_pixel_mmse_ic_extraction_worker(dict_input: dict) -> dict:
    """Compute one pixel extraction and return tabular results for parent-side writes."""
    n = dict_input["n"]
    D = dict_input["D"]
    D_sampling = dict_input["D_sampling"]
    N_MCMC = dict_input["N_MCMC"]
    T_MC = dict_input["T_MC"]
    T_BI = dict_input["T_BI"]
    effective_T_BI = dict_input["effective_T_BI"]
    len_mc = dict_input["len_mc"]
    list_idx_sampling = dict_input["list_idx_sampling"]
    list_names = dict_input["list_names"]
    list_fixed_values_lin = dict_input["list_fixed_values_lin"]
    list_mcmc_folders = dict_input["list_mcmc_folders"]
    scaler = dict_input["scaler"]
    Var_n_true = dict_input["Var_n_true"]
    lower_bounds_lin = dict_input["lower_bounds_lin"]
    upper_bounds_lin = dict_input["upper_bounds_lin"]
    model_name = dict_input["model_name"]
    list_CI = dict_input["list_CI"]
    point_challenger_n = dict_input["point_challenger_n"]
    plot_ESS = dict_input["plot_ESS"]
    plot_1D_chains = dict_input["plot_1D_chains"]
    plot_2D_chains = dict_input["plot_2D_chains"]
    folder_path_1D_chain = dict_input["folder_path_1D_chain"]
    folder_path_1D_hist = dict_input["folder_path_1D_hist"]
    folder_path_2D_hist = dict_input["folder_path_2D_hist"]
    folder_path_2D_proba = dict_input["folder_path_2D_proba"]
    N = dict_input["N"]
    sub_var = dict_input["sub_var"]

    list_Var_n_lin_full = np.zeros((N_MCMC, len_mc, D))
    for d in range(D):
        if d not in list_idx_sampling:
            list_Var_n_lin_full[:, :, d] = list_fixed_values_lin[0, d] * 1

    for seed, mc_path in enumerate(list_mcmc_folders):
        with h5py.File(mc_path, "r") as f:
            if sub_var:
                f = f[sub_var]
            list_Var_n_lin_sub = np.array(f["list_var"][effective_T_BI:, n, :])

        for idx, d in enumerate(list_idx_sampling):
            list_Var_n_lin_full[seed, :, d] = list_Var_n_lin_sub[:, idx]

    list_Var_n_lin_full_flatter = list_Var_n_lin_full.reshape((N_MCMC * len_mc, D))

    dict_ci_per = {
        ci: {"lower": (100 - ci) / 2, "upper": 100 - (100 - ci) / 2} for ci in list_CI
    }
    list_per = [dict_ci_per[ci]["lower"] for ci in list_CI]
    list_per += [dict_ci_per[ci]["upper"] for ci in list_CI]
    list_per.sort()
    dict_per = {
        per: np.percentile(list_Var_n_lin_full_flatter, per, axis=0) for per in list_per
    }

    list_Var_n_scaled_full_flatter = scaler.from_lin_to_scaled(
        list_Var_n_lin_full_flatter
    )
    Var_n_MMSE_scaled = np.mean(list_Var_n_scaled_full_flatter, axis=0)
    Var_n_MMSE_lin = scaler.from_scaled_to_lin(
        Var_n_MMSE_scaled.reshape((1, D))
    ).flatten()

    assert Var_n_MMSE_lin.shape == (D,), (
        f"shape {Var_n_MMSE_lin.shape}, should be {(D,)}"
    )

    df_estim = pd.DataFrame()
    df_estim["n"] = n * np.ones((D,), dtype=np.int32)
    df_estim["d"] = np.arange(D)
    df_estim["Var_MMSE"] = Var_n_MMSE_lin * 1
    for per in list_per:
        df_estim[f"per_{per:.1f}".replace(".", "p")] = dict_per[per]

    df_first_elt_valid_mc = None
    if Var_n_true is not None:
        first_elt_arr = -np.ones((N_MCMC, D_sampling))
        for seed in range(N_MCMC):
            for idx_d, d in enumerate(list_idx_sampling):
                if list_Var_n_lin_full[seed, 0, d] < Var_n_true[d]:
                    (idx,) = np.where(list_Var_n_lin_full[seed, :, d] >= Var_n_true[d])
                else:
                    (idx,) = np.where(list_Var_n_lin_full[seed, :, d] <= Var_n_true[d])

                if idx.size > 0:
                    first_elt_arr[seed, idx_d] = idx[0]

        list_dict = [
            {
                "seed": seed,
                "n": n,
                "d": d,
                "first_elt_valid_mc": int(first_elt_arr[seed, idx_d]),
            }
            for seed in range(N_MCMC)
            for idx_d, d in enumerate(list_idx_sampling)
        ]
        df_first_elt_valid_mc = pd.DataFrame.from_records(list_dict)

    df_ess_nd = None
    if plot_ESS:
        list_Var_n_scaled_full = list_Var_n_scaled_full_flatter.reshape(
            (N_MCMC, len_mc, D)
        )
        list_dict_output = []
        for d in list_idx_sampling:
            ess_ = ess.compute_ess(list_Var_n_scaled_full[:, :, d])
            list_dict_output.append(
                {
                    "n": n,
                    "d": d,
                    "seed": "overall",
                    "model_name": model_name,
                    "ess": ess_,
                }
            )
        df_ess_nd = pd.DataFrame.from_records(list_dict_output)

    if plot_1D_chains:
        for d in list_idx_sampling:
            true_val = Var_n_true[d] if Var_n_true is not None else None

            title = f"Markov Chain of {list_names[d]}"
            if N > 1:
                title += f" of pixel {n}"

            histograms.plot_1D_chain(
                list_Var_lin_nd=list_Var_n_lin_full_flatter[:, d],
                n=n if N > 1 else None,
                d=d,
                folder_path=folder_path_1D_chain,
                title=title,
                lower_bounds_lin=lower_bounds_lin,
                upper_bounds_lin=upper_bounds_lin,
                N_MCMC=N_MCMC,
                T_MC=T_MC,
                T_BI=T_BI,
                true_val=true_val,
            )

            title = f"Sample histogram of {list_names[d]}"
            if N > 1:
                title += f" of pixel {n}"

            histograms.plot_1D_hist(
                list_Var_lin_seed=list_Var_n_lin_full_flatter[:, d],
                n=n if N > 1 else None,
                d=d,
                folder_path=folder_path_1D_hist,
                title=title,
                lower_bounds_lin=lower_bounds_lin,
                upper_bounds_lin=upper_bounds_lin,
                seed=None,
                estimator=Var_n_MMSE_lin[d],
                true_val=true_val,
            )

    if plot_2D_chains and D > 1:
        for idx_d1, d1 in enumerate(list_idx_sampling):
            for d2 in list_idx_sampling[idx_d1 + 1 :]:
                if Var_n_true is not None:
                    true_val = Var_n_true[[d1, d2]] * 1
                else:
                    true_val = None

                histograms.plot_2D_hist(
                    list_Var_n_lin_full_flatter[:, [d1, d2]],
                    n,
                    d1,
                    d2,
                    model_name,
                    folder_path_2D_hist,
                    list_names,
                    lower_bounds_lin,
                    upper_bounds_lin,
                    Var_MMSE=Var_n_MMSE_lin[[d1, d2]],
                    true_val=true_val,
                    point_challenger=point_challenger_n,
                )

                try:
                    histograms.plot_2D_proba_contours(
                        list_Var_n_lin_full_flatter[:, [d1, d2]],
                        n,
                        d1,
                        d2,
                        model_name,
                        folder_path_2D_proba,
                        list_names,
                        lower_bounds_lin,
                        upper_bounds_lin,
                        Var_MMSE=Var_n_MMSE_lin[[d1, d2]],
                        true_val=true_val,
                        point_challenger=point_challenger_n,
                    )
                except Exception as error:
                    msg = "Issue with proba contour plot for (n, d1, d2) = "
                    msg += f"({n}, {d1}, {d2})"
                    msg += f" | Error: {error}"
                    print(msg)

    return {
        "n": n,
        "df_estim": df_estim,
        "df_first_elt_valid_mc": df_first_elt_valid_mc,
        "df_ess_nd": df_ess_nd,
    }


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
        list_fixed_values_lin: List,
        lower_bounds_lin: Union[np.ndarray, List[float]],
        upper_bounds_lin: Union[np.ndarray, List[float]],
        list_names: List[str],
        sub_var: Optional[str] = "",
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

        self.sub_var = sub_var

        if isinstance(lower_bounds_lin, list):
            lower_bounds_lin = np.array(lower_bounds_lin)
        if isinstance(upper_bounds_lin, list):
            upper_bounds_lin = np.array(upper_bounds_lin)

        self.lower_bounds_lin = lower_bounds_lin
        self.upper_bounds_lin = upper_bounds_lin

        self.list_idx_sampling = list_idx_sampling
        self.list_fixed_values_lin = list_fixed_values_lin

        self.N = N
        self.D = upper_bounds_lin.size
        self.D_sampling = len(self.list_idx_sampling)

        self.list_names = list_names

    def read_data(self):
        pass

    def create_folders(self) -> Tuple[str, str, str, str, str]:
        folder_path_mc = f"{self.path_img}/mc"

        folder_path_1D = f"{folder_path_mc}/{self.model_name}_1D"
        folder_path_1D_chain = os.path.join(folder_path_1D, "chains", self.sub_var)
        folder_path_1D_hist = os.path.join(folder_path_1D, "hist", self.sub_var)

        folder_path_2D = f"{folder_path_mc}/{self.model_name}_2D"
        folder_path_2D_chain = os.path.join(folder_path_2D, "chains", self.sub_var)
        folder_path_2D_hist = os.path.join(folder_path_2D, "hist", self.sub_var)
        folder_path_2D_proba = os.path.join(
            folder_path_2D, "proba_contours", self.sub_var
        )

        for path_ in [
            folder_path_1D_chain,
            folder_path_1D_hist,
            folder_path_2D_chain,
            folder_path_2D_hist,
            folder_path_2D_proba,
        ]:
            if not os.path.isdir(path_):
                os.makedirs(path_)

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
        Var_true_scaled_full: np.ndarray,
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
        list_fixed_values_lin = np.zeros((1, self.D))
        for d, value in enumerate(self.list_fixed_values_lin):
            if value is not None:
                list_fixed_values_lin[0, d] = value * 1

        len_mc = (self.T_MC - self.T_BI) // self.freq_save

        # * global part of the function
        if Var_true_scaled_full is not None:
            Var_true_lin = scaler.from_scaled_to_lin(Var_true_scaled_full)
        else:
            Var_true_lin = None

        list_params = []
        for n in range(self.N):
            list_params.append(
                {
                    "n": n,
                    "D": self.D,
                    "D_sampling": self.D_sampling,
                    "N_MCMC": self.N_MCMC,
                    "T_MC": self.T_MC,
                    "T_BI": self.T_BI,
                    "effective_T_BI": self.effective_T_BI,
                    "len_mc": len_mc,
                    "list_idx_sampling": self.list_idx_sampling,
                    "list_names": self.list_names,
                    "list_fixed_values_lin": list_fixed_values_lin,
                    "list_mcmc_folders": list_mcmc_folders,
                    "scaler": scaler,
                    "Var_n_true": (
                        Var_true_lin[n] * 1 if Var_true_lin is not None else None
                    ),
                    "lower_bounds_lin": self.lower_bounds_lin,
                    "upper_bounds_lin": self.upper_bounds_lin,
                    "model_name": self.model_name,
                    "list_CI": list_CI,
                    "point_challenger_n": (
                        {
                            "name": point_challenger["name"],
                            "value": point_challenger["value"][n, :],
                        }
                        if len(point_challenger) > 0
                        else point_challenger
                    ),
                    "plot_ESS": plot_ESS,
                    "plot_1D_chains": plot_1D_chains,
                    "plot_2D_chains": plot_2D_chains,
                    "folder_path_1D_chain": folder_path_1D_chain,
                    "folder_path_1D_hist": folder_path_1D_hist,
                    "folder_path_2D_hist": folder_path_2D_hist,
                    "folder_path_2D_proba": folder_path_2D_proba,
                    "N": self.N,
                    "sub_var": self.sub_var,
                }
            )

        list_results = []
        try:
            print("mc.py - running pixel-wise extraction in parallel.")

            # ==========================================
            # LOCK DOWN ENVIRONMENT FOR CHILD PROCESSES
            # 'spawn' will inherit these variables.
            # ==========================================
            thread_vars = [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",  # Sometimes used by Pandas
            ]
            old_env = {var: os.environ.get(var) for var in thread_vars}

            for var in thread_vars:
                os.environ[var] = "1"
            # ==========================================

            with ProcessPoolExecutor(
                max_workers=int(self.max_workers),
                mp_context=mp.get_context("spawn"),
            ) as p:
                list_results = list(
                    tqdm(
                        p.map(_one_pixel_mmse_ic_extraction_worker, list_params),
                        total=int(self.N),
                    )
                )

            # ==========================================
            # RESTORE ORIGINAL ENVIRONMENT
            # ==========================================
            for var, val in old_env.items():
                if val is None:
                    del os.environ[var]
                else:
                    os.environ[var] = val
            # ==========================================

        except Exception as error:
            warnings.warn(
                f"The parallel pixel-wise result extraction failed ({error}). "
                "Extracting in series instead."
            )
            for params in tqdm(list_params):
                list_results.append(_one_pixel_mmse_ic_extraction_worker(params))

        path_folder = os.path.join(self.path_data_csv_out_mcmc, self.sub_var)
        os.makedirs(path_folder, exist_ok=True)

        list_results = sorted(list_results, key=lambda x: x["n"])

        df_estim_all = pd.concat([res["df_estim"] for res in list_results], axis=0)
        path_overall_results = os.path.join(
            path_folder, f"estimation_Var_MMSE_CI_{self.model_name}.csv"
        )
        df_estim_all.to_csv(path_overall_results, mode="w")

        list_df_first = [
            res["df_first_elt_valid_mc"]
            for res in list_results
            if res["df_first_elt_valid_mc"] is not None
        ]
        if list_df_first:
            df_first_elt_all = pd.concat(list_df_first, axis=0)
            path_file = os.path.join(
                path_folder,
                f"first_elt_valid_mc_{self.model_name}.csv",
            )
            df_first_elt_all.to_csv(path_file, mode="w")

        if plot_ESS:
            list_df_ess = [
                res["df_ess_nd"] for res in list_results if res["df_ess_nd"] is not None
            ]
            if list_df_ess:
                df_ess_all = pd.concat(list_df_ess, axis=0)
                path_file = os.path.join(
                    path_folder,
                    f"estimation_ESS_{self.model_name}.csv",
                )
                df_ess_all.to_csv(path_file, mode="w")

        return

    def main(
        self,
        scaler: Scaler,
        Var_true_scaled_full: np.ndarray,
        list_mcmc_folders: List[str],
        plot_ESS: bool,
        plot_1D_chains: bool,
        plot_2D_chains: bool,
        plot_comparisons_yspace: bool,
        point_challenger: Dict = {},
        list_CI: List[int] = [],
    ):
        print("starting full mc analysis")
        (
            folder_path_1D_chain,
            folder_path_1D_hist,
            folder_path_2D_chain,
            folder_path_2D_hist,
            folder_path_2D_proba,
        ) = self.create_folders()

        self.full_mc_analysis(
            scaler,
            Var_true_scaled_full,
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
        print("full mc analysis done")
        return
