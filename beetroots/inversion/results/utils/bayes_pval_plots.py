import os
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from scipy.stats import beta

from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.inversion.results.utils.abstract_util import ResultsUtil
from beetroots.sampler.abstract_sampler import Sampler


class ResultsBayesPvalues(ResultsUtil):
    r"""Bayesian model checking accounting for uncertainties on the p-value due to Monte Carlo evaluation.
    The method is described in :cite:t:`paludProblemesInversesTest2023a`.
    """

    __slots = (
        "model_name",
        "path_img",
        "path_data_csv_out",
        "N_MCMC",
        "N",
        "D",
    )

    CONFIDENCE_THRESHOLD_ALPHA = 0.05
    r"""..., denoted :math:`\alpha` in the article"""

    CONFIDENCE_THRESHOLD_DELTA = 0.1
    r"""..., denoted :math:`\delta` in the article"""

    ESS_OPTIM = Sampler.ESS_OPTIM * 1
    r"""number of random reproduced observations to draw to evaluate the model checking p-value for optimization procedures"""

    def __init__(
        self,
        model_name: str,
        chain_type: str,
        path_img: str,
        path_data_csv_out: str,
        N_MCMC: int,
        N: int,
        D_sampling: int,
        plot_ESS: bool,
    ):
        """

        Parameters
        ----------
        model_name : str
            _description_
        chain_type : str
            _description_
        path_img : str
            _description_
        path_data_csv_out : str
            _description_
        N_MCMC : int
            _description_
        N : int
            _description_
        D_sampling : int
            _description_
        plot_ESS : bool
            _description_
        """
        assert chain_type in ["mcmc", "optim_map", "optim_mle"]

        self.model_name = model_name
        self.chain_type = chain_type
        self.path_img = path_img
        self.path_data_csv_out = path_data_csv_out

        self.N_MCMC = N_MCMC
        self.N = N
        self.D_sampling = D_sampling

        self.plot_ESS = plot_ESS

    def read_data(self) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        # ess
        if self.chain_type == "mcmc":
            if self.plot_ESS:
                path_file = f"{self.path_data_csv_out}/"
                path_file += f"estimation_ESS_{self.model_name}.csv"

                df_ess_model = pd.read_csv(path_file, index_col=["n", "d"])
                df_ess_model = df_ess_model.sort_index()
                df_ess_model = df_ess_model.reset_index(drop=False)

            else:
                df_ess_model = None
        else:
            list_dicts = [
                {"n": n, "d": d, "ess": self.ESS_OPTIM, "seed": "overall"}
                for n in range(self.N)
                for d in range(self.D_sampling)
            ]
            df_ess_model = pd.DataFrame.from_records(list_dicts)
            df_ess_model = df_ess_model.set_index(["n", "d"])
            df_ess_model = df_ess_model.sort_index().reset_index(drop=False)

        # estimated pval
        path_file = f"{self.path_data_csv_out}/"
        path_file += f"p_value_llh_{self.model_name}.csv"

        df_pval = pd.read_csv(path_file, index_col=["seed", "n"])
        df_pval = df_pval.sort_index().reset_index(drop=False)

        assert df_ess_model is None or len(df_ess_model) == self.N * self.D_sampling
        assert len(df_pval) == self.N * self.N_MCMC
        return df_ess_model, df_pval

    def create_folders(self) -> str:
        folder_path_inter = f"{self.path_img}/p-values"
        folder_path = f"{folder_path_inter}/{self.model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return folder_path

    def main(
        self,
        list_idx_sampling: List[int],
        map_shaper: Optional[MapShaper],
    ) -> None:

        if self.N < 1:
            msg = "this function should only be called when N > 1 "
            msg += "to avoid 1-pixel maps"
            raise ValueError(msg)

        df_ess_model, df_pval = self.read_data()
        folder_path = self.create_folders()

        print("starting Bayesian p-value plots")

        if self.plot_ESS:
            assert df_ess_model is not None
            df_ess_overall = df_ess_model[(df_ess_model["seed"] == "overall")]
            df_ess = df_ess_overall.groupby("n")[["ess"]].min()
            df_ess = df_ess.sort_index()
            ess = df_ess["ess"].values

        list_decisions = []

        for seed in range(self.N_MCMC):
            df_pval_seed = df_pval[df_pval["seed"] == seed]
            df_pval_seed = df_pval_seed.sort_values("n")
            pval_estim = df_pval_seed["p-value-llh"].values

            decision_estim_seed = np.where(
                pval_estim < self.CONFIDENCE_THRESHOLD_ALPHA,
                -1,
                1,
            )

            if self.plot_ESS:
                proba_reject = np.zeros((self.N,))
                for n in range(self.N):
                    rv = beta(
                        1 + ess[n] * pval_estim[n],
                        1 + ess[n] * (1 - pval_estim[n]),
                    )
                    proba_reject[n] = rv.cdf(self.CONFIDENCE_THRESHOLD_ALPHA)

                decision_pval_bayes = np.where(
                    proba_reject > 1 - self.CONFIDENCE_THRESHOLD_DELTA,
                    -1,
                    np.where(
                        proba_reject < self.CONFIDENCE_THRESHOLD_DELTA,
                        1,
                        0,
                    ),
                )
            else:
                proba_reject = [None for n in range(self.N)]
                decision_pval_bayes = [None for n in range(self.N)]

            list_decisions += [
                {
                    "seed": seed,
                    "n": n,
                    "pval_estim": pval_estim[n],
                    "decision_estim_seed": decision_estim_seed[n],
                    "proba_reject": proba_reject[n],
                    "decision_pval_bayes": decision_pval_bayes[n],
                }
                for n in range(self.N)
            ]

            if map_shaper is not None:
                pval_estim_seed_shaped = map_shaper.from_vector_to_map(pval_estim)
                decision_estim_seed_shaped = map_shaper.from_vector_to_map(
                    decision_estim_seed
                )
                if proba_reject[0] is not None:
                    proba_reject_shaped = map_shaper.from_vector_to_map(proba_reject)
                    decision_pval_bayes_shaped = map_shaper.from_vector_to_map(
                        decision_pval_bayes
                    )

                plt.figure(figsize=(8, 6))
                title = "Estimated p-values"
                if self.N_MCMC > 1:
                    title += f" for seed={seed}"
                plt.title(title)

                plt.imshow(
                    pval_estim_seed_shaped,
                    norm=colors.LogNorm(),
                    cmap="jet",
                    origin="lower",
                )
                plt.colorbar()
                # plt.tight_layout()
                filename = f"{folder_path}/pval_estim_{self.chain_type}"
                filename += f"_seed{seed}.PNG"
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

                #
                plt.figure(figsize=(8, 6))
                title = "Model rejection from estimated p-values"
                if self.N_MCMC > 1:
                    title += f" (seed={seed})"
                plt.title(title)

                plt.imshow(
                    decision_estim_seed_shaped,
                    cmap=colors.ListedColormap(["black", "white"]),
                    origin="lower",
                )

                colorbar = plt.colorbar(ticks=[-1 / 2, 1 / 2])
                colorbar.set_ticklabels(
                    ["reject", "reproduced"],
                    rotation=90,
                    va="center",
                )

                filename = f"{folder_path}/decision_pval_estim_{self.chain_type}"
                filename += f"_seed{seed}.PNG"
                plt.savefig(filename, bbox_inches="tight")
                # plt.tight_layout()
                plt.close()

                # * with bayesian computation of pval
                # Bayesian p-values
                if self.plot_ESS:
                    plt.figure(figsize=(8, 6))
                    plt.title(
                        r"Proba. reject per pixel $\mathbb{P}[p_n \leq \alpha]$",
                    )
                    plt.imshow(
                        proba_reject_shaped,
                        cmap="jet",
                        origin="lower",
                    )
                    plt.colorbar()
                    # plt.tight_layout()
                    plt.savefig(
                        f"{folder_path}/proba_reject_{self.chain_type}_seed{seed}.PNG",
                        bbox_inches="tight",
                    )
                    plt.close()

                    # decision process
                    plt.figure(figsize=(8, 6))
                    plt.title("Model rejection from p-values with uncertainties")
                    plt.imshow(
                        decision_pval_bayes_shaped,
                        cmap=colors.ListedColormap(["black", "grey", "white"]),
                        origin="lower",
                        norm=colors.Normalize(vmin=-1, vmax=1),
                    )

                    colorbar = plt.colorbar(ticks=[-2 / 3, 0, 2 / 3])
                    colorbar.set_ticklabels(
                        ["reject", "unclear", "no reject"],
                        rotation=90,
                        va="center",
                    )
                    plt.savefig(
                        f"{folder_path}/decision_from_bayes_pval_{self.chain_type}_seed{seed}.PNG",
                        bbox_inches="tight",
                    )
                    plt.close()

        df_pval = pd.DataFrame.from_records(list_decisions)
        df_pval.to_csv(f"{self.path_data_csv_out}/pvalues_analysis.csv")

        print("Bayesian p-value plots plots done")
        return
