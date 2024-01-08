import os
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.inversion.results.utils.abstract_util import ResultsUtil


class ResultsCLPPD(ResultsUtil):

    __slots = (
        "model_name",
        "path_img",
        "path_data_csv_out",
        "N",
        "D",
    )

    def __init__(
        self,
        model_name: str,
        chain_type: str,
        path_img: str,
        path_data_csv_out: str,
        N_MCMC: int,
        N: int,
        L: int,
    ):
        assert chain_type in ["mcmc", "optim_map", "optim_mle"]

        self.model_name = model_name
        self.path_img = path_img
        self.path_data_csv_out = path_data_csv_out
        self.chain_type = chain_type

        self.N_MCMC = N_MCMC

        self.N = N
        self.L = L

    def read_data(
        self,
        list_chains_folders: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        clppd = np.zeros((self.N_MCMC, self.N, self.L))
        # p_values = np.zeros((self.N_MCMC, self.N, self.L))
        p_values_llh = np.zeros((self.N_MCMC, self.N))

        index = pd.MultiIndex.from_product(
            [list(range(self.N_MCMC)), list(range(self.N))],
            names=["seed", "n"],
        )
        df_clppd = pd.DataFrame(
            np.zeros((self.N_MCMC * self.N, self.L)),
            columns=np.arange(self.L),
            index=index,
        )
        # df_p_values = pd.DataFrame(
        #     np.zeros((self.N_MCMC * self.N, self.L)),
        #     columns=np.arange(self.L),
        #     index=index,
        # )
        df_p_values_llh = pd.DataFrame(
            np.zeros((self.N_MCMC * self.N,)),
            columns=["p-value-llh"],
            index=index,
        )

        for seed, mc_path in enumerate(list_chains_folders):
            with h5py.File(mc_path, "r") as f:
                clppd[seed] = np.array(f["clppd"])
                df_clppd.loc[(seed, slice(None)), :] = clppd[seed] * 1

                # p_values[seed] = np.array(f["p-values-y"])
                # df_p_values.loc[(seed, slice(None)), :] = p_values[seed] * 1

                p_values_llh[seed] = np.array(f["p-values-llh"])
                df_p_values_llh.loc[(seed, slice(None)), :] = p_values_llh[seed] * 1

        path_file = f"{self.path_data_csv_out}/"
        path_file += f"clppd_inter_{self.model_name}.csv"
        df_clppd.to_csv(path_file)

        # path_file = f"{self.path_data_csv_out}/"
        # path_file += f"p_value_inter_{self.model_name}.csv"
        # df_p_values.to_csv(path_file)

        path_file = f"{self.path_data_csv_out}/"
        path_file += f"p_value_llh_{self.model_name}.csv"
        df_p_values_llh.to_csv(path_file)

        return clppd, p_values_llh  # p_values,

    def create_folders(self) -> str:
        folder_path_inter = f"{self.path_img}/clppd"
        folder_path = f"{folder_path_inter}/{self.model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return folder_path

    def main(
        self,
        list_chains_folders: List[str],
        map_shaper: Optional[MapShaper],
    ) -> None:

        if self.N < 1:
            msg = "this function should only be called when N > 1 "
            msg += "to avoid 1-pixel maps"
            raise ValueError(msg)

        clppd, p_values_llh = self.read_data(list_chains_folders)  # p_values,
        folder_path = self.create_folders()

        print("starting clppd plots")

        clppd = -2 / self.L * np.sum(np.log(clppd), axis=2)  # (N_MCMC, N)

        if map_shaper is not None:
            for seed in range(self.N_MCMC):
                clppd_seed_shaped = map_shaper.from_vector_to_map(clppd[seed])

                plt.figure(figsize=(8, 6))
                if self.N_MCMC == 1:
                    plt.title("CLPPD")
                else:
                    plt.title(f"CLPPD for seed={seed}")

                plt.imshow(
                    clppd_seed_shaped,
                    norm=colors.LogNorm(),
                    cmap="jet",
                    origin="lower",
                )
                plt.colorbar()
                # plt.tight_layout()
                plt.savefig(
                    f"{folder_path}/clppd_{self.chain_type}_seed{seed}.PNG",
                    bbox_inches="tight",
                )
                plt.close()

            # # * p-values computed with y^{rep}_\ell <= y_\ell
            # p_values = np.min(p_values, axis=2)  # (N_MCMC, N)
            # assert p_values.shape == (
            #     self.N_MCMC,
            #     self.N,
            # ), f"should be {(self.N_MCMC, self.N)}, is {p_values.shape}"

            # if map_shaper is not None:
            #     for seed in range(self.N_MCMC):
            #         p_values_seed_shaped = map_shaper.from_vector_to_map(
            #             p_values[seed],
            #         )

            #         plt.figure(figsize=(8, 6))
            #         title = "p-values"
            #         if self.N_MCMC > 1:
            #             title += f" for seed={seed}"
            #         plt.title(title)

            #         plt.imshow(
            #             p_values_seed_shaped,
            #             norm=colors.LogNorm(),
            #             cmap="jet",
            #             origin="lower",
            #         )
            #         plt.colorbar()
            #         # plt.tight_layout()
            #         filename = f"{folder_path}/p_values_{self.chain_type}"
            #         filename += f"_seed{seed}.PNG"
            #         plt.savefig(filename, bbox_inches="tight")
            #         plt.close()

            print("clppd plots done")
        return
