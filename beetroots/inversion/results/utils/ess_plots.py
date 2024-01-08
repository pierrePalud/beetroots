import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from beetroots.inversion.plots.map_shaper import MapShaper
from beetroots.inversion.results.utils.abstract_util import ResultsUtil


class ResultsESS(ResultsUtil):

    __slots = (
        "model_name",
        "path_img",
        "path_data_csv_out_mcmc",
        "N",
        "D",
    )

    def __init__(
        self,
        model_name: str,
        path_img: str,
        path_data_csv_out_mcmc: str,
        N: int,
        D_sampling: int,
    ):
        self.model_name = model_name
        self.path_img = path_img
        self.path_data_csv_out_mcmc = path_data_csv_out_mcmc

        self.N = N
        self.D_sampling = D_sampling

    def read_data(self) -> pd.DataFrame:
        path_file = f"{self.path_data_csv_out_mcmc}/"
        path_file += f"estimation_ESS_{self.model_name}.csv"

        df_ess_model = pd.read_csv(path_file, index_col=["n", "d"])
        df_ess_model = df_ess_model.sort_index().reset_index(drop=False)

        assert len(df_ess_model) == self.N * self.D_sampling
        return df_ess_model

    def create_folders(self) -> str:
        folder_path_inter = f"{self.path_img}/ess"
        folder_path = f"{folder_path_inter}/{self.model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return folder_path

    def main(
        self,
        map_shaper: MapShaper,
        list_names: List[str],
        list_idx_sampling: List[int],
    ) -> None:

        if self.N < 1:
            msg = "this function should only be called when N > 1 "
            msg += "to avoid 1-pixel maps"
            raise ValueError(msg)

        df_ess_model = self.read_data()
        folder_path = self.create_folders()

        print("starting ESS plots")

        for d in list_idx_sampling:
            df_ess_overall = df_ess_model[
                (df_ess_model["seed"] == "overall") & (df_ess_model["d"] == d)
            ]
            df_ess_overall = df_ess_overall.sort_values("n")

            ess_arr = df_ess_overall.loc[:, "ess"].values
            ess_arr_shaped = map_shaper.from_vector_to_map(ess_arr)

            plt.figure(figsize=(8, 6))
            plt.title(f"ESS per pixel for {list_names[d]}")
            plt.imshow(
                ess_arr_shaped,
                norm=colors.LogNorm(vmin=1.0),
                cmap="jet",
                origin="lower",
            )
            plt.colorbar()
            # plt.tight_layout()
            plt.savefig(f"{folder_path}/ESS_d{d}.PNG", bbox_inches="tight")
            plt.close()

        print("ESS plots done")
        return
