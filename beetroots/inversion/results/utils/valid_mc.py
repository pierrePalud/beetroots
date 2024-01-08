import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from beetroots.inversion.results.utils.abstract_util import ResultsUtil


class ResultsValidMC(ResultsUtil):

    __slots__ = (
        "model_name",
        "path_img",
        "path_data_csv_out_mcmc",
        "N_MCMC",
        "effective_len_mc",
        "N",
        "D",
    )

    def __init__(
        self,
        model_name: str,
        path_img: str,
        path_data_csv_out_mcmc: str,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        freq_save: int,
        N: int,
        D_sampling: int,
    ):
        self.model_name = model_name
        self.path_img = path_img
        self.path_data_csv_out_mcmc = path_data_csv_out_mcmc

        self.N_MCMC = N_MCMC
        self.effective_len_mc = (T_MC - T_BI) // freq_save

        self.N = N
        self.D_sampling = D_sampling

    def read_data(self) -> pd.DataFrame:
        path_file = f"{self.path_data_csv_out_mcmc}/"
        path_file += f"first_elt_valid_mc_{self.model_name}.csv"

        df_valid_mc = pd.read_csv(path_file)

        # assert len(df_valid_mc) == self.N_MCMC * self.N * self.D_sampling, f"len df_valid_mc={len(df_valid_mc)}, N_MCMC={self.N_MCMC}, N={self.N}, D_sampling={self.D_sampling}"
        return df_valid_mc

    def create_folders(self) -> str:
        folder_path_inter = f"{self.path_img}/well_reconstructed"
        folder_path = f"{folder_path_inter}/{self.model_name}"
        for path_ in [folder_path_inter, folder_path]:
            if not os.path.isdir(path_):
                os.mkdir(path_)

        return folder_path

    def main(
        self,
        list_names: List[str],
        list_idx_sampling: List[int],
    ) -> None:
        print("starting plot proportion of well reconstructed pixels")

        folder_path = self.create_folders()
        df_valid_mc = self.read_data()

        incr = 100 / self.N

        for seed in range(self.N_MCMC):
            df_seed = df_valid_mc[df_valid_mc["seed"] == seed]

            list_evolution = np.zeros((self.effective_len_mc, self.D_sampling))

            for idx_d, d in enumerate(list_idx_sampling):
                idx_arr_d = df_seed.loc[df_seed["d"] == d, "first_elt_valid_mc"].values
                for idx in idx_arr_d:
                    if idx > 0:
                        list_evolution[int(idx) :, idx_d] += incr

            plt.figure(figsize=(8, 6))
            plt.title("evolution of proportion of valid MC")
            idx_d = 0
            for d, name in enumerate(list_names):
                if d in list_idx_sampling:
                    plt.plot(list_evolution[:, idx_d], label=name)
                    idx_d += 1

            plt.plot(np.mean(list_evolution, 1), "k--", label="overall")
            plt.grid()
            plt.legend()

            filename = f"{folder_path}/prop_well_reconstruct_pixels_all_"
            filename += f"seed{seed}.PNG"
            plt.savefig(filename, bbox_inches="tight")
            plt.close()

        print("plot proportion of well reconstructed pixels done")
        return
