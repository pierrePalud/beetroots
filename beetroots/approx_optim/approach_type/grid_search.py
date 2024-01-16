# from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from beetroots.approx_optim.approach_type.abstract_approach_type import ApproachType


class GridSearchApproach(ApproachType):
    def optimization(
        self,
    ):
        print("starting Kolmogorov Smirnov distances computations")

    list_params = [
        {
            "a0": a0,
            "a1": a1,
            "log10_f_Theta": log10_f_Theta,
            "list_log10_f_grid": list_log10_f_grid,
            "sigma_a": sigma_a,
            "sigma_m": sigma_m,
            "N_samples_y": N_samples_y,
            "pdf_kde_log10_f_Theta": pdf_kde_log10_f_Theta,
        }
        for a0 in list_a0
        for a1 in list_a1
    ]

    with ProcessPoolExecutor(max_workers=30, mp_context=mp.get_context("fork")) as p:
        list_results = list(
            tqdm(p.map(estimate_avg_dks, list_params), total=len(list_params))
        )
    print("Kolmogorov Smirnov distances computations done")

    # * save list_dists (in list_results) as table in csv
    list_dicts = []
    for dict_output in list_results:
        a0 = dict_output["a0"]
        a1 = dict_output["a1"]
        list_statistics = list(dict_output["list_dists"])
        dict_ = {"a0": a0, "a1": a1}
        for log10_f_Theta, statistic in zip(list_log10_f_grid, list_statistics):
            dict_[log10_f_Theta] = statistic

        list_dicts.append(dict_)

    df_statistics = pd.DataFrame(list_dicts)
    df_statistics.to_csv(f"{output_path}/results_statistics.csv")

    # * save values of avg dist
    list_dicts = []
    for dict_output in list_results:
        a0 = dict_output["a0"]
        a1 = dict_output["a1"]
        avg_dist = dict_output["avg_dist"]
        dict_ = {"a0": a0, "a1": a1, "avg_dist": avg_dist}
        list_dicts.append(dict_)

    df_avg_dist = pd.DataFrame(list_dicts)
    df_avg_dist.to_csv(f"{output_path}/results_avg_dist.csv")

    def plot_cost_map(self):
        AA0, AA1 = np.meshgrid(list_a0, list_a1)

    arr_avg_dist = np.zeros_like(AA0)

    for dict_output in list_results:
        a0 = dict_output["a0"]
        a1 = dict_output["a1"]
        avg_dist = dict_output["avg_dist"]

        idx_a0 = np.where(list_a0 == a0)[0][0]
        idx_a1 = np.where(list_a1 == a1)[0][0]
        # print(idx_a, N_a, idx_alpha_f, N_alpha_f)
        # print(arr_avg_dist.shape)
        arr_avg_dist[idx_a1, idx_a0] = avg_dist

    plt.figure(figsize=(8, 6))
    plt.title(f"neg log average KS distance on line {ell}")
    plt.contourf(AA0, AA1, -np.log10(arr_avg_dist), levels=200)
    plt.colorbar(format=lambda x, _: f"{x:.3f}")
    plt.xlabel(r"mixing interval center $(a_{\ell,1} + a_{\ell,0})/2$")
    plt.ylabel(r"mixing interval radius $(a_{\ell,1} - a_{\ell,0})/2$")
    plt.axvline(log10_f0, c="r", ls="--", label="equal variances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/final_log.PNG")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.title("average KS distance on marginals")
    plt.contourf(AA0, AA1, arr_avg_dist, levels=200)
    plt.colorbar(format=lambda x, _: f"{x:1.2e}")
    plt.xlabel(r"mixing interval center $(a_{\ell,1} + a_{\ell,0})/2$")
    plt.ylabel(r"mixing interval radius $(a_{\ell,1} - a_{\ell,0})/2$")
    plt.axvline(log10_f0, c="r", ls="--", label="equal variances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/final_lin.PNG")
    plt.close()
