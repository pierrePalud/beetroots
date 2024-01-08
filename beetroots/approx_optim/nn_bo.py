from typing import List, Union

import numpy as np
from scipy.stats import gaussian_kde

from beetroots.approx_optim.abstract_approx_optim import ApproxParamsOptim
from beetroots.approx_optim.approach_type.bo import BayesianOptimizationApproach
from beetroots.approx_optim.forward_map.nn import ApproxOptimNN


class ApproxParamsOptimNNBO(
    ApproxParamsOptim, ApproxOptimNN, BayesianOptimizationApproach
):
    def main(
        self,
        dict_forward_model: str,
        lower_bounds_lin: Union[np.ndarray, List],
        upper_bounds_lin: Union[np.ndarray, List],
        n_iter: int,
    ):
        assert isinstance(forward_model_name, str)
        assert isinstance(angle, float)
        assert isinstance(lower_bounds_lin, (np.ndarray, list))
        assert isinstance(upper_bounds_lin, (np.ndarray, list))
        assert isinstance(n_iter, int) and n_iter > 0

        if isinstance(lower_bounds_lin, list):
            lower_bounds_lin = np.array(lower_bounds_lin)
        if isinstance(lower_bounds_lin, list):
            lower_bounds_lin = np.array(lower_bounds_lin)

        (
            log10_f0,
            bounds_a0_low,
            bounds_a0_high,
            bounds_a1_low,
            bounds_a1_high,
        ) = self.setup_params_bounds()

        log10_f_Theta = self.compute_log10_f_Theta(
            dict_forward_model,
            lower_bounds_lin,
            upper_bounds_lin,
        )
        log10_f_Theta_low = log10_f_Theta.min(axis=0)  # (L,)
        log10_f_Theta_high = log10_f_Theta.max(axis=0)  # (L,)

        list_log10_f_grid = np.zeros((self.log10_f_grid_size, self.L))
        pdf_kde_log10_f_Theta = np.zeros((self.log10_f_grid_size, self.L))
        for ell in range(self.L):
            list_log10_f_grid[:, ell] = np.linspace(
                log10_f_Theta_low[ell],
                log10_f_Theta_high[ell],
                self.log10_f_grid_size,
            )
            kde_log10_f_Theta = gaussian_kde(log10_f_Theta[:, ell])
            pdf_kde_log10_f_Theta[:, ell] = kde_log10_f_Theta.pdf(
                list_log10_f_grid[:, ell],
            )

        for ell in range(self.L):
            print(f"starting line {ell} ({self.list_lines[ell]})")
            self.plot_hist_log10_f_Theta(
                log10_f_Theta[:, ell],
                log10_f_Theta_low[ell],
                log10_f_Theta_high[ell],
                list_log10_f_grid[:, ell],
                pdf_kde_log10_f_Theta[:, ell],
                ell,
            )

            for n in range(self.N):
                pbounds = {
                    "a0": (bounds_a0_low[n, ell], bounds_a0_high[n, ell]),
                    "a1": (bounds_a1_low, bounds_a1_high),
                }
                self.save_setup_to_json(n, ell, pbounds)
                self.optimization(
                    first_points=[(log10_f0[n, ell], 0.5)],
                    init_points=3,
                    n_iter=n_iter,  # 50
                    list_log10_f_grid=list_log10_f_grid[:, ell],
                    pdf_kde_log10_f_Theta=pdf_kde_log10_f_Theta[:, ell],
                    pbounds=pbounds,
                    sigma_a_val=self.sigma_a[n, ell],
                    sigma_m_val=self.sigma_m[n, ell],
                    n=n,
                    ell=ell,
                )

        df_best = self.extract_optimal_params()
        df_best = df_best.set_index(["n", "ell"])

        for n in range(self.N):
            for ell in range(self.L):
                best_point = df_best.loc[(n, ell), ["a0_best", "a1_best"]].values
                self.plot_hist_log10_f_Theta_with_optim_results(
                    log10_f_Theta=log10_f_Theta[:, ell],
                    log10_f_Theta_low=log10_f_Theta_low[ell],
                    log10_f_Theta_high=log10_f_Theta_high[ell],
                    list_log10_f_grid=list_log10_f_grid[:, ell],
                    pdf_kde_log10_f_Theta=pdf_kde_log10_f_Theta[:, ell],
                    n=n,
                    ell=ell,
                    best_point=best_point,
                )

        self.plots_postprocessing(
            bounds_a0_low,
            bounds_a0_high,
            bounds_a1_low,
            bounds_a1_high,
            n_iter,
        )
