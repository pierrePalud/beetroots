import time
from typing import Dict, List, Optional, Union

import numpy as np

from beetroots.inversion.results.results_mcmc import ResultsExtractorMCMC
from beetroots.inversion.results.results_optim_map import ResultsExtractorOptimMAP
from beetroots.inversion.run.run_mcmc import RunMCMC
from beetroots.inversion.run.run_optim_map import RunOptimMAP
from beetroots.modelling.likelihoods.approx_censored_add_mult import (
    MixingModelsLikelihood,
)
from beetroots.modelling.likelihoods.gaussian_censored import CensoredGaussianLikelihood
from beetroots.modelling.posterior import Posterior
from beetroots.modelling.priors.l22_laplacian_prior import L22LaplacianSpatialPrior
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.sampler.my_sampler import MySampler
from beetroots.sampler.saver.my_saver import MySaver
from beetroots.sampler.utils.my_sampler_params import MySamplerParams
from beetroots.simulations.astro.posterior_type.abstract_posterior_type import (
    SimulationPosteriorType,
)
from beetroots.space_transform.transform import MyScaler


class SimulationMySampler(SimulationPosteriorType):
    def setup_posteriors(
        self,
        scaler,
        forward_map,
        y,
        sigma_a,
        sigma_m,
        omega,
        syn_map,
        with_spatial_prior: bool,
        spatial_prior_params: Optional[SpatialPriorParams],
        indicator_margin_scale: float,
        lower_bounds_lin: Union[np.ndarray, List[float]],
        upper_bounds_lin: Union[np.ndarray, List[float]],
        list_gaussian_approx_params: List[bool],
        list_mixing_model_params: List[Dict[str, str]],
    ) -> None:

        if with_spatial_prior:
            assert spatial_prior_params is not None
            prior_spatial = L22LaplacianSpatialPrior(
                spatial_prior_params,
                self.cloud_name,
                N=self.N,
                D=self.D_sampling,
                df=syn_map,
                list_idx_sampling=self.list_idx_sampling,
            )
        else:
            prior_spatial = None

        # indicator prior
        if isinstance(lower_bounds_lin, list):
            lower_bounds_lin = np.array(lower_bounds_lin)
        if isinstance(upper_bounds_lin, list):
            upper_bounds_lin = np.array(upper_bounds_lin)

        lower_bounds = scaler.from_lin_to_scaled(
            lower_bounds_lin.reshape((1, self.D)),
        ).flatten()
        upper_bounds = scaler.from_lin_to_scaled(
            upper_bounds_lin.reshape((1, self.D)),
        ).flatten()
        prior_indicator = SmoothIndicatorPrior(
            self.D_sampling,
            self.N,
            indicator_margin_scale,
            lower_bounds,
            upper_bounds,
            list_idx_sampling=self.list_idx_sampling,
        )
        prior_indicator_1pix = SmoothIndicatorPrior(
            self.D_sampling,
            1,
            indicator_margin_scale,
            lower_bounds,
            upper_bounds,
            list_idx_sampling=self.list_idx_sampling,
        )

        # likelihood

        dict_posteriors = {}
        for i, dict_params in enumerate(list_mixing_model_params):
            model_name = f"mixing_{i}"

            likelihood_mixing = MixingModelsLikelihood(
                forward_map,
                self.D_sampling,
                self.L,
                self.N,
                y,
                sigma_a,
                sigma_m,
                omega,
                path_transition_params=dict_params["path_transition_params"],
                list_lines_fit=self.list_lines_fit * 1,
            )
            # separable+True: if no spatial prior, still possible to
            # run chromatic Gibbs on MTM
            posterior_mixing = Posterior(
                self.D_sampling,
                self.L,
                self.N,
                likelihood=likelihood_mixing,
                prior_spatial=prior_spatial,
                prior_indicator=prior_indicator,
                separable=True,  # only used if no spatial prior
            )
            dict_posteriors[model_name] = posterior_mixing

        for is_raw in list_gaussian_approx_params:
            name = "raw" if is_raw else "transformed"
            model_name = f"gaussian_approx_{name}"

            if is_raw:
                m_a = np.zeros((self.N, self.L))
                s_a = sigma_a * 1
            else:
                m_a = 0  # (np.exp(sigma_m**2 / 2) - 1) * y
                s_a = np.sqrt(
                    y**2
                    # * np.exp(sigma_m**2)
                    * (np.exp(sigma_m**2) - 1)
                    + sigma_a**2
                )

            likelihood_censor = CensoredGaussianLikelihood(
                forward_map,
                self.D_sampling,
                self.L,
                self.N,
                y,
                s_a,
                omega,
                bias=m_a,
            )
            # separable+True: if no spatial prior, still possible to run
            # chromatic Gibbs on MTM
            posterior_censor = Posterior(
                self.D_sampling,
                self.L,
                self.N,
                likelihood=likelihood_censor,
                prior_spatial=prior_spatial,
                prior_indicator=prior_indicator,
                separable=False,
            )
            dict_posteriors[model_name] = posterior_censor

        return dict_posteriors, scaler, prior_indicator_1pix

    def inversion_optim_mle(self):
        pass

    def inversion_optim_map(
        self,
        dict_posteriors: Dict[str, Posterior],
        scaler: MyScaler,
        my_sampler_params: MySamplerParams,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        batch_size: int = 10,
        freq_save: int = 1,
        start_from: Optional[str] = None,
        can_run_in_parallel: bool = True,
    ) -> None:
        tps_init = time.time()

        sampler_ = MySampler(my_sampler_params, self.D_sampling, self.L, self.N)

        saver_ = MySaver(
            N=self.N,
            D=self.D,
            D_sampling=self.D_sampling,
            L=self.L,
            scaler=scaler,
            batch_size=batch_size,
            freq_save=freq_save,
            list_idx_sampling=self.list_idx_sampling,
        )

        run_optim_map = RunOptimMAP(self.path_data_csv_out, self.max_workers)
        run_optim_map.main(
            dict_posteriors,
            sampler_,
            saver_,
            scaler,
            N_MCMC,
            T_MC,
            path_raw=self.path_raw,
            start_from=start_from,
            freq_save=freq_save,
            can_run_in_parallel=can_run_in_parallel,
        )

        results_optim_map = ResultsExtractorOptimMAP(
            self.path_data_csv_out_optim_map,
            self.path_img,
            self.path_raw,
            N_MCMC,
            T_MC,
            T_BI,
            freq_save,
            self.max_workers,
        )
        for model_name, posterior in dict_posteriors.items():
            results_optim_map.main(
                posterior=posterior,
                model_name=model_name,
                scaler=scaler,
                #
                list_idx_sampling=self.list_idx_sampling,
                list_fixed_values=self.list_fixed_values,
                #
                estimator_plot=self.plots_estimator,
                Theta_true_scaled=self.Theta_true_scaled,
            )

        duration = time.time() - tps_init  # is seconds
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        msg = "Simulation and analysis finished. Total duration : "
        msg += f"{duration_str} s\n"
        print(msg)

        list_model_names = list(dict_posteriors.keys())
        return list_model_names

    def inversion_mcmc(
        self,
        dict_posteriors: Dict[str, Posterior],
        scaler: MyScaler,
        my_sampler_params: MySamplerParams,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        plot_1D_chains: bool,
        plot_2D_chains: bool,
        plot_ESS: bool,
        plot_comparisons_yspace: bool,
        #
        batch_size: int = 10,
        freq_save: int = 1,
        start_from: Optional[str] = None,
        #
        regu_spatial_N0: Union[int, float] = np.infty,
        regu_spatial_scale: Optional[float] = 1.0,
        regu_spatial_vmin: Optional[float] = 1e-8,
        regu_spatial_vmax: Optional[float] = 1e8,
        #
        y_valid: Optional[np.ndarray] = None,
        sigma_a_valid: Optional[np.ndarray] = None,
        omega_valid: Optional[np.ndarray] = None,
        sigma_m_valid: Optional[np.ndarray] = None,
        #
        can_run_in_parallel: bool = True,
        point_challenger: Dict = {},
        list_CI: List[int] = [68, 90, 95, 99],
    ) -> None:
        tps_init = time.time()

        sampler_ = MySampler(my_sampler_params, self.D_sampling, self.L, self.N)

        saver_ = MySaver(
            self.N,
            self.D,
            self.D_sampling,
            self.L,
            scaler,
            batch_size=batch_size,
            freq_save=freq_save,
            list_idx_sampling=self.list_idx_sampling,
        )

        run_mcmc = RunMCMC(self.path_data_csv_out, self.max_workers)
        run_mcmc.main(
            dict_posteriors,
            sampler_,
            saver_,
            scaler,
            N_MCMC,
            T_MC,
            T_BI=T_BI,
            path_raw=self.path_raw,
            path_csv_mle=self.path_data_csv_out_optim_mle,
            path_csv_map=self.path_data_csv_out_optim_map,
            start_from=start_from,
            freq_save=freq_save,
            #
            regu_spatial_N0=regu_spatial_N0,
            regu_spatial_scale=regu_spatial_scale,
            regu_spatial_vmin=regu_spatial_vmin,
            regu_spatial_vmax=regu_spatial_vmax,
            #
            can_run_in_parallel=can_run_in_parallel,
        )

        results_mcmc = ResultsExtractorMCMC(
            self.path_data_csv_out_mcmc,
            self.path_img,
            self.path_raw,
            N_MCMC,
            T_MC,
            T_BI,
            freq_save,
            self.max_workers,
        )
        for model_name, posterior in dict_posteriors.items():
            results_mcmc.main(
                posterior=posterior,
                model_name=model_name,
                scaler=scaler,
                list_names=self.list_names_plots,
                list_idx_sampling=self.list_idx_sampling,
                list_fixed_values=self.list_fixed_values,
                #
                plot_1D_chains=plot_1D_chains,
                plot_2D_chains=plot_2D_chains,
                plot_ESS=plot_ESS,
                plot_comparisons_yspace=plot_comparisons_yspace,
                #
                estimator_plot=self.plots_estimator,
                analyze_regularization_weight=np.isfinite(regu_spatial_N0),
                list_lines_fit=self.list_lines_fit,
                Theta_true_scaled=self.Theta_true_scaled,
                list_lines_valid=self.list_lines_valid,
                y_valid=y_valid,
                sigma_a_valid=sigma_a_valid,
                omega_valid=omega_valid,
                sigma_m_valid=sigma_m_valid,
                point_challenger=point_challenger,
                list_CI=list_CI,
            )

        duration = time.time() - tps_init  # is seconds
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        msg = "Simulation and analysis finished. Total duration : "
        msg += f"{duration_str} s\n"
        print(msg)

        list_model_names = list(dict_posteriors.keys())
        return list_model_names
