import time
from typing import Dict, List, Optional, Union

import numpy as np

from beetroots.inversion.results.results_mcmc_hierarchical import (
    ResultsExtractorHierarchicalMCMC,
)
from beetroots.inversion.run.run_mcmc import RunMCMC
from beetroots.modelling.forward_maps.identity import BasicForwardMap

# from beetroots.inversion.run.run_optim_map import RunOptimMAP
from beetroots.modelling.likelihoods.gaussian_censored import CensoredGaussianLikelihood
from beetroots.modelling.likelihoods.log_normal import LogNormalLikelihood
from beetroots.modelling.posterior import Posterior
from beetroots.modelling.priors.l22_laplacian_prior import L22LaplacianSpatialPrior
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.sampler.hierarchical_sampler import HierarchicalSampler
from beetroots.sampler.my_sampler import MySampler
from beetroots.sampler.saver.hierarchical_saver import HierarchicalSaver
from beetroots.sampler.utils.psgldparams import PSGLDParams
from beetroots.simulations.astro.posterior_type.abstract_posterior_type import (
    SimulationPosteriorType,
)
from beetroots.space_transform.transform import MyScaler


class SimulationHierarchicalSampler(SimulationPosteriorType):
    def setup_posteriors(
        self,
        scaler: float,
        forward_map: float,
        y,
        sigma_a,
        sigma_m,
        omega,
        syn_map,
        with_spatial_prior: bool,
        spatial_prior_params: SpatialPriorParams,
        indicator_margin_scale: float,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
        list_gaussian_approx_params: List[bool],
        list_mixing_model_params: List,
    ) -> None:
        assert len(list_gaussian_approx_params) == 0
        assert len(list_mixing_model_params) == 0

        if with_spatial_prior:
            prior_spatial = L22LaplacianSpatialPrior(
                spatial_prior_params,
                self.cloud_name,
                self.D,
                self.N,
                df=syn_map,
            )
        else:
            prior_spatial = None

        # indicator prior
        lower_bounds = scaler.from_lin_to_scaled(
            lower_bounds_lin.reshape((1, self.D)),
        ).flatten()
        upper_bounds = scaler.from_lin_to_scaled(
            upper_bounds_lin.reshape((1, self.D)),
        ).flatten()
        prior_indicator = SmoothIndicatorPrior(
            self.D,
            self.N,
            indicator_margin_scale,
            lower_bounds,
            upper_bounds,
        )
        prior_indicator_1pix = SmoothIndicatorPrior(
            self.D,
            1,
            indicator_margin_scale,
            lower_bounds,
            upper_bounds,
        )

        # likelihood
        likelihood_u = CensoredGaussianLikelihood(
            BasicForwardMap(self.L, self.N),  # identity: D = 1 here
            self.L,  # ! D = 1 for this object. to be double checked
            self.L,
            self.N,
            y,
            sigma_a,
            omega,
        )
        prior_u = LogNormalLikelihood(
            forward_map,
            self.D,  # ! size of the parameter of the log-normal distribution
            self.L,
            self.N,
            y,  # TODO: could put 0s instead
            sigma_m,
        )

        conditional_u = Posterior(
            self.L,
            self.L,
            self.N,
            likelihood_u,
            prior=prior_u,
        )

        likelihood_Theta = LogNormalLikelihood(
            forward_map,  # TODO: check whether this is the right object!
            self.D,
            self.L,
            self.N,
            y,  # TODO: could put 0s instead
            sigma_m,
        )

        conditional_Theta = Posterior(
            self.D,
            self.L,
            self.N,
            likelihood_Theta,
            prior_spatial=prior_spatial,  # ! seems to create a problem
            prior_indicator=prior_indicator,
        )
        dict_posteriors = {"hierarchical": [conditional_u, conditional_Theta]}
        return dict_posteriors, scaler, prior_indicator_1pix

    def inversion_optim_mle(self):
        raise NotImplementedError("Yet to be implemented")

    def inversion_optim_map(
        self,
        dict_posteriors: Dict[str, Posterior],
        scaler: MyScaler,
        sampler_: MySampler,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        batch_size: int = 10,
        freq_save: int = 1,
        start_from: Optional[str] = None,
    ) -> None:
        raise NotImplementedError("Yet to be implemented")

    def inversion_mcmc(
        self,
        dict_posteriors: Dict[str, Posterior],
        scaler: MyScaler,
        psgld_params: List[PSGLDParams],
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
    ) -> None:
        tps_init = time.time()

        sampler_ = HierarchicalSampler(
            psgld_params=psgld_params,
            D=self.D,
            L=self.L,
            N=self.N,
        )

        saver_ = HierarchicalSaver(
            self.N,
            self.D,
            self.L,
            scaler,
            batch_size=batch_size,
            freq_save=freq_save,
        )

        run_mcmc = RunMCMC(self.path_data_csv_out, self.max_workers)
        run_mcmc.main(
            dict_posteriors,
            sampler_,
            saver_,
            scaler,
            N_MCMC,
            T_MC,
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
        )

        results_mcmc = ResultsExtractorHierarchicalMCMC(
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
                posterior,
                model_name,
                scaler,
                self.list_names_plots,
                #
                plot_1D_chains,
                plot_2D_chains,
                plot_ESS,
                plot_comparisons_yspace,
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
            )

        duration = time.time() - tps_init  # is seconds
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        msg = "Simulation and analysis finished. Total duration : "
        msg += f"{duration_str} s\n"
        print(msg)
        return
