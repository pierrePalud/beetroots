from typing import List, Union

import numpy as np

from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.sampler.utils.psgldparams import PSGLDParams
from beetroots.simulations.abstract_simulation import Simulation
from beetroots.simulations.astro.forward_map.abstract_poly_reg import (
    SimulationPolynomialReg,
)
from beetroots.simulations.astro.observation.abstract_toy_case import SimulationToyCase
from beetroots.simulations.astro.posterior_type.abstract_hierarchical import (
    SimulationHierarchicalSampler,
)


class SimulationToyCaseNNHierarchicalPosterior(
    Simulation,
    SimulationToyCase,
    SimulationPolynomialReg,
    SimulationHierarchicalSampler,
):
    def setup(
        self,
        forward_model_name: str,
        angle: float,
        #
        sigma_a_float: float,
        sigma_m_float: float,
        omega_float: float,
        #
        indicator_margin_scale: float,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
        #
        with_spatial_prior: bool = True,
        spatial_prior_params: Union[None, SpatialPriorParams] = None,
        list_gaussian_approx_params: List[str] = [],
        list_mixing_model_params: List[str] = [],
    ):
        self.list_lines_valid = []

        scaler, forward_map = self.setup_forward_map(
            forward_model_name=forward_model_name,
            angle=angle,
        )

        sigma_a = sigma_a_float * np.ones((self.N, self.L))
        sigma_m = sigma_m_float * np.ones((self.N, self.L))
        omega = omega_float * np.ones((self.N, self.L))

        syn_map, y = self.setup_observation(
            scaler=scaler,
            forward_map=forward_map,
            sigma_a=sigma_a,
            sigma_m=sigma_m,
            omega=omega,
        )

        # run setup
        dict_posteriors, scaler, prior_indicator_1pix = self.setup_posteriors(
            scaler=scaler,
            forward_map=forward_map,
            y=y,
            sigma_a=sigma_a,
            sigma_m=sigma_m,
            omega=omega,
            syn_map=syn_map,
            with_spatial_prior=with_spatial_prior,
            spatial_prior_params=spatial_prior_params,
            indicator_margin_scale=indicator_margin_scale,
            lower_bounds_lin=lower_bounds_lin,
            upper_bounds_lin=upper_bounds_lin,
            list_gaussian_approx_params=list_gaussian_approx_params,
            list_mixing_model_params=list_mixing_model_params,
        )

        y_valid = None
        sigma_a_valid = None
        omega_valid = None

        return (
            dict_posteriors,
            scaler,
            prior_indicator_1pix,
            y_valid,  # None
            sigma_a_valid,  # None
            omega_valid,  # None
        )


if __name__ == "__main__":
    N_1_side = 10

    sigma_a_float = 1.38715e-10
    sigma_m_float = np.log(1.1)
    omega_float = 3 * sigma_a_float

    indicator_margin_scale = 1e-1
    lower_bounds_lin = np.array([1e-1, 1e5, 1e0, 1e0])
    upper_bounds_lin = np.array([1e1, 1e9, 1e5, 4e1])

    if N_1_side == 10:
        psgld_params_mle = PSGLDParams(
            3e-3, 1e-5, 0.99, np.array([0.2, 0.8]), 250, False, False
        )  # 0.05
        psgld_params_map = PSGLDParams(
            1e-3, 1e-5, 0.99, np.array([0.2, 0.8]), 250, False, False
        )  # 0.2
        psgld_params_mcmc = [
            PSGLDParams(
                1e-10, 1, 0.99, np.array([0.0, 1.0]), 200, True, True
            ),  # p(mtm) = 0
            PSGLDParams(
                1e-6, 1, 0.99, np.array([0.5, 0.5]), 250, True, True
            ),  # p(mtm) = 0.2
        ]
        initial_regu_weights = np.array([1.0, 1.0, 1.0, 1.0])
    elif N_1_side == 30:
        psgld_params_mle = PSGLDParams(
            3e-3, 1e-5, 0.99, np.array([0.2, 0.8]), 250, False, False
        )  # 0.05
        psgld_params_map = PSGLDParams(
            3e-4, 1e-5, 0.99, np.array([0.5, 0.5]), 50, False, False
        )  # 0.2
        psgld_params_mcmc = [
            PSGLDParams(
                1e-10, 1, 0.99, np.array([0.0, 1.0]), 200, True, True
            ),  # p(mtm) = 0
            PSGLDParams(
                1e-6, 1, 0.99, np.array([0.5, 0.5]), 250, True, True
            ),  # p(mtm) = 0.2
        ]
        initial_regu_weights = np.array([10.0, 1.0, 2.0, 4.0])
    elif N_1_side == 64:
        psgld_params_mle = PSGLDParams(
            1e-3, 1e-5, 0.99, np.array([0.2, 0.8]), 250, False, False
        )
        psgld_params_map = PSGLDParams(
            5e-4, 1e-5, 0.99, np.array([0.5, 0.5]), 20, False, False
        )
        psgld_params_mcmc = PSGLDParams(
            3e-5, 1e-5, 0.99, np.array([0.5, 0.5]), 10, True, True
        )
        initial_regu_weights = np.array([10.0, 2.0, 3.0, 4.0])

    elif N_1_side == 90:
        psgld_params_mle = PSGLDParams(
            1e-3, 1e-5, 0.99, np.array([0.2, 0.8]), 250, False, False
        )
        psgld_params_map = PSGLDParams(
            5e-4, 1e-5, 0.99, np.array([0.5, 0.5]), 250, False, False
        )
        psgld_params_mcmc = PSGLDParams(
            3e-6, 1e-5, 0.99, np.array([0.5, 0.5]), 50, True, True
        )

        initial_regu_weights = np.array([10.0, 4.0, 5.0, 3.0])

    elif N_1_side == 100:
        psgld_params_mle = PSGLDParams(
            1e-3, 1e-5, 0.99, np.array([0.2, 0.8]), 250, False, False
        )
        psgld_params_map = PSGLDParams(
            5e-4, 1e-5, 0.99, np.array([0.5, 0.5]), 250, False, False
        )
        psgld_params_mcmc = PSGLDParams(
            1e-6, 1e-5, 0.99, np.array([0.5, 0.5]), 20, True, True
        )
        initial_regu_weights = np.array([10.0, 4.0, 5.0, 3.0])

    else:
        raise NotImplementedError("invalid N")

    # basic example for debugging purposes
    spatial_prior_params = SpatialPriorParams(
        name="L2-laplacian",
        use_next_nearest_neighbours=False,
        initial_regu_weights=initial_regu_weights,
        use_clustering=False,
        n_clusters=None,  # 4,
        cluster_algo=None,  # "spectral_clustering",
    )

    list_lines_fit = [
        "co_v0_j4__v0_j3",
        "co_v0_j5__v0_j4",
        "co_v0_j6__v0_j5",
        "co_v0_j7__v0_j6",
        "co_v0_j8__v0_j7",
        "co_v0_j9__v0_j8",
        "co_v0_j10__v0_j9",
        "co_v0_j11__v0_j10",
        "co_v0_j12__v0_j11",
        "co_v0_j13__v0_j12",
    ]

    simulation = SimulationToyCaseNNHierarchicalPosterior(
        N_1_side**2,
        max_workers=30,
        list_lines_fit=list_lines_fit,
    )

    (
        dict_posteriors,
        scaler,
        prior_indicator_1pix,
        y_valid,  # None
        sigma_a_valid,  # None
        omega_valid,  # None
    ) = simulation.setup(
        forward_model_name="polynomial_regression_deg6",
        angle=0.0,
        #
        sigma_a_float=sigma_a_float,
        sigma_m_float=sigma_m_float,
        omega_float=omega_float,
        #
        indicator_margin_scale=indicator_margin_scale,
        lower_bounds_lin=lower_bounds_lin,
        upper_bounds_lin=upper_bounds_lin,
        #
        with_spatial_prior=True,
        spatial_prior_params=spatial_prior_params,
        #
        list_gaussian_approx_params=[],
        list_mixing_model_params=[],
    )

    simulation.save_and_plot_setup(
        dict_posteriors,
        lower_bounds_lin,
        upper_bounds_lin,
        scaler,
    )

    simulation.inversion_mcmc(
        dict_posteriors=dict_posteriors,
        scaler=scaler,
        psgld_params=psgld_params_mcmc,
        N_MCMC=1,
        T_MC=1000,
        T_BI=100,
        #
        plot_1D_chains=True,
        plot_2D_chains=True,
        plot_ESS=False,
        plot_comparisons_yspace=False,
        #
        batch_size=30,
        freq_save=1,
        start_from=None,
        #
        regu_spatial_N0=np.infty,
        regu_spatial_scale=1.0,
        regu_spatial_vmin=1e-8,
        regu_spatial_vmax=1e8,
    )
