import os
import time
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from gaussian_mixture_likelihood import GaussianMixtureLikelihood
from matplotlib.patches import Ellipse

from beetroots.inversion.results.results_mcmc import ResultsExtractorMCMC
from beetroots.inversion.run.run_mcmc import RunMCMC
from beetroots.modelling.forward_maps.identity import BasicForwardMap
from beetroots.modelling.posterior import Posterior
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.sampler.my_sampler import MySampler
from beetroots.sampler.saver.my_saver import MySaver
from beetroots.sampler.utils.psgldparams import PSGLDParams
from beetroots.simulations.abstract_simulation import Simulation
from beetroots.space_transform.id_transform import IdScaler


def confidence_ellipse(x, cov, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = x[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = x[1]

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def read_gaussian_distribution_file(modes_filename: str):
    """imports the list of modes positions and covariance matrices of the gaussian mixture model to sample from.

    Parameters
    ----------
    modes_filename : str
        name of the file that contains the list of modes positions and covariance matrices of the gaussian mixture model to sample from.

    Returns
    -------
    n_means : int
        number of modes in the gaussian mixture
    list_means : np.ndarray of shape (n_means, D)
        list of modes' positions
    list_cov : np.ndarray of shape (n_means, D, D)
        list of modes' covariance matrices
    """
    D = 2
    path_data = f"{os.path.dirname(os.path.abspath(__file__))}/data"
    path_data = os.path.abspath(path_data)

    df_mixture = pd.read_csv(f"{path_data}/{modes_filename}")

    n_means = len(df_mixture)

    list_means = df_mixture.loc[
        np.arange(n_means), ["x_0", "x_1"]
    ].values  # (n_means, self.D)

    list_cov = np.zeros((n_means, D, D))
    for i in range(n_means):
        list_cov[i, 0, 0] = df_mixture.at[i, "sigma2_0"]
        list_cov[i, 1, 1] = df_mixture.at[i, "sigma2_1"]
        list_cov[i, 1, 0] = df_mixture.at[i, "covariance"]
        list_cov[i, 0, 1] = df_mixture.at[i, "covariance"]

    return n_means, list_means, list_cov


class SimulationGaussianMixture(Simulation):

    __slots__ = (
        "max_workers",
        "n_means",
        "list_means",
        "list_cov",
        "N",
        "D",
        "L",
        "list_names",
        "Theta_true_scaled",
    )

    def __init__(
        self,
        modes_filename: str,
        params: dict,
        max_workers: int = 10,
        small_size: int = 16,
        medium_size: int = 20,
        bigger_size: int = 24,
    ):
        self.create_empty_output_folders("toy_gaussian_mixture", params, ".")
        self.setup_plot_text_sizes(small_size, medium_size, bigger_size)

        self.max_workers = max_workers

        n_means, list_means, list_cov = read_gaussian_distribution_file(
            modes_filename,
        )

        self.n_means = n_means
        self.list_means = list_means
        self.list_cov = list_cov

        self.N = 1
        self.D = list_means.shape[1]
        self.L = self.D * 1

        self.list_names = [r"$" + f"x_{d}" + "$" for d in range(1, self.D + 1)]

    def plot_ellipses(
        self,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax.set_title(
            r"true gaussians with their $2 \sigma$ confidence interval",
        )
        ax.scatter(
            self.list_means[:, 0],
            self.list_means[:, 1],
            marker="+",
            c="k",
        )

        for i, x in enumerate(self.list_means):
            confidence_ellipse(x, self.list_cov[i], ax, 2.0, edgecolor="red")

        ax.set_xlim([lower_bounds_lin[0], upper_bounds_lin[0]])
        ax.set_ylim([lower_bounds_lin[1], upper_bounds_lin[1]])
        ax.set_xlabel(self.list_names[0])
        ax.set_ylabel(self.list_names[1])
        # ax.grid()
        ax.grid()
        fig.tight_layout()
        fig.savefig(f"{self.path_img}/true_gaussians.PNG")
        fig.show()

    def setup_forward_map(self) -> Tuple[IdScaler, BasicForwardMap]:
        """sets both scaler and forward map to the identity

        Returns
        -------
        IdScaler
            identity scaler
        BasicForwardMap
            identity forward map
        """
        return IdScaler(), BasicForwardMap(self.L, self.N)

    def setup_posteriors(
        self,
        indicator_margin_scale: float,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
    ) -> Tuple[dict, IdScaler]:
        # likelihood
        scaler, forward_map = self.setup_forward_map()
        likelihood_ = GaussianMixtureLikelihood(
            forward_map,
            self.D,
            self.list_means,
            self.list_cov,
        )

        # indicator prior
        list_idx_sampling = np.arange(self.D)

        lower_bounds_lin = np.array(lower_bounds_lin)
        upper_bounds_lin = np.array(upper_bounds_lin)

        lower_bounds = scaler.from_lin_to_scaled(
            lower_bounds_lin.reshape((self.N, self.D)),
        ).flatten()
        upper_bounds = scaler.from_lin_to_scaled(
            upper_bounds_lin.reshape((self.N, self.D)),
        ).flatten()
        prior_indicator = SmoothIndicatorPrior(
            self.D,
            self.N,
            indicator_margin_scale,
            lower_bounds,
            upper_bounds,
            list_idx_sampling,
        )

        # posterior
        posterior_ = Posterior(
            self.D,
            self.L,
            self.N,
            likelihood_,
            prior=None,
            prior_spatial=None,
            prior_indicator=prior_indicator,
        )
        dict_posteriors = {"gaussian_mixture": posterior_}
        return dict_posteriors, scaler

    def setup(
        self,
        indicator_margin_scale: float,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
    ):
        dict_posteriors, scaler = self.setup_posteriors(
            indicator_margin_scale,
            lower_bounds_lin,
            upper_bounds_lin,
        )

        for model_name in list(dict_posteriors.keys()):
            folder_path = f"{self.path_raw}/{model_name}"
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)

        if self.D == 2:
            self.plot_ellipses(
                lower_bounds_lin,
                upper_bounds_lin,
            )

        Theta_true_scaled = np.mean(self.list_means, 0)
        self.Theta_true_scaled = Theta_true_scaled.reshape((self.N, self.D))
        return dict_posteriors, scaler

    def inversion_mcmc(
        self,
        dict_posteriors: Dict[str, Posterior],
        scaler: IdScaler,
        sampler_: MySampler,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        plot_1D_chains: bool = True,
        plot_2D_chains: bool = True,
        plot_ESS: bool = True,
        freq_save: int = 1,
        can_run_in_parallel: bool = True,
        start_from: Optional[str] = None,
    ) -> None:
        tps_init = time.time()

        saver_ = MySaver(
            N=self.N,
            D=self.D,
            D_sampling=self.D,
            L=self.L,
            scaler=scaler,
            batch_size=100,
            list_idx_sampling=np.arange(self.D),
        )

        run_mcmc = RunMCMC(self.path_data_csv_out, self.max_workers)
        run_mcmc.main(
            dict_posteriors,
            sampler_,
            saver_,
            scaler,
            N_MCMC,
            T_MC,
            T_BI,
            path_raw=self.path_raw,
            path_csv_mle=self.path_data_csv_out_optim_mle,
            path_csv_map=self.path_data_csv_out_optim_map,
            start_from=start_from,
            freq_save=freq_save,
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
                list_names=self.list_names,
                list_idx_sampling=np.arange(self.D),
                list_fixed_values=[None for _ in range(self.D)],
                #
                plot_1D_chains=plot_1D_chains,
                plot_2D_chains=plot_2D_chains,
                plot_ESS=plot_ESS,
                #
                plot_comparisons_yspace=False,
                estimator_plot=None,
                analyze_regularization_weight=False,
                list_lines_fit=self.list_names,
                Theta_true_scaled=self.Theta_true_scaled * 1,
            )

        duration = time.time() - tps_init  # is seconds
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        msg = "Simulation and analysis finished. Total duration : "
        msg += f"{duration_str} s"
        print(msg)
        return


if __name__ == "__main__":
    path_data = f"{os.path.dirname(os.path.abspath(__file__))}/data"

    params = SimulationGaussianMixture.load_params(path_data)

    simulation_gmm = SimulationGaussianMixture("gaussian_mixture.csv", params)

    sampler_ = MySampler(
        PSGLDParams(**params["sampling_params"]["mcmc"]),
        simulation_gmm.D,
        simulation_gmm.L,
        simulation_gmm.N,
    )

    dict_posteriors, scaler = simulation_gmm.setup(**params["prior_indicator"])

    simulation_gmm.inversion_mcmc(
        dict_posteriors,
        scaler,
        sampler_,
        **params["run_params"]["mcmc"],
    )
