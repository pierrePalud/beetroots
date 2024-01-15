import os
import time
from typing import Dict, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sensor_loc_forward import SensorLocForwardMap
from sensor_loc_likelihood import SensorLocalizationLikelihood

from beetroots.inversion.results.results_mcmc import ResultsExtractorMCMC
from beetroots.inversion.run.run_mcmc import RunMCMC
from beetroots.modelling.posterior import Posterior
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.sampler.my_sampler import MySampler
from beetroots.sampler.saver.my_saver import MySaver
from beetroots.sampler.utils.my_sampler_params import MySamplerParams
from beetroots.simulations.abstract_simulation import Simulation
from beetroots.space_transform.id_transform import IdScaler


class SensorLocalizationSimulation(Simulation):

    __slots__ = (
        "max_workers",
        "D",
        "L",
        "N",
        "list_names",
        "K",
        "Theta_true_scaled",
    )

    def __init__(
        self,
        params: Dict,
        max_workers: int = 10,
        small_size: int = 16,
        medium_size: int = 20,
        bigger_size: int = 24,
    ):
        self.create_empty_output_folders(params["simu_init"]["simu_name"], params, ".")
        self.setup_plot_text_sizes(small_size, medium_size, bigger_size)

        self.max_workers = max_workers

        self.D = 2
        self.list_names = [r"$" + f"x_{d}" + "$" for d in range(1, self.D + 1)]

    def plot_observation_graph(
        self,
        filename_obs: str,
        lower_bounds_lin,
        upper_bounds_lin,
    ):
        filepath = f"{os.path.dirname(os.path.abspath(__file__))}/data"

        df_sensors = pd.read_csv(
            f"{filepath}/sensors_localizations_rescaled.csv",
            index_col="sensor_id",
        )

        df_obs = pd.read_csv(f"{filepath}/{filename_obs}")
        list_detections = list(
            df_obs.loc[df_obs["observed"], ["sensor_id_1", "sensor_id_2"]].values
        )
        list_detections = [
            tuple(list(elt)) for elt in list_detections if elt[0] < elt[1]
        ]

        plt.figure(figsize=(10, 10))

        # plot edges
        for edge in list_detections:
            id_0, id_1 = edge
            plt.plot(
                [df_sensors.at[id_0, "x"], df_sensors.at[id_1, "x"]],
                [df_sensors.at[id_0, "y"], df_sensors.at[id_1, "y"]],
                "k-",
                linewidth=2,
            )

        # plot known points
        plt.scatter(
            df_sensors.loc[df_sensors["known"], "x"],
            df_sensors.loc[df_sensors["known"], "y"],
            c="r",
            marker="s",
            s=75,
        )

        # plot unknown points
        plt.scatter(
            df_sensors.loc[~df_sensors["known"], "x"],
            df_sensors.loc[~df_sensors["known"], "y"],
            c="b",
            marker="o",
            s=75,
        )

        # low_theta = -0.25
        # low_y = -0.32
        # delta = 1.05 - (-0.32)
        plt.xlim([lower_bounds_lin[0], upper_bounds_lin[0]])
        plt.ylim([lower_bounds_lin[1], upper_bounds_lin[1]])
        # plt.axis("equal")
        # plt.grid()
        # plt.legend()
        plt.savefig(
            f"{self.path_img}/graph_sensors.PNG",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()
        return df_sensors, list_detections

    def _read_mc_values(self, N_MCMC: int, T_BI: int) -> np.ndarray:
        mc_paths = [
            f"{self.path_raw}/standard_sensor_loc/mcmc_{i}/mc_chains.hdf5"
            for i in range(N_MCMC)  # + mc_type
        ]

        for i, mc_path in enumerate(mc_paths):
            with h5py.File(mc_path, "r") as f:
                if i == 0:
                    list_Theta_lin = np.array(f["list_Theta"][T_BI:])

                else:
                    list_Theta_lin = np.concatenate(
                        [list_Theta_lin, np.array(f["list_Theta"][T_BI:])]
                    )
        return list_Theta_lin

    def plot_overlayed_marginals(
        self,
        df_sensors: pd.DataFrame,
        list_detections: list,
        N_MCMC: int,
        T_BI: int,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
    ):

        list_Theta_lin = self._read_mc_values(N_MCMC, T_BI)
        freq = 1

        plt.figure(figsize=(8, 8))
        for n in range(self.N):
            _ = plt.scatter(
                list_Theta_lin[::freq, n, 0].flatten(),
                list_Theta_lin[::freq, n, 1].flatten(),
                label=f"{n}",
                s=4,
            )

        # plot edges
        for edge in list_detections:
            id_0, id_1 = edge
            plt.plot(
                [df_sensors.at[id_0, "x"], df_sensors.at[id_1, "x"]],
                [df_sensors.at[id_0, "y"], df_sensors.at[id_1, "y"]],
                "k-",
                linewidth=1.5,
            )

        # plot known points
        plt.scatter(
            df_sensors.loc[df_sensors["known"], "x"],
            df_sensors.loc[df_sensors["known"], "y"],
            c="r",
            marker="s",
            s=75,
        )

        # plot unknown points
        plt.scatter(
            df_sensors.loc[~df_sensors["known"], "x"],
            df_sensors.loc[~df_sensors["known"], "y"],
            c="b",
            marker="o",
            s=75,
        )

        # low_theta = -0.25
        # low_y = -0.35
        # delta = 1.05 - (-0.32)
        # plt.xlim([low_theta, low_theta + delta])
        # plt.ylim([low_y, low_y + delta])

        list_pos = [-0.25, 0, 0.25, 0.5, 0.75, 1]
        plt.xticks(list_pos, [f"{elt:.2f}" for elt in list_pos])
        plt.yticks(list_pos, [f"{elt:.2f}" for elt in list_pos])

        plt.xlim([lower_bounds_lin[0], upper_bounds_lin[0]])
        plt.ylim([lower_bounds_lin[1], upper_bounds_lin[1]])
        # plt.xticklabels()
        # plt.yticklabels([f"{elt}" for elt in list_pos])
        # plt.axis("equal")
        # plt.grid()
        # plt.legend()

        plt.tight_layout()
        plt.savefig(
            f"{self.path_img}/marginals.PNG",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()

    def read_true_positions(self) -> np.ndarray:
        filename = f"{os.path.dirname(os.path.abspath(__file__))}/data/sensors_localizations_rescaled.csv"
        df_sensors = pd.read_csv(filename, index_col="sensor_id")

        self.L = len(df_sensors)
        self.N = len(df_sensors[~df_sensors["known"]])
        self.K = self.L - self.N

        assert self.K == 3, self.K
        # assert self.N == 8
        # assert self.L == 11

        mask = ~df_sensors["known"]
        self.Theta_true_scaled = df_sensors.loc[mask, ["x", "y"]].values

        Theta_ref = df_sensors.loc[df_sensors["known"], ["x", "y"]].values
        assert Theta_ref.shape == (self.K, self.D)
        return Theta_ref

    def setup_forward_map(self) -> Tuple[IdScaler, SensorLocForwardMap]:
        scaler = IdScaler()

        Theta_ref = self.read_true_positions()
        forward_map = SensorLocForwardMap(
            self.D,
            self.L,
            self.N,
            Theta_ref,
        )
        return scaler, forward_map

    def setup_observation(self, filename_obs: str) -> np.ndarray:
        filename = f"{os.path.dirname(os.path.abspath(__file__))}/data/{filename_obs}"
        df_obs = pd.read_csv(filename)

        y = df_obs.loc[df_obs["sensor_id_1"] >= self.K, "y"].values.reshape(
            (self.N, self.L)
        )
        return y

    def setup_posteriors(
        self,
        filename_obs: str,
        R: float,
        sigma_a: float,
        indicator_margin_scale: float,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
    ) -> Tuple[Dict[str, Posterior], IdScaler]:
        # likelihood
        scaler, forward_map = self.setup_forward_map()
        y = self.setup_observation(filename_obs)

        likelihood_sensor = SensorLocalizationLikelihood(
            forward_map,
            self.N,
            self.L,
            y,
            sigma_a,
            R,
        )

        if isinstance(lower_bounds_lin, list):
            lower_bounds_lin = np.array(lower_bounds_lin)
        if isinstance(upper_bounds_lin, list):
            upper_bounds_lin = np.array(upper_bounds_lin)

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
            list_idx_sampling=np.arange(self.D),
        )

        # posterior
        posterior_ = Posterior(
            self.D,
            self.L,
            self.N,
            likelihood_sensor,
            prior=None,
            prior_spatial=None,
            prior_indicator=prior_indicator,
            separable=False,
        )
        dict_posteriors = {"standard_sensor_loc": posterior_}
        return dict_posteriors, scaler

    def setup(
        self,
        filename_obs: str,
        R: float,
        sigma_a: float,
        indicator_margin_scale: float,
        lower_bounds_lin: np.ndarray,
        upper_bounds_lin: np.ndarray,
    ):
        dict_posteriors, scaler = self.setup_posteriors(
            filename_obs,
            R,
            sigma_a,
            indicator_margin_scale,
            lower_bounds_lin,
            upper_bounds_lin,
        )

        for model_name in list(dict_posteriors.keys()):
            folder_path = f"{self.path_raw}/{model_name}"
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)

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
        start_from: Optional[str] = None,
    ) -> None:
        tps_init = time.time()

        saver_ = MySaver(
            N=self.N,
            D=self.D,
            D_sampling=self.D * 1,
            L=self.L,
            scaler=scaler,
            batch_size=100,
        )

        run_mcmc = RunMCMC(self.path_data_csv_out, self.max_workers)
        run_mcmc.main(
            dict_posteriors=dict_posteriors,
            sampler_=sampler_,
            saver_=saver_,
            scaler=scaler,
            N_runs=N_MCMC,
            max_iter=T_MC,
            T_BI=T_BI,
            path_raw=self.path_raw,
            path_csv_mle=self.path_data_csv_out_optim_mle,
            path_csv_map=self.path_data_csv_out_optim_map,
            start_from=start_from,
            freq_save=freq_save,
        )

        results_mcmc = ResultsExtractorMCMC(
            path_data_csv_out_mcmc=self.path_data_csv_out_mcmc,
            path_img=self.path_img,
            path_raw=self.path_raw,
            N_MCMC=N_MCMC,
            T_MC=T_MC,
            T_BI=T_BI,
            freq_save=freq_save,
            max_workers=self.max_workers,
        )
        for model_name, posterior in dict_posteriors.items():
            results_mcmc.main(
                posterior=posterior,
                model_name=model_name,
                scaler=scaler,
                list_names=self.list_names,
                list_idx_sampling=np.arange(self.D),
                list_fixed_values=[None] * self.D,
                #
                plot_1D_chains=plot_1D_chains,
                plot_2D_chains=plot_2D_chains,
                plot_ESS=plot_ESS,
                #
                plot_comparisons_yspace=False,
                estimator_plot=None,
                analyze_regularization_weight=False,
                list_lines_fit=[f"dist{i}" for i in range(self.L)],
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

    params = SensorLocalizationSimulation.load_params(path_data)

    simulation_sensor = SensorLocalizationSimulation(params)
    dict_posteriors, scaler = simulation_sensor.setup(
        filename_obs=params["filename_obs"],
        **params["likelihood"],
        **params["prior_indicator"],
    )

    sampler_ = MySampler(
        MySamplerParams(**params["sampling_params"]["mcmc"]),
        simulation_sensor.D,
        simulation_sensor.L,
        simulation_sensor.N,
    )
    simulation_sensor.inversion_mcmc(
        dict_posteriors,
        scaler,
        sampler_,
        **params["run_params"]["mcmc"],
    )

    simulation_sensor.setup_plot_text_sizes(26, 26, 26)
    df_sensors, list_detections = simulation_sensor.plot_observation_graph(
        params["filename_obs"],
        params["prior_indicator"]["lower_bounds_lin"],
        params["prior_indicator"]["upper_bounds_lin"],
    )
    simulation_sensor.plot_overlayed_marginals(
        df_sensors,
        list_detections,
        params["run_params"]["mcmc"]["N_MCMC"],
        params["run_params"]["mcmc"]["T_BI"],
        params["prior_indicator"]["lower_bounds_lin"],
        params["prior_indicator"]["upper_bounds_lin"],
    )
