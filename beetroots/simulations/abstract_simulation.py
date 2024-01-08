import abc
import datetime
import os
import sys
import warnings
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import yaml
from cerberus import Validator

from beetroots.inversion.reporting.report_astro import ReportAstro
from beetroots.simulations import data_validation

warnings.filterwarnings("ignore")


class Simulation(abc.ABC):
    def __init__(
        self,
        simu_name: str,
        cloud_name: str,
        params_names: Dict[str, str],
        list_lines_fit: List[str],
        params: dict,
        max_workers: int = 10,
        small_size: int = 16,
        medium_size: int = 20,
        bigger_size: int = 24,
    ):
        # TODO : adapt to include non astro obs
        self.cloud_name = cloud_name
        self.list_lines_fit = list_lines_fit
        self.L = len(list_lines_fit)
        self.simu_name = simu_name

        self.create_empty_output_folders(simu_name, params)
        self.setup_plot_text_sizes(small_size, medium_size, bigger_size)

        self.max_workers = max_workers

        self.list_names = list(params_names.keys())
        self.list_names_plots = list(params_names.values())
        self.D = len(self.list_names)  # Number of physical parameters

        self.list_fixed_values = list(params["forward_model"]["fixed_params"].values())

        try:
            self.pixels_of_interest = params["pixels_of_interest"]
        except:
            print("no pixels of interest")
            self.pixels_of_interest = {}

        self.list_idx_sampling = [
            i
            for i, v in enumerate(params["forward_model"]["fixed_params"].values())
            if v is None
        ]
        self.D_sampling = len(self.list_idx_sampling)

        # number of params used in forward model
        if "kappa" in self.list_names:
            self.D_no_kappa = self.D - 1
        else:
            self.D_no_kappa = self.D * 1

    def setup_plot_text_sizes(
        self,
        small_size: int = 16,
        medium_size: int = 20,
        bigger_size: int = 24,
    ) -> None:
        """Defines text sizes on matplotlib plots"""
        plt.rc("font", size=small_size)  # controls default text sizes
        plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
        plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
        plt.rc("legend", fontsize=small_size)  # legend fontsize
        plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title
        return

    def create_empty_output_folders(self, name: str, params: dict) -> None:
        r"""creates the directories that receive the results of the sampling

        Parameters
        ----------
        name : str
            name of the simulation to be run
        params : dict
            set of params used for the simulation

        Returns
        -------
        tuple of str
            paths of the created directories
        """
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H")

        # path to the outputs dir
        path_ouput_general = f"{os.path.abspath(__file__)}/../../../../outputs"
        path_ouput_general = os.path.abspath(path_ouput_general)

        path_output_sim = f"{path_ouput_general}/{name}_{dt_string}"
        self.path_output_sim = os.path.abspath(path_output_sim)

        self.path_img = path_output_sim + "/img"
        self.path_raw = path_output_sim + "/raw"
        self.path_data_csv = path_output_sim + "/data"
        self.path_data_csv_in = self.path_data_csv + "/input"
        self.path_data_csv_out = self.path_data_csv + "/output"
        self.path_data_csv_out_mcmc = self.path_data_csv_out + "/mcmc"
        self.path_data_csv_out_optim_map = self.path_data_csv_out + "/optim_map"
        self.path_data_csv_out_optim_mle = self.path_data_csv_out + "/optim_mle"

        for folder_path in [
            path_ouput_general,
            self.path_output_sim,
            self.path_img,
            self.path_raw,
            self.path_data_csv,
            self.path_data_csv_in,
            self.path_data_csv_out,
            self.path_data_csv_out_mcmc,
            self.path_data_csv_out_optim_map,
            self.path_data_csv_out_optim_mle,
        ]:
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)

        # empty potentially existing results
        for folder_path in [
            self.path_data_csv_out_mcmc,
            self.path_data_csv_out_optim_map,
            self.path_data_csv_out_optim_mle,
        ]:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)

        # equivalent of copy paste of the config file to output folder
        with open(f"{self.path_data_csv_in}/config_{name}.yaml", "w") as file:
            yaml.dump(params, file)

        return

    @classmethod
    def load_params(cls, path_data_cloud: str) -> dict:
        # print("don't forget to put back sys.argv in load_params")
        if len(sys.argv) < 2:
            print("Please provide the name of the YAML file as an argument.")
            sys.exit(1)

        yaml_file = sys.argv[1]

        # ? Set filename here for debugging.
        # yaml_file = "input_params_nn_direct_N10_fixed_angle_nopmala.yaml"

        with open(os.path.abspath(f"{path_data_cloud}/{yaml_file}")) as f:
            params = yaml.safe_load(f)

        # inputs validation
        v = Validator(data_validation.schema, allow_unknown=True)
        is_input_correct: bool = v.validate(params)

        print(f"is input file correct: {is_input_correct}")

        if not is_input_correct:
            print(f"identified errors: {v.errors}")

        return params

    def generate_report(
        self,
        list_model_names: List[str],
        to_run_optim_map: bool,
        to_run_mcmc: bool,
    ):
        # Create a new report object
        my_report = ReportAstro(
            self.D,
            self.D_sampling,
            self.path_img,
            list_model_names,
            self.pixels_of_interest,
            to_run_optim_map,
            to_run_mcmc,
        )
        my_report.main(
            self.cloud_name,
            self.simu_name,
            self.list_lines_fit,
            self.list_lines_valid,
        )
