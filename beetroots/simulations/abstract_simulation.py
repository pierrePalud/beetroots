import abc
import datetime
import os
import shutil
import sys
import warnings
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import yaml
from cerberus import Validator

warnings.filterwarnings("ignore")


class Simulation(abc.ABC):
    r"""abstract class for the main class of the inversion. Its instanciations set up, launch and analyze the results of inversions."""

    def setup_plot_text_sizes(
        self,
        small_size: int = 16,
        medium_size: int = 20,
        bigger_size: int = 24,
    ) -> None:
        r"""defines text sizes on matplotlib plots

        Parameters
        ----------
        small_size : int, optional
            size for basic text, axes titles, xticks and yticks, by default 16
        medium_size : int, optional
            size of the axis labels, by default 20
        bigger_size : int, optional
            size of the figure title, by default 24
        """
        plt.rc("font", size=small_size)  # controls default text sizes
        plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
        plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
        plt.rc("legend", fontsize=small_size)  # legend fontsize
        plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title
        return

    def create_empty_output_folders(
        self, simu_name: str, path_yaml_file: str, path_outputs: str
    ) -> None:
        r"""creates the directories that receive the results of the sampling, and saves the ``.yaml`` input file there for reproducibility

        Parameters
        ----------
        simu_name : str
            name of the simulation to be run
        path_yaml_file : str
            path of the folder containing the data and yaml files
        path_outputs: str
            folder where to write outputs
        """
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H")

        # path to the outputs dir
        path_output_sim = f"{path_outputs}/{simu_name}_{dt_string}"
        self.path_output_sim = os.path.abspath(path_output_sim)
        r"""str: path to the root folder of the inversion results, e.g., ``./outputs/simu1``"""

        self.path_img = path_output_sim + "/img"
        r"""str: path to the image folder in the inversion results, e.g., ``./outputs/simu1/img``"""
        self.path_raw = path_output_sim + "/raw"
        r"""str: path to the folder containing the raw inversion results, i.e., the ``.hdf5`` files, e.g., ``./outputs/simu1/raw``"""
        self.path_data_csv = path_output_sim + "/data"
        r"""str: path to the data folder in the inversion results, e.g., ``./outputs/simu1/data``"""
        self.path_data_csv_in = self.path_data_csv + "/input"
        r"""str: path to the input data folder in the inversion results, e.g., ``./outputs/simu1/data/input``"""
        self.path_data_csv_out = self.path_data_csv + "/output"
        r"""str: path to the output data folder in the inversion results, e.g., ``./outputs/simu1/data/output``"""
        self.path_data_csv_out_mcmc = self.path_data_csv_out + "/mcmc"
        r"""str: path to the mcmc output data folder in the inversion results, e.g., ``./outputs/simu1/data/output/mcmc``"""
        self.path_data_csv_out_optim_map = self.path_data_csv_out + "/optim_map"
        r"""str: path to the MAP estimation (with optimization approach) output data folder in the inversion results, e.g., ``./outputs/simu1/data/output/optim_map``"""
        self.path_data_csv_out_optim_mle = self.path_data_csv_out + "/optim_mle"
        r"""str: path to the MLE estimation (with optimization approach) output data folder in the inversion results, e.g., ``./outputs/simu1/data/output/optim_mle``"""

        # create the folders if necessary
        for folder_path in [
            path_outputs,
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
        shutil.copyfile(path_yaml_file, f"{self.path_data_csv_in}/input_file.yaml")

        return

    @classmethod
    def load_params(cls, path_data_cloud: str, yaml_file: str) -> dict:
        r"""

        Parameters
        ----------
        path_data_cloud : str
            path to the folder containing the ``.yaml`` input file. This folder should also contain the data necessary to set up the inversion.
        yaml_file : str
            name of the ``.yaml`` input file to be read

        Returns
        -------
        dict
            content of the ``.yaml`` input file
        """
        with open(os.path.abspath(f"{path_data_cloud}/{yaml_file}")) as f:
            params = yaml.safe_load(f)

        return params

    @classmethod
    @abc.abstractmethod
    def parse_args(cls):
        pass

    @classmethod
    def check_input_params_file(cls, params: dict, schema: dict) -> None:
        r"""checks the validity of the params contained in the ``.yaml`` input file using the cerberus python package

        Parameters
        ----------
        params : dict
            content of the ``.yaml`` file
        schema : dict
            cerberus validation schema
        """
        # inputs validation
        v = Validator(schema, allow_unknown=True)
        is_input_correct: bool = v.validate(params)

        print(f"is input file correct: {is_input_correct}")

        if not is_input_correct:
            print(f"identified errors: {v.errors}")

        return None
