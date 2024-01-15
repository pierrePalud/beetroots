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

warnings.filterwarnings("ignore")


class Simulation(abc.ABC):
    r"""abstract class for the main class of the inversion. Its instanciations set up, launch and analyze the results of inversions."""

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
        folder_path: Optional[str] = None,
    ):
        r"""

        Parameters
        ----------
        simu_name : str
            name of the full inversion procedure, used to name the outputs folder
        cloud_name : str
            name of the observed cloud
        params_names : Dict[str, str]
            pairs of names for each parameter, with first the standard name (to be found as title of column in DataFrames) and second a latex name (to be displayed in figures). For instance, for the thermal pressure "P": r"$P_{th}$)"
        list_lines_fit : List[str]
            names of the observables used for the inversion
        params : dict
            content of the ``.yaml`` input file
        max_workers : int, optional
            maximum number of workers that can be used for inversion or results extraction, by default 10
        small_size : int, optional
            size for basic text, axes titles, xticks and yticks, by default 16
        medium_size : int, optional
            size of the axis labels, by default 20
        bigger_size : int, optional
            size of the figure title, by default 24
        folder_path : Optional[str], optional
            path to the output folder, by default None
        """
        self.cloud_name = cloud_name
        r"""str: name of the observed cloud"""
        self.list_lines_fit = list_lines_fit
        r"""List[str]: names of the observables used for inversion"""
        self.L = len(list_lines_fit)
        r"""int: number of observables per component :math:`n` used for inversion, e.g., per pixel"""
        self.simu_name = simu_name
        r"""str: name of the full inversion procedure, used to name the outputs folder"""

        self.create_empty_output_folders(simu_name, params, folder_path)
        self.setup_plot_text_sizes(small_size, medium_size, bigger_size)

        self.max_workers = max_workers
        r"""int: maximum number of workers that can be used for inversion or results extraction"""

        self.list_names = list(params_names.keys())
        r"""List[str]: names of the physical parameters in files, e.g., as titles of a DataFrame column, e.g., P_th or P for the thermal pressure"""
        self.list_names_plots = list(params_names.values())
        r"""List[str]: names of the physical parameters in figures, e.g., $P_{th}$ for the thermal pressure"""

        self.D = len(self.list_names)  # Number of physical parameters
        r"""int: total number of physical parameters"""

        self.list_fixed_values = list(params["forward_model"]["fixed_params"].values())
        r"""List[float | None] list of values for the physical parameters that are not to be samples"""

        try:
            pixels_of_interest = params["pixels_of_interest"]
        except:
            print("no pixels of interest")
            pixels_of_interest = {}
        self.pixels_of_interest = pixels_of_interest
        r"""Dict[int, str]: pairs of (index, name) for pixels of interest that are to be highlighted in maps"""

        self.list_idx_sampling = [
            i
            for i, v in enumerate(params["forward_model"]["fixed_params"].values())
            if v is None
        ]
        r"""List[int]: indices of the physical parameters that are to be sampled (the other ones are fixed)"""

        self.D_sampling = len(self.list_idx_sampling)
        r"""int: number of physical parameters to be sampled"""

        # number of params used in forward model
        if "kappa" in self.list_names:
            D_no_kappa = self.D - 1
        else:
            D_no_kappa = self.D * 1
        self.D_no_kappa = D_no_kappa * 1
        r"""number of physical parameters excluding the scaling parameter :math:`\kappa` (used in astrophysical applications). When "kappa" is not in ``list_names``, then ``D_no_kappa = D``"""

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
        self, name: str, params: dict, folder_path: Optional[str] = None
    ) -> None:
        r"""creates the directories that receive the results of the sampling, and saves the ``.yaml`` input file there for reproducibility

        Parameters
        ----------
        name : str
            name of the simulation to be run
        params : dict
            set of params used for the simulation
        folder_path: str, optional
            folder where to write outputs, by default None

        Returns
        -------
        tuple of str
            paths of the created directories
        """
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H")

        # path to the outputs dir
        if folder_path is None:
            path_ouput_general = f"{os.path.abspath(__file__)}/../../../outputs"
        else:
            path_ouput_general = f"{folder_path}/outputs"

        path_ouput_general = os.path.abspath(path_ouput_general)

        path_output_sim = f"{path_ouput_general}/{name}_{dt_string}"
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
        r"""

        Parameters
        ----------
        path_data_cloud : str
            path to the folder containing the ``.yaml`` input file. This folder should also contain the data necessary to set up the inversion.

        Returns
        -------
        dict
            content of the ``.yaml`` input file
        """
        # print("don't forget to put back sys.argv in load_params")
        if len(sys.argv) < 2:
            print("Please provide the name of the YAML file as an argument.")
            sys.exit(1)

        yaml_file = sys.argv[1]

        # ? Set a filename here for debugging.
        # yaml_file = "input_params_nn_direct_N10_fixed_angle_nopmala.yaml"

        with open(os.path.abspath(f"{path_data_cloud}/{yaml_file}")) as f:
            params = yaml.safe_load(f)

        return params

    @classmethod
    def check_input_params_file(cls, params: dict, schema: dict) -> None:
        # inputs validation
        v = Validator(schema, allow_unknown=True)
        is_input_correct: bool = v.validate(params)

        print(f"is input file correct: {is_input_correct}")

        if not is_input_correct:
            print(f"identified errors: {v.errors}")

        return None

    def generate_report(
        self,
        list_model_names: List[str],
        to_run_optim_map: bool,
        to_run_mcmc: bool,
    ):
        r"""gathers the results of the inversion, such as images, in a single ``.pdf`` file to simplify the result understanding

        .. warning::

            Unfinished method.
            Should only be used for astrophysical applications.

        Parameters
        ----------
        list_model_names : List[str]
            names of the models to extract. An inversion relies on one model only.
        to_run_optim_map : bool
            wether the MAP estimation with an optimization procedure was run, and thus wether the report should display the associated results
        to_run_mcmc : bool
            wether an MCMC-based inference was run, and thus wether the report should display the associated results
        """
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
