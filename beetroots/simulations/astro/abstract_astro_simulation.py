import os
import sys
from typing import Dict, List, Optional, Tuple

from beetroots.simulations.abstract_simulation import Simulation


class AstroSimulation(Simulation):
    def __init__(
        self,
        simu_name: str,
        cloud_name: str,
        max_workers: int,
        params_names: Dict[str, str],
        list_lines_fit: List[str],
        yaml_file: str,
        path_data: str,
        path_outputs: str,
        path_models: str,
        forward_model_fixed_params: Dict[str, Optional[float]],
        pixels_of_interest: Dict[int, str] = {},
        small_size: int = 16,
        medium_size: int = 20,
        bigger_size: int = 24,
    ):
        r"""

        Parameters
        ----------
        simu_name : str
            name of the full inversion procedure, used to name the outputs folder
        cloud_name : str
            name of the observed cloud
        max_workers : int
            maximum number of workers to run the program
        params_names : Dict[str, str]
            pairs of names for each parameter, with first the standard name (to be found as title of column in DataFrames) and second a latex name (to be displayed in figures). For instance, for the thermal pressure "P": r"$P_{th}$)"
        list_lines_fit : List[str]
            names of the observables used for the inversion
        max_workers : int, optional
            maximum number of workers that can be used for inversion or results extraction, by default 10
        small_size : int, optional
            size for basic text, axes titles, xticks and yticks, by default 16
        medium_size : int, optional
            size of the axis labels, by default 20
        bigger_size : int, optional
            size of the figure title, by default 24
        path_outputs : str
            path to the output folder to be created
        """
        self.simu_name = simu_name
        r"""str: name of the full inversion procedure, used to name the outputs folder"""

        self.cloud_name = cloud_name
        r"""str: name of the observed cloud"""

        self.max_workers = max_workers
        r"""int: maximum number of workers that can be used for inversion or results extraction"""

        self.list_lines_fit = list_lines_fit
        r"""List[str]: names of the observables used for inversion"""
        self.L = len(list_lines_fit)
        r"""int: number of observables per component :math:`n` used for inversion, e.g., per pixel"""

        self.create_empty_output_folders(
            simu_name,
            path_yaml_file=f"{path_data}/{yaml_file}",
            path_outputs=path_outputs,
        )
        self.setup_plot_text_sizes(small_size, medium_size, bigger_size)

        self.list_names = list(params_names.keys())
        r"""List[str]: names of the physical parameters in files, e.g., as titles of a DataFrame column, e.g., P_th or P for the thermal pressure"""
        self.list_names_plots = list(params_names.values())
        r"""List[str]: names of the physical parameters in figures, e.g., $P_{th}$ for the thermal pressure"""

        self.D = len(self.list_names)  # Number of physical parameters
        r"""int: total number of physical parameters involved in the forward map"""

        self.list_fixed_values = list(forward_model_fixed_params.values())
        r"""List[float | None] list of values for the physical parameters that are not to be samples"""

        self.pixels_of_interest = pixels_of_interest
        r"""Dict[int, str]: pairs of (index, name) for pixels of interest that are to be highlighted in maps"""

        self.list_idx_sampling = [
            i for i, v in enumerate(forward_model_fixed_params.values()) if v is None
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

        self.MODELS_PATH = path_models
        r"""str: path to the folder containing all the already defined and saved models (i.e., polynomials or neural networks)"""

        self.DATA_PATH = path_data
        r"""str: path to the folder containing the yaml input files and the observation data"""

    @classmethod
    def parse_args(cls) -> Tuple[str, str, str, str]:
        """parses the inputs of the command-line, that should contain

        - the name of the input YAML file
        - path to the data folder
        - path to the models folder
        - path to the outputs folder to be created (by default '.')

        Returns
        -------
        str
            name of the input YAML file
        str
            path to the data folder
        str
            path to the models folder
        str
            path to the outputs folder to be created (by default '.')
        """
        if len(sys.argv) < 4:
            raise ValueError(
                "Please provide the following arguments: \n 1) the name of the input YAML file, \n 2) the path to the data folder, \n 3) the path to the models folder, \n 4) the path to the outputs folder to be created (by default '.')"
            )

        yaml_file = sys.argv[1]
        path_data = os.path.abspath(sys.argv[2])
        path_models = os.path.abspath(sys.argv[3])

        path_outputs = (
            os.path.abspath(sys.argv[4]) if len(sys.argv) == 5 else os.path.abspath(".")
        )
        path_outputs += "/outputs"

        print(f"input file name: {yaml_file}")
        print(f"path to data folder: {path_data}")
        print(f"path to models folder: {path_models}")
        print(f"path to outputs folder: {path_outputs}")

        return yaml_file, path_data, path_models, path_outputs
