import abc
from typing import List


class ResultsUtil(abc.ABC):
    """abstract class for the extraction of a specific result from the data saved during an inversion. All the daughter classes have three key methods:

    * ``read_data``, to read the data necessary for the computation of the specific result of interest
    * ``create_folders``, to create the folders where the specific result is to be saved
    * ``main``, to run the two previous methods and compute the specific result
    """

    @abc.abstractmethod
    def read_data(self, list_chains_folders: List[str], **kwargs):
        r"""read the data necessary for the computation of the specific result of interest

        Parameters
        ----------
        list_chains_folders : List[int]
            list of the paths to the folders containing the raw results
        """
        pass

    @abc.abstractmethod
    def create_folders(self, **kwargs) -> str:
        r"""create the folder where the specific result is to be saved

        Returns
        -------
        str
            path to the folder where the specific result is to be saved
        """
        pass

    @abc.abstractmethod
    def main(self, **kwargs):
        r"""runs the two previous methods and compute the specific result"""
        pass
