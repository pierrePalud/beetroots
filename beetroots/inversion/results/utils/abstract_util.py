import abc


class ResultsUtil(abc.ABC):
    @abc.abstractmethod
    def read_data(self, list_chains_folders):
        pass

    @abc.abstractmethod
    def create_folders(self):
        pass

    @abc.abstractmethod
    def main(self):
        pass
