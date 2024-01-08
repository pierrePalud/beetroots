import abc


class ResultsExtractor(abc.ABC):
    @abc.abstractmethod
    def read_estimator(self, path_data_csv_out: str):
        pass

    @abc.abstractmethod
    def main(self):
        pass
