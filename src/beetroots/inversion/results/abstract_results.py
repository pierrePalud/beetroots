import abc


class ResultsExtractor(abc.ABC):
    r"""extractor of the results of an inversion"""

    @abc.abstractmethod
    def main(self, **kwargs):
        pass
