import abc


class Run(abc.ABC):
    r"""abstract class for inversion approach supervision, including sampling, optimization (MAP and MLE)"""

    @abc.abstractmethod
    def prepare_run(self):
        r"""prepares the inversion"""
        pass

    @abc.abstractmethod
    def run(self):
        r"""runs the inversion"""
        pass

    @abc.abstractmethod
    def main(self):
        r"""sequentially calls ``prepare_run`` and ``run``"""
        pass
