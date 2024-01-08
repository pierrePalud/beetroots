import abc


class Run(abc.ABC):
    @abc.abstractmethod
    def prepare_run(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def main(self):
        pass
