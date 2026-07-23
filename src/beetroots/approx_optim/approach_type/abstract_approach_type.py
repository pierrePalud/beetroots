import abc


class ApproachType(abc.ABC):
    r"""abstract class to define the optimization method for the likelihood parameter adjustment"""

    @abc.abstractmethod
    def optimization(self):
        pass

    @abc.abstractmethod
    def plot_cost_map(self):
        pass
