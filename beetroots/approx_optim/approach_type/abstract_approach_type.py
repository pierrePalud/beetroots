import abc


class ApproachType(abc.ABC):
    @abc.abstractmethod
    def optimization(self):
        pass

    @abc.abstractmethod
    def plot_cost_map(self):
        pass
