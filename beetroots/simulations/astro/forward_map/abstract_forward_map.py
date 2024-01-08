import abc


class SimulationForwardMap(abc.ABC):
    @abc.abstractmethod
    def setup_forward_map(self):
        pass
