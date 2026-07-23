import abc


class SimulationTargetDistributionType(abc.ABC):
    @abc.abstractmethod
    def setup_target_distribution(self):
        pass

    @abc.abstractmethod
    def inversion_mcmc(self):
        pass

    @abc.abstractmethod
    def inversion_optim(self):
        pass
