import abc


class SimulationPosteriorType(abc.ABC):
    @abc.abstractmethod
    def setup_posteriors(self):
        pass

    @abc.abstractmethod
    def inversion_mcmc(self):
        pass

    @abc.abstractmethod
    def inversion_optim_map(self):
        pass

    @abc.abstractmethod
    def inversion_optim_mle(self):
        pass
