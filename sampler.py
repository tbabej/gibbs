from logger import LoggerMixin


class IsingSampler(LoggerMixin):
    """
    A general class that represents a backend capable of sampling IsingModels.
    """

    def sample(self, model, temperature=1):
        """
        Given a model formulation in the form of IsingModel, return the
        IsingSamples obtained.
        """

        raise NotImplementedError
