from design_bench.core import DesignProblem, Design, Container
import abc


class Algorithm(abc.ABC):
    """
    A general interface for model-based optimization algorithms
    For example, one might use a forward model with gradient descent

    Usage:

    p = DesignProblem(...)
    algorithm = Algorithm(p, ...)
    design = algorithm.solve(...)
    """

    def __init__(self,
                 training_dp: DesignProblem,
                 validation_dp: DesignProblem,
                 **kwargs):
        """
        Create a general interface for optimizations algorithms that solve
        model-based optimization problems

        Arguments:

        training_dp: DesignProblem
            A specific design problem that will be solved using the algorithm
            implemented in this class
        validation_dp: DesignProblem
            A specific design problem that will be used for validation in
            order to prevent overfitting
        """

        self.training_dp = training_dp
        self.validation_dp = validation_dp
        self.__dict__.update(kwargs)

    @property
    def is_continuous(self) -> bool:
        return self.training_dp.is_continuous

    @property
    def is_discrete(self) -> bool:
        return self.training_dp.is_discrete

    @property
    def is_fused(self) -> bool:
        return self.training_dp.is_fused

    @abc.abstractmethod
    def solve(self, condition: Container = None) -> Design:
        """
        Solve the design problem by conditioning on inputs and finding a
        design that maximizes the design problem score

        Arguments:

        inputs: Any
            An optimal input used to parameterize the design problem
            For example, conditioning on an attribute like color
        """

        return NotImplemented
