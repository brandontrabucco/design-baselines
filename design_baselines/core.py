from design_bench.core import DesignProblem, Design
from typing import Any
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
                 design_problem: DesignProblem,
                 **kwargs):
        """
        Create a general interface for optimizations algorithms that solve
        model-based optimization problems

        Arguments:

        design_problem: DesignProblem
            A specific design problem that will be solved using the algorithm
            implemented in this class
        """

        self.design_problem = design_problem
        self.__dict__.update(kwargs)

    @property
    def is_continuous(self) -> bool:
        return self.design_problem.is_continuous

    @property
    def is_discrete(self) -> bool:
        return self.design_problem.is_discrete

    @property
    def is_fused(self) -> bool:
        return self.design_problem.is_fused

    @abc.abstractmethod
    def solve(self, inputs: Any = None) -> Design:
        """
        Solve the design problem by conditioning on inputs and finding a
        design that maximizes the design problem score

        Arguments:

        inputs: Any
            An optimal input used to parameterize the design problem
            For example, conditioning on an attribute like color
        """

        return NotImplemented
