from design_baselines.core import Algorithm
from design_bench.core import DesignProblem, Design, Container
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DenseInputLayer(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(DenseInputLayer, self).__init__()
        self.lin = nn.Linear(input_size, hidden_size)
        self.emb = nn.Embedding(num_embeddings, embedding_size)

    def forward(self, *xs):
        emb = self.emb(xs[1])
        return torch.cat((self.lin(xs[0]), emb), dim=1), emb


class DenseResBlock(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size=None):
        super(DenseResBlock, self).__init__()
        self.bn0 = nn.BatchNorm1d(input_size),
        self.relu0 = nn.ReLU(),
        self.lin0 = nn.Linear(input_size, hidden_size),
        self.bn1 = nn.BatchNorm1d(hidden_size),
        self.relu1 = nn.ReLU(),
        self.lin1 = nn.Linear(hidden_size, input_size)

    def forward(self, *xs):

        h = self.

        return xs[0] + self.model(x)


class ConvResBlock(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(ConvResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(),
            nn.Conv2d(input_size, hidden_size, 3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, input_size, 3, padding=1))

    def forward(self, x):
        return x + self.model(x)


class LSGAN(Algorithm):
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
                 hidden_size=512,
                 batch_size=32,
                 training_iterations=2000,
                 discrete_size=1000,
                 latent_dim=256):
        """
        Create a general interface for optimizations algorithms that solve
        model-based optimization problems

        Arguments:

        design_problem: DesignProblem
            A specific design problem that will be solved using the algorithm
            implemented in this class
        """

        Algorithm.__init__(self,
                           design_problem,
                           hidden_size=hidden_size,
                           batch_size=batch_size,
                           training_iterations=training_iterations,
                           discrete_size=discrete_size,
                           latent_dim=latent_dim)

        assert self.design_problem.is_continuous

        # sample bounds for the generator
        self.shift = (self.design_problem.design_space.upper +
                      self.design_problem.design_space.lower) / 2
        self.scale = (self.design_problem.design_space.upper -
                      self.design_problem.design_space.lower) / 2

        generator = nn.Sequential(
            nn.Linear(1 + latent_dim, hidden_size),
            DenseResBlock(hidden_size, hidden_size * 2),
            DenseResBlock(hidden_size, hidden_size * 2),
            DenseResBlock(hidden_size, hidden_size * 2),
            nn.Linear(hidden_size, self.shift.size),
        )

    def solve(self,
              score: np.ndarray = None,
              condition: Container = None) -> Design:
        """
        Solve the design problem by conditioning on inputs and finding a
        design that maximizes the design problem score

        Arguments:

        inputs: Any
            An optimal input used to parameterize the design problem
            For example, conditioning on an attribute like color
        """

        assert score is not None

        x = score
        if condition is not None:

            if condition.is_continuous:
                x = tf.concat([x, condition.cont], axis=-1)

            if condition.is_discrete:
                z = tf.one_hot(condition.disc, self.discrete_size)
                z = tf.reshape(z, [tf.shape(z)[0],
                                   tf.shape(z)[1] * self.discrete_size])
                x = tf.concat([x, z], axis=-1)

        noise = tf.random.normal([1, self.latent_dim])
        sample = self.generator(tf.concat([x, noise], axis=-1),
                                training=True) * self.scale + self.shift

        design = self.design_problem.design_space.sample(n=1)
        design.cont = sample.numpy()
        design.score = score
        design.condition = condition
        return design
