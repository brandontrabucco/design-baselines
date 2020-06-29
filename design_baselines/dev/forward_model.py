from design_baselines.core import Algorithm
from design_bench.core import DesignProblem, Design
from typing import Any
import tensorflow as tf


class ForwardModel(Algorithm):
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
                 num_layers=2,
                 hidden_size=512,
                 batch_size=32,
                 training_iterations=10000):
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
                           num_layers=num_layers,
                           hidden_size=hidden_size)

        assert self.design_problem.is_continuous

        layers = []
        for i in range(self.num_layers):
            layers.extend([tf.keras.layers.Dense(self.hidden_size),
                           tf.keras.layers.Activation('relu')])
        layers.append(tf.keras.layers.Dense(1))
        self.m = tf.keras.Sequential(layers)

        optim = tf.keras.optimizers.Adam()
        for i in range(training_iterations):
            design = self.design_problem.sample(n=batch_size)
            print(design.cont[0].tolist())

            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(
                    tf.keras.losses.logcosh(
                        design.score, self.m(design.cont)))
                print(f"iteration {i} loss {loss.numpy()}")

            grads = tape.gradient(
                loss, self.m.trainable_variables)
            optim.apply_gradients(
                zip(grads, self.m.trainable_variables))

    def solve(self, inputs: Any = None) -> Design:
        """
        Solve the design problem by conditioning on inputs and finding a
        design that maximizes the design problem score

        Arguments:

        inputs: Any
            An optimal input used to parameterize the design problem
            For example, conditioning on an attribute like color
        """

        design = self.design_problem.design_space.sample(n=1)
        x = tf.Variable(tf.convert_to_tensor(design.cont))

        optim = tf.keras.optimizers.Adam()
        for i in range(100):

            with tf.GradientTape() as tape:
                loss = -tf.reduce_mean(self.m(x))

            grads = tape.gradient(loss, [x])
            optim.apply_gradients(zip(grads, [x]))

        design.cont = x.numpy()
        design.score = self.m(design.cont).numpy()
        return design
