from design_baselines.core import Algorithm
from design_bench.core import DesignProblem, Design, Container
import tensorflow as tf
import numpy as np


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
                 training_iterations=10000,
                 discrete_size=1000):
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
                           hidden_size=hidden_size,
                           batch_size=batch_size,
                           training_iterations=training_iterations,
                           discrete_size=discrete_size)

        assert self.design_problem.is_continuous

        layers = []
        for i in range(self.num_layers):
            layers.extend([tf.keras.layers.Dense(self.hidden_size),
                           tf.keras.layers.Activation('relu')])
        layers.append(tf.keras.layers.Dense(1))
        self.m = tf.keras.Sequential(layers)

        optim = tf.keras.optimizers.Adam(learning_rate=0.00001)
        for i in range(training_iterations):
            design = self.design_problem.sample(n=batch_size)
            design.score = np.nan_to_num(design.score)

            with tf.GradientTape() as tape:

                x = design.cont
                if design.condition is not None:

                    if design.condition.is_continuous:
                        x = tf.concat([x, design.condition.cont], axis=-1)

                    if design.condition.is_discrete:
                        z = tf.one_hot(design.condition.disc, self.discrete_size)
                        z = tf.reshape(z, [tf.shape(z)[0],
                                           tf.shape(z)[1] * self.discrete_size])
                        x = tf.concat([x, z], axis=-1)

                loss = tf.reduce_mean(
                    tf.keras.losses.logcosh(
                        design.score, self.m(x)))
                print(f"iteration {i} loss {loss.numpy()}")

            grads = tape.gradient(
                loss, self.m.trainable_variables)
            optim.apply_gradients(
                zip(grads, self.m.trainable_variables))

    def solve(self, condition: Container = None) -> Design:
        """
        Solve the design problem by conditioning on inputs and finding a
        design that maximizes the design problem score

        Arguments:

        inputs: Any
            An optimal input used to parameterize the design problem
            For example, conditioning on an attribute like color
        """

        design = self.design_problem.design_space.sample(n=1)
        design.condition = condition
        design.cont = tf.Variable(tf.convert_to_tensor(design.cont))

        optim = tf.keras.optimizers.Adam()
        for i in range(100):

            with tf.GradientTape() as tape:

                x = design.cont
                if design.condition is not None:

                    if design.condition.is_continuous:
                        x = tf.concat([x, design.condition.cont], axis=-1)

                    if design.condition.is_discrete:
                        z = tf.one_hot(design.condition.disc, self.discrete_size)
                        z = tf.reshape(z, [tf.shape(z)[0],
                                           tf.shape(z)[1] * self.discrete_size])
                        x = tf.concat([x, z], axis=-1)

                loss = -tf.reduce_mean(self.m(x))

            grads = tape.gradient(loss, [design.cont])
            optim.apply_gradients(zip(grads, [design.cont]))

            design.cont.assign(tf.clip_by_value(
                design.cont,
                self.design_problem.design_space.lower,
                self.design_problem.design_space.upper))

        x = design.cont
        design.cont = x.numpy()
        if design.condition is not None:

            if design.condition.is_continuous:
                x = tf.concat([x, design.condition.cont], axis=-1)

            if design.condition.is_discrete:
                x = [x, design.condition.disc]

        design.score = self.m(x).numpy()
        return design
