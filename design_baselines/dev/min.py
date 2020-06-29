from design_baselines.core import Algorithm
from design_bench.core import DesignProblem, Design, Container
import tensorflow as tf
import numpy as np


class MIN(Algorithm):
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
                           num_layers=num_layers,
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

        # build a generator networks that samples designs
        # conditioned on a score and noise
        layers = []
        for i in range(self.num_layers):
            layers.extend([tf.keras.layers.Dense(self.hidden_size),
                           tf.keras.layers.Activation('relu')])
        layers.append(tf.keras.layers.Dense(
            self.design_problem.design_space.lower.size))
        layers.append(tf.keras.layers.Activation('tanh'))
        self.generator = tf.keras.Sequential(layers)

        # build a discriminator network that predicts the probability
        # that a design and score pair are real
        layers = []
        for i in range(self.num_layers):
            layers.extend([tf.keras.layers.Dense(self.hidden_size),
                           tf.keras.layers.Activation('relu')])
        layers.append(tf.keras.layers.Dense(1))
        layers.append(tf.keras.layers.Activation('sigmoid'))
        self.discriminator = tf.keras.Sequential(layers)

        g_optim = tf.keras.optimizers.Adam(learning_rate=0.000001)
        d_optim = tf.keras.optimizers.Adam(learning_rate=0.000001)

        for i in range(training_iterations):

            design = self.design_problem.sample(n=batch_size)
            design.score = np.nan_to_num(design.score)

            # train the generator

            if i % 5 == 0:

                with tf.GradientTape() as g_tape:

                    x = design.score
                    if design.condition is not None:

                        if design.condition.is_continuous:
                            x = tf.concat([x, design.condition.cont], axis=-1)

                        if design.condition.is_discrete:
                            z = tf.one_hot(design.condition.disc, self.discrete_size)
                            z = tf.reshape(z, [tf.shape(z)[0],
                                               tf.shape(z)[1] * self.discrete_size])
                            x = tf.concat([x, z], axis=-1)

                    noise = tf.random.normal([batch_size, self.latent_dim])
                    sample = self.generator(tf.concat([x, noise], axis=-1),
                                            training=True) * self.scale + self.shift
                    fake_validity = self.discriminator(
                        tf.concat([sample, x], axis=-1), training=True)

                    loss = tf.reduce_mean(tf.math.log(1.0 - fake_validity))

                grads = g_tape.gradient(
                    loss, self.generator.trainable_variables)
                g_optim.apply_gradients(
                    zip(grads, self.generator.trainable_variables))

            # train the discriminator

            with tf.GradientTape() as d_tape:

                x = design.score
                if design.condition is not None:

                    if design.condition.is_continuous:
                        x = tf.concat([x, design.condition.cont], axis=-1)

                    if design.condition.is_discrete:
                        z = tf.one_hot(design.condition.disc, self.discrete_size)
                        z = tf.reshape(z, [tf.shape(z)[0],
                                           tf.shape(z)[1] * self.discrete_size])
                        x = tf.concat([x, z], axis=-1)

                noise = tf.random.normal([batch_size, self.latent_dim])
                sample = self.generator(tf.concat([x, noise], axis=-1),
                                        training=True) * self.scale + self.shift
                fake_validity = self.discriminator(
                    tf.concat([sample, x], axis=-1), training=True)
                real_validity = self.discriminator(
                    tf.concat([design.cont, x], axis=-1), training=True)

                loss = -tf.reduce_mean(tf.math.log(real_validity) +
                                       tf.math.log(1.0 - fake_validity))
                fake_acc = tf.reduce_mean(tf.cast(fake_validity < 0.5, tf.float32))
                real_acc = tf.reduce_mean(tf.cast(real_validity > 0.5, tf.float32))
                print(f"d iteration {i} fake {fake_acc.numpy()} real {real_acc.numpy()}")

            grads = d_tape.gradient(
                loss, self.discriminator.trainable_variables)
            d_optim.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

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
