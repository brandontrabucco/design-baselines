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
                 training_dp: DesignProblem,
                 validation_dp: DesignProblem,
                 num_layers=2,
                 hidden_size=512,
                 batch_size=32,
                 training_iterations=10000,
                 init_lr=0.0001,
                 num_sgd_steps=10,
                 discrete_size=1000,
                 init_from_dataset=True,
                 conservative_noise_std=0.1,
                 conservative_lambda=10.0,
                 conservative_weight=0.0,
                 add_noise=False,
                 noise_std=0.01,
                 label_interpolation=True):
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

        Algorithm.__init__(self,
                           training_dp,
                           validation_dp,
                           num_layers=num_layers,
                           hidden_size=hidden_size,
                           batch_size=batch_size,
                           training_iterations=training_iterations,
                           init_lr=init_lr,
                           num_sgd_steps=num_sgd_steps,
                           discrete_size=discrete_size,
                           init_from_dataset=init_from_dataset,
                           conservative_noise_std=conservative_noise_std,
                           conservative_lambda=conservative_lambda,
                           conservative_weight=conservative_weight,
                           add_noise=add_noise,
                           noise_std=noise_std,
                           label_interpolation=label_interpolation)

        assert self.training_dp.is_continuous
        self.training_losses = []
        self.training_rms = []
        self.training_std = []
        self.validation_losses = []
        self.validation_rms = []
        self.validation_std = []

        layers = []
        for i in range(self.num_layers):
            layers.extend([tf.keras.layers.Dense(self.hidden_size),
                           tf.keras.layers.Activation('relu')])
        layers.append(tf.keras.layers.Dense(1))
        self.m = tf.keras.Sequential(layers)

        optim = tf.keras.optimizers.Adam(learning_rate=self.init_lr)
        for i in range(training_iterations):

            design1 = self.training_dp.sample(n=batch_size)
            design1.score = np.nan_to_num(design1.score)

            design2 = self.training_dp.sample(n=batch_size)
            design2.score = np.nan_to_num(design2.score)

            if self.add_noise:
                ub = self.training_dp.design_space.upper
                lb = self.training_dp.design_space.lower
                scale = (ub - lb) / 2

                design1.cont += tf.random.normal(
                    design1.cont.shape) * scale * self.noise_std
                design2.cont += tf.random.normal(
                    design1.cont.shape) * scale * self.noise_std

            with tf.GradientTape() as tape:

                x1 = design1.cont
                if design1.condition is not None:

                    if design1.condition.is_continuous:
                        x1 = tf.concat([x1, design1.condition.cont], axis=-1)

                    if design1.condition.is_discrete:
                        z1 = tf.one_hot(design1.condition.disc, self.discrete_size)
                        z1 = tf.reshape(z1, [tf.shape(z1)[0],
                                             tf.shape(z1)[1] * self.discrete_size])
                        x1 = tf.concat([x1, z1], axis=-1)

                x2 = design2.cont
                if design2.condition is not None:

                    if design2.condition.is_continuous:
                        x2 = tf.concat([x2, design2.condition.cont], axis=-1)

                    if design2.condition.is_discrete:
                        z2 = tf.one_hot(design2.condition.disc, self.discrete_size)
                        z2 = tf.reshape(z2, [tf.shape(z2)[0],
                                             tf.shape(z2)[1] * self.discrete_size])
                        x2 = tf.concat([x2, z2], axis=-1)

                a = tf.random.uniform([batch_size, 1])
                if self.label_interpolation:
                    x = x1 * a + x2 * (1 - a)
                    label = design1.score * a + design2.score * (1 - a)
                else:
                    x = x1
                    label = design1.score

                if hasattr(design1, 'reweighting_weights'):
                    if self.label_interpolation:
                        reweighting_weights = design1.reweighting_weights * a + \
                                              design2.reweighting_weights * (1 - a)
                    else:
                        reweighting_weights = design1.reweighting_weights
                else:
                    reweighting_weights = 1

                x_ns = x + tf.random.normal(x.shape) * self.conservative_noise_std
                d_ns = tf.linalg.norm(x - x_ns, axis=-1, keepdims=True)
                loss_ns = tf.reduce_mean(
                    reweighting_weights *
                    tf.keras.losses.logcosh(
                        design1.score - d_ns * conservative_lambda,
                        self.m(x)))

                loss = loss_ns * self.conservative_weight + tf.reduce_mean(
                    reweighting_weights *
                    tf.keras.losses.logcosh(label, self.m(x)))
                rms = tf.reduce_mean((
                    label -
                    self.m(x)) ** 2) ** 0.5
                std = tf.reduce_mean((
                    label -
                    tf.reduce_mean(label)) ** 2) ** 0.5

                self.training_losses.append(loss.numpy())
                self.training_rms.append(rms.numpy())
                self.training_std.append(std.numpy())

                print(f"training {i} "
                      f"loss {loss.numpy()} "
                      f"rms {rms.numpy()} "
                      f"std {std.numpy()}")

            grads = tape.gradient(loss, self.m.trainable_variables)
            optim.apply_gradients(zip(grads, self.m.trainable_variables))
            optim.lr.assign(self.init_lr * (1 - (i + 1) / training_iterations))

            design = self.validation_dp.sample(n=batch_size)
            design.score = np.nan_to_num(design.score)

            x = design.cont
            if design.condition is not None:

                if design.condition.is_continuous:
                    x = tf.concat([x, design.condition.cont], axis=-1)

                if design.condition.is_discrete:
                    z = tf.one_hot(design.condition.disc, self.discrete_size)
                    z = tf.reshape(z, [tf.shape(z)[0],
                                       tf.shape(z)[1] * self.discrete_size])
                    x = tf.concat([x, z], axis=-1)

            if hasattr(design, 'reweighting_weights'):
                reweighting_weights = design.reweighting_weights
            else:
                reweighting_weights = 1

            loss = tf.reduce_mean(
                reweighting_weights *
                tf.keras.losses.logcosh(design.score, self.m(x)))
            rms = tf.reduce_mean((
                design.score -
                self.m(x)) ** 2) ** 0.5
            std = tf.reduce_mean((
                design.score -
                tf.reduce_mean(design.score)) ** 2) ** 0.5

            self.validation_losses.append(loss.numpy())
            self.validation_rms.append(rms.numpy())
            self.validation_std.append(std.numpy())

            print(f"validate {i} "
                  f"loss {loss.numpy()} "
                  f"rms {rms.numpy()} "
                  f"std {std.numpy()}")

    def solve(self, condition: Container = None) -> Design:
        """
        Solve the design problem by conditioning on inputs and finding a
        design that maximizes the design problem score

        Arguments:

        inputs: Any
            An optimal input used to parameterize the design problem
            For example, conditioning on an attribute like color
        """

        if self.init_from_dataset:
            design = self.training_dp.sample(n=1)
        else:
            design = self.training_dp.design_space.sample(n=1)
        design.condition = condition
        design.cont = tf.Variable(tf.convert_to_tensor(design.cont))

        optim = tf.keras.optimizers.Adam(learning_rate=0.001)
        for i in range(self.num_sgd_steps):

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
                self.training_dp.design_space.lower,
                self.training_dp.design_space.upper))

        x = design.cont
        design.cont = x.numpy()
        if design.condition is not None:

            if design.condition.is_continuous:
                x = tf.concat([x, design.condition.cont], axis=-1)

            if design.condition.is_discrete:
                x = [x, design.condition.disc]

        design.score = self.m(x).numpy()
        return design
