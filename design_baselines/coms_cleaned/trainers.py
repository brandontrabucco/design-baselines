from design_baselines.utils import spearman
from design_baselines.utils import cont_noise
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class ConservativeObjectiveModel(tf.Module):

    def __init__(self,
                 forward_model,
                 forward_model_opt=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001,
                 initial_alpha=1.0,
                 alpha_opt=tf.keras.optimizers.Adam,
                 alpha_lr=0.05,
                 target_conservatism=-0.05,
                 inner_lr=0.01,
                 outer_lr=0.01,
                 inner_gradient_steps=1,
                 outer_gradient_steps=20,
                 beta=0.9,
                 entropy_coefficient=0.9,
                 continuous_noise_std=0.0):
        """A trainer class for building a conservative objective model
        by optimizing a model to make conservative predictions

        Arguments:

        forward_model: tf.keras.Model
            a tf.keras model that accepts designs from an MBO dataset
            as inputs and predicts their score
        forward_model_opt: tf.keras.Optimizer
            an optimizer such as the Adam optimizer that defines
            how to update weights using gradients
        forward_model_lr: float
            the learning rate for the optimizer used to update the
            weights of the forward model during training
        initial_alpha: float
            the initial value of the lagrange multiplier in the
            conservatism objective of the forward model
        alpha_opt: tf.keras.Optimizer
            an optimizer such as the Adam optimizer that defines
            how to update the lagrange multiplier
        alpha_lr: float
            the learning rate for the optimizer used to update the
            lagrange multiplier during training
        target_conservatism: float
            the degree to which the predictions of the model
            underestimate the true score function
        inner_lr: float
            the learning rate for the gradient ascent optimizer
            used to find adversarial solution particles
        outer_lr: float
            the learning rate for the gradient ascent optimizer
            used to find conservative solution particles
        inner_gradient_steps: int
            the number of gradient ascent steps used to find
            adversarial solution particles
        outer_gradient_steps: int
            the number of gradient ascent steps used to find
            conservative solution particles
        beta: float in [0, 1]
            the degree to which the trust region optimizer
            finds conservative solutions.
        continuous_noise_std: float
            the standard deviation of the gaussian noise
            added to dataset x values
        """

        super().__init__()
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_opt(
            learning_rate=forward_model_lr)

        # lagrangian dual descent variables
        log_alpha = np.log(initial_alpha).astype(np.float32)
        self.log_alpha = tf.Variable(log_alpha)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.math.exp)
        self.alpha_opt = alpha_opt(learning_rate=alpha_lr)

        # algorithm hyper parameters
        self.target_conservatism = target_conservatism
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_gradient_steps = inner_gradient_steps
        self.outer_gradient_steps = outer_gradient_steps
        self.beta = beta
        self.entropy_coefficient = entropy_coefficient
        self.continuous_noise_std = continuous_noise_std

    @tf.function(experimental_relax_shapes=True)
    def inner_optimize(self,
                       x,
                       **kwargs):
        """Using gradient descent find adversarial versions of x
        that maximize the score predicted by the model

        Args:

        inner_x: tf.Tensor
            the starting point for the optimizer that will be
            updated using gradient ascent

        Returns:

        optimized_x: tf.Tensor
            a new design found by perform gradient ascent starting
            from the initial x provided as an argument
        """

        # gradient ascent on the predicted score
        def inner_gradient_step(xt):
            with tf.GradientTape() as tape:
                tape.watch(xt)

                # entropy using the gaussian kernel
                entropy = tf.reduce_mean(
                    (xt[tf.newaxis] - xt[:, tf.newaxis]) ** 2)

                # the predicted score according to the forward model
                score = (self.entropy_coefficient * entropy +
                         self.forward_model(xt, **kwargs))

            # update the particles to maximize the predicted score
            return xt + self.inner_lr * tape.gradient(score, xt),

        # use a python for loop (tf.while_loop enters an
        # infinite loop during back prop due to a hidden bug)
        for i in range(self.inner_gradient_steps):

            # perform a single step of gradient ascent
            x = inner_gradient_step(x)[0]

        # return an optimized inner_x
        return x

    @tf.function(experimental_relax_shapes=True)
    def outer_optimize(self,
                       x,
                       beta,
                       steps,
                       **kwargs):
        """Using gradient descent find adversarial versions of x
        that maximize the conservatism of the model

        Args:

        x: tf.Tensor
            the starting point for the optimizer that will be
            updated using gradient ascent
        steps: int
            the number of gradient ascent steps to take in order to
            find x that maximizes conservatism

        Returns:

        optimized_x: tf.Tensor
            a new design found by perform gradient ascent starting
            from the initial x provided as an argument
        """

        # gradient ascent on the conservatism
        def gradient_step(xt):
            with tf.GradientTape() as tape:
                tape.watch(xt)

                # entropy using the gaussian kernel
                entropy = tf.reduce_mean(
                    (xt[tf.newaxis] - xt[:, tf.newaxis]) ** 2)

                # the predicted score according to the forward model
                score = self.forward_model(xt, **kwargs)

                # particles found after optimizing the predicted score
                next_xt = self.inner_optimize(xt, **kwargs)

                # the predicted score of optimized candidate solutions
                next_score = self.forward_model(next_xt, **kwargs)

                # the conservatism of the current set of particles
                loss = (self.entropy_coefficient * entropy +
                        score - beta * next_score)

            # update the particles to maximize the conservatism
            return tf.stop_gradient(
                xt + self.outer_lr * tape.gradient(loss, xt)),

        # use a while loop to perform gradient ascent on the score
        return tf.while_loop(
            lambda xt: True, gradient_step, (x,),
            maximum_iterations=steps)[0]

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   x,
                   y):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of training labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        # corrupt the inputs with noise
        x = cont_noise(x, self.continuous_noise_std)

        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:

            # calculate the prediction error and accuracy of the model
            d_pos = self.forward_model(x, training=True)
            mse = tf.keras.losses.mean_squared_error(y, d_pos)
            statistics[f'train/mse'] = mse

            # evaluate how correct the rank fo the model predictions are
            rank_corr = spearman(y[:, 0], d_pos[:, 0])
            statistics[f'train/rank_corr'] = rank_corr

            # calculate negative samples starting from the dataset
            x_neg = self.outer_optimize(
                x, self.beta, self.outer_gradient_steps, training=False)
            x_neg = tf.stop_gradient(x_neg)

            # calculate the prediction error and accuracy of the model
            d_neg = self.forward_model(x_neg, training=False)
            conservatism = d_pos[:, 0] - d_neg[:, 0]
            statistics[f'train/conservatism'] = conservatism

            # build a lagrangian for dual descent
            alpha_loss = -(self.alpha * self.target_conservatism -
                           self.alpha * conservatism)
            statistics[f'train/alpha'] = self.alpha

            # loss that combines maximum likelihood with a constraint
            model_loss = mse - self.alpha * conservatism
            total_loss = tf.reduce_mean(model_loss)
            alpha_loss = tf.reduce_mean(alpha_loss)

        # calculate gradients using the model
        alpha_grads = tape.gradient(alpha_loss, self.log_alpha)
        model_grads = tape.gradient(
            total_loss, self.forward_model.trainable_variables)

        # take gradient steps on the model
        self.alpha_opt.apply_gradients([[alpha_grads, self.log_alpha]])
        self.forward_model_opt.apply_gradients(zip(
            model_grads, self.forward_model.trainable_variables))

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      x,
                      y):
        """Perform a validation step on an ensemble of models
        without using bootstrapping weights

        Args:

        x: tf.Tensor
            a batch of validation inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of validation labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d_pos = self.forward_model(x, training=False)
        mse = tf.keras.losses.mean_squared_error(y, d_pos)
        statistics[f'validate/mse'] = mse

        # evaluate how correct the rank fo the model predictions are
        rank_corr = spearman(y[:, 0], d_pos[:, 0])
        statistics[f'validate/rank_corr'] = rank_corr

        # calculate negative samples starting from the dataset
        x_neg = self.outer_optimize(
            x, self.beta, self.outer_gradient_steps, training=False)

        # calculate the prediction error and accuracy of the model
        d_neg = self.forward_model(x_neg, training=False)
        conservatism = d_pos.mean()[:, 0] - d_neg[:, 0]
        statistics[f'validate/conservatism'] = conservatism
        return statistics

    def train(self,
              dataset):
        """Perform training using gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.train_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def validate(self,
                 dataset):
        """Perform validation on an ensemble of models without
        using bootstrapping weights

        Args:

        dataset: tf.data.Dataset
            the validation dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def launch(self,
               train_data,
               validate_data,
               logger,
               epochs):
        """Launch training and validation for the model for the specified
        number of epochs, and log statistics

        Args:

        train_data: tf.data.Dataset
            the training dataset already batched and prefetched
        validate_data: tf.data.Dataset
            the validation dataset already batched and prefetched
        logger: Logger
            an instance of the logger used for writing to tensor board
        epochs: int
            the number of epochs through the data sets to take
        """

        for e in range(epochs):
            for name, loss in self.train(train_data).items():
                logger.record(name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(name, loss, e)
