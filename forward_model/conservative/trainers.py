from forward_model.utils import spearman
from collections import defaultdict
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Conservative(tf.Module):

    def __init__(self,
                 forward_model,
                 forward_model_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001,
                 target_conservative_gap=tf.constant(10.0),
                 initial_alpha=0.0001,
                 alpha_optim=tf.keras.optimizers.Adam,
                 alpha_lr=0.001,
                 perturbation_lr=0.001,
                 perturbation_steps=100):
        """Build a trainer for a conservative forward model with negatives
        sampled from a perturbation distribution

        Args:

        forward_model: tf.keras.Model
            a keras model that accepts vectorized inputs and returns scores
        target_conservative_gap: float
            the target gap between f(x) and f(\tilde x) for training
        initial_alpha: float
            the initial value for the alpha lagrange multiplier
        forward_model_optim: __class__
            the optimizer class to use for optimizing the forward model
        forward_model_lr: float
            the learning rate for the forward model optimizer
        alpha_optim: __class__
            the optimizer class to use for optimizing the lagrange multiplier
        alpha_lr: float
            the learning rate for the lagrange multiplier optimizer
        perturbation_lr: float
            the learning rate used when finding adversarial examples
        perturbation_steps: int
            the number of gradient steps taken to find adversarial examples
        """

        super().__init__()
        self.forward_model = forward_model
        self.target_conservative_gap = target_conservative_gap
        self.optim = forward_model_optim(learning_rate=forward_model_lr)

        # create training machinery for alpha
        self.log_alpha = tf.Variable(np.log(initial_alpha).astype(np.float32))
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        self.alpha_optim = alpha_optim(learning_rate=alpha_lr)

        # create machinery for sampling adversarial examples
        self.perturbation_lr = perturbation_lr
        self.perturbation_steps = perturbation_steps

    def optimize(self,
                 x,
                 **kwargs):
        """Using gradient descent find adversarial versions of x that maximize
        the score predicted by the forward model

        Args:

        x: tf.Tensor
            the original value of the tensor being optimized

        Returns:

        optimized_x: tf.Tensor
            the perturbed value of x that maximizes the score function
        """

        for _ in tf.range(self.perturbation_steps):
            with tf.GradientTape() as tape:
                tape.watch(x)
                score = self.forward_model(x, **kwargs)
            x = x + self.perturbation_lr * tape.gradient(score, x)
        return x

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   X,
                   y):
        """Perform a training step of gradient descent on a forward model
        with an adversarial perturbation distribution

        Args:

        X: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of training labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        with tf.GradientTape(persistent=True) as tape:

            # calculate the prediction error and accuracy of the model
            prediction = self.forward_model(X, training=True)
            mse = tf.keras.losses.mse(y, prediction)
            rank_correlation = spearman(y[:, 0], prediction[:, 0])

            # calculate the conservative gap
            perturb = tf.stop_gradient(self.optimize(X))
            conservative = (self.forward_model(perturb, training=True)[:, 0] -
                            self.forward_model(X, training=True)[:, 0])
            gap = (self.alpha * self.target_conservative_gap -
                   self.alpha * conservative)

            # build the total and lagrangian losses
            total_loss = tf.reduce_mean(mse + self.alpha * conservative)
            alpha_loss = tf.reduce_mean(gap)

        grads = tape.gradient(
            total_loss, self.forward_model.trainable_variables)
        self.optim.apply_gradients(
            zip(grads, self.forward_model.trainable_variables))
        grad = tape.gradient(alpha_loss, self.log_alpha)
        self.alpha_optim.apply_gradients([[grad, self.log_alpha]])

        return {'train/mse': mse,
                'train/alpha_loss': gap,
                'train/alpha': tf.convert_to_tensor(self.alpha),
                'train/rank_correlation': rank_correlation,
                'train/conservative': conservative}

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      X,
                      y):
        """Perform a validation step on a forward model with an
        adversarial perturbation distribution

        Args:

        X: tf.Tensor
            a batch of validation inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of validation labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        # calculate the prediction error and accuracy of the model
        prediction = self.forward_model(X, training=False)
        mse = tf.keras.losses.mse(y, prediction)
        rank_correlation = spearman(y[:, 0], prediction[:, 0])

        # calculate the conservative gap
        perturb = tf.stop_gradient(self.optimize(X))
        conservative = (self.forward_model(perturb, training=False)[:, 0] -
                        self.forward_model(X, training=False)[:, 0])

        return {'validate/mse': mse,
                'validate/rank_correlation': rank_correlation,
                'validate/conservative': conservative}

    def train(self,
              dataset):
        """Train a conservative forward model and collect negative
        samples using a perturbation distribution

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)

        for X, y in dataset:
            for name, tensor in self.train_step(X, y).items():
                statistics[name].append(tensor)

        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)

        return statistics

    def validate(self,
                 dataset):
        """Validate a conservative forward model using a validation dataset
        and return the average validation loss

        Args:

        dataset: tf.data.Dataset
            the validation dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)

        for X, y in dataset:
            for name, tensor in self.validate_step(X, y).items():
                statistics[name].append(tensor)

        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)

        return statistics

    def launch(self,
               train_data,
               validate_data,
               logger,
               epochs,
               start_epoch=0):
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

        # train and validate the neural network models
        for e in range(start_epoch, start_epoch + epochs):
            for name, loss in self.train(train_data).items():
                logger.record(name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(name, loss, e)

    def get_saveables(self):
        """Collects and returns stateful objects that are serializeable
        using the tensorflow checkpoint format

        Returns:

        saveables: dict
            a dict containing stateful objects compatible with checkpoints
        """

        return {"forward_model": self.forward_model,
                "optim": self.optim,
                "log_alpha": self.log_alpha,
                "alpha_optim": self.alpha_optim}
