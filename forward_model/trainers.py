from forward_model.utils import spearman
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Conservative(tf.Module):

    def __init__(self,
                 forward_model,
                 perturbation,
                 forward_model_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001,
                 target_conservative_gap=tf.constant(10.0),
                 initial_alpha=0.0001,
                 alpha_optim=tf.keras.optimizers.Adam,
                 alpha_lr=0.001):
        """Build a trainer for a conservative forward model with negatives
        sampled from a perturbation distribution

        Args:

        forward_model: tf.keras.Model
            a keras model that accepts vectorized inputs and returns scores
        perturbation: Perturbation
            a distribution that returns new samples conditioned on an X
        target_conservative_gap: float
            the target gap between f(x) and f(\tilde x) for conservative training
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
        """

        super().__init__()
        self.forward_model = forward_model
        self.perturbation = perturbation
        self.target_conservative_gap = target_conservative_gap
        self.optim = forward_model_optim(learning_rate=forward_model_lr)

        # create training machinery for alpha
        self.log_alpha = tf.Variable(np.log(initial_alpha).astype(np.float32))
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        self.alpha_optim = alpha_optim(learning_rate=alpha_lr)

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
            prediction = self.forward_model(X, training=True)
            mse = tf.keras.losses.mse(y, prediction)
            rank_correlation = spearman(y[:, 0], prediction[:, 0])

            perturb = tf.stop_gradient(self.perturbation(X))
            conservative = (self.forward_model(perturb, training=True)[:, 0] -
                            self.forward_model(X, training=True)[:, 0])
            gap = (self.alpha * self.target_conservative_gap -
                   self.alpha * conservative)

            total_loss = tf.reduce_mean(mse + self.alpha * conservative)
            alpha_loss = tf.reduce_mean(gap)

        grads = tape.gradient(
            total_loss, self.forward_model.trainable_variables)
        self.optim.apply_gradients(
            zip(grads, self.forward_model.trainable_variables))

        grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optim.apply_gradients(zip(grads, [self.log_alpha]))

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

        prediction = self.forward_model(X, training=False)
        mse = tf.keras.losses.mse(y, prediction)
        rank_correlation = spearman(y[:, 0], prediction[:, 0])

        perturb = tf.stop_gradient(self.perturbation(X))
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


class BootstrapEnsemble(tf.Module):

    def __init__(self,
                 oracles,
                 oracle_optim=tf.keras.optimizers.Adam,
                 oracle_lr=0.001):
        """Build a trainer for an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        oracles: List[tf.keras.Model]
            a list of keras model that predict distributions over scores
        oracle_optim: __class__
            the optimizer class to use for optimizing the oracle model
        oracle__lr: float
            the learning rate for the oracle model optimizer
        """

        super().__init__()
        self.oracles = oracles
        self.optims = [oracle_optim(
            learning_rate=oracle_lr) for _ in oracles]

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   X,
                   y,
                   b):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        X: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of training labels shaped like [batch_size, 1]
        b: tf.Tensor
            bootstrap indicators shaped like [batch_size, num_oracles]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        for i, (oracle, optim) in enumerate(
                zip(self.oracles, self.optims)):

            with tf.GradientTape() as tape:
                prediction = oracle(X, training=True)
                mu, log_std = tf.split(prediction, 2, axis=-1)

                d = tfpd.MultivariateNormalDiag(
                    loc=mu, scale_diag=tf.math.softplus(log_std))

                nll = -d.log_prob(y)
                total_loss = tf.reduce_sum(
                    b[:, i] * nll) / tf.reduce_sum(b[:, i])

            grads = tape.gradient(
                total_loss, oracle.trainable_variables)
            optim.apply_gradients(
                zip(grads, oracle.trainable_variables))

            statistics[f'oracle_{i}/train/nll'] = nll

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      X,
                      y):
        """Perform a validation step on an ensemble of models
        without using bootstrapping weights

        Args:

        X: tf.Tensor
            a batch of validation inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of validation labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        for i, (oracle, optim) in enumerate(
                zip(self.oracles, self.optims)):

            prediction = oracle(X, training=False)
            mu, log_std = tf.split(prediction, 2, axis=-1)

            d = tfpd.MultivariateNormalDiag(
                loc=mu, scale_diag=tf.math.softplus(log_std))

            nll = -d.log_prob(y)
            statistics[f'oracle_{i}/validate/nll'] = nll

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

        for X, y, b in dataset:
            for name, tensor in self.train_step(X, y, b).items():
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

        for X, y in dataset:
            for name, tensor in self.validate_step(X, y).items():
                statistics[name].append(tensor)

        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)

        return statistics


class ModelInversion(tf.Module):

    def __init__(self,
                 generator,
                 discriminator,
                 latent_size=32,
                 optim=tf.keras.optimizers.Adam,
                 **optimizer_kwargs):
        """Build a trainer for a conservative forward model with negatives
        sampled from a perturbation distribution

        Args:

        generator: tf.keras.Model
            a model that accepts scores and returns designs x
        discriminator: tf.keras.Model
            a model that predicts which design and score pairs are real
        optim: __class__
            the optimizer class to use such as tf.keras.optimizers.SGD
        **optimizer_kwargs: dict
            additional keyword arguments passed to optim
        """

        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_size = latent_size
        self.generator_optim = optim(**optimizer_kwargs)
        self.discriminator_optim = optim(**optimizer_kwargs)
        self.optimizer_kwargs = optimizer_kwargs

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

        d_loss = tf.zeros([0])
        real_accuracy = tf.zeros([0])
        g_loss = tf.zeros([0])
        fake_accuracy = tf.zeros([0])
        for X, y, w in dataset:

            with tf.GradientTape() as tape:
                real_p = self.discriminator(tf.concat([X, y], 1), training=True)
                real_loss = w * tf.keras.losses.mse(tf.ones_like(y), real_p)
                X_fake = self.generator(tf.concat([
                    tf.random.normal([X.shape[0], self.latent_size]), y], 1), training=True)
                fake_p = self.discriminator(tf.concat([X_fake, y], 1))
                fake_loss = w * tf.keras.losses.mse(tf.zeros_like(y), fake_p)
                loss = real_loss + fake_loss
                d_loss = tf.concat([d_loss, loss], 0)
                real_accuracy = tf.concat([
                    real_accuracy, tf.cast(real_p[:, 0] > 0.5, tf.float32)], 0)
                fake_accuracy = tf.concat([
                    fake_accuracy, tf.cast(fake_p[:, 0] < 0.5, tf.float32)], 0)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(
                loss, self.discriminator.trainable_variables)
            self.discriminator_optim.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                X_fake = self.generator(tf.concat([
                    tf.random.normal([X.shape[0], self.latent_size]), y], 1))
                fake_loss = w * tf.keras.losses.mse(
                    tf.ones_like(y),
                    self.discriminator(tf.concat([X_fake, y], 1)))
                loss = fake_loss
                g_loss = tf.concat([g_loss, loss], 0)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(
                loss, self.generator.trainable_variables)
            self.generator_optim.apply_gradients(
                zip(grads, self.generator.trainable_variables))

        return {"discriminator_train": d_loss,
                "generator_train": g_loss,
                "real_accuracy": real_accuracy,
                "fake_accuracy": fake_accuracy}

    def validate(self,
                 dataset):
        """Validate a conservative forward model using a validation dataset
        and return the average validation loss

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        d_loss = tf.zeros([0])
        real_accuracy = tf.zeros([0])
        g_loss = tf.zeros([0])
        fake_accuracy = tf.zeros([0])
        for X, y, w in dataset:

            real_p = self.discriminator(tf.concat([X, y], 1))
            real_loss = w * tf.keras.losses.mse(tf.ones_like(y), real_p)
            X_fake = self.generator(tf.concat([
                tf.random.normal([X.shape[0], self.latent_size]), y], 1))
            fake_p = self.discriminator(tf.concat([X_fake, y], 1))
            fake_loss = w * tf.keras.losses.mse(tf.zeros_like(y), fake_p)
            loss = real_loss + fake_loss
            d_loss = tf.concat([d_loss, loss], 0)
            real_accuracy = tf.concat([
                real_accuracy, tf.cast(real_p[:, 0] > 0.5, tf.float32)], 0)
            fake_accuracy = tf.concat([
                fake_accuracy, tf.cast(fake_p[:, 0] < 0.5, tf.float32)], 0)

            X_fake = self.generator(tf.concat([
                tf.random.normal([X.shape[0], self.latent_size]), y], 1))
            fake_loss = w * tf.keras.losses.mse(
                tf.ones_like(y),
                self.discriminator(tf.concat([X_fake, y], 1)))
            loss = fake_loss
            g_loss = tf.concat([g_loss, loss], 0)

        return {"discriminator_train": d_loss,
                "generator_train": g_loss,
                "real_accuracy": real_accuracy,
                "fake_accuracy": fake_accuracy}
