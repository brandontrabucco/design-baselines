from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow as tf


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

    def get_distribution(self,
                         X):
        """Build the mixture distribution implied by the set of oracles
        that are trained in this module

        Args:

        X: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        distributon: tfpd.Distribution
            the mixture of gaussians distribution implied by the oracles
        """

        loc = []
        scale = []
        for oracle in self.oracles:

            prediction = oracle(X, training=False)
            mu, log_std = tf.split(prediction, 2, axis=-1)

            loc.append(mu)
            scale.append(tf.math.softplus(log_std))

        loc = tf.stack(loc, axis=2)
        scale = tf.stack(scale, axis=2)

        num_components = len(self.oracles)
        weights = tf.fill([num_components], 1 / num_components)

        mixture = tfpd.Categorical(probs=weights)
        component = tfpd.Normal(loc=loc, scale=scale)

        return tfpd.MixtureSameFamily(
            mixture_distribution=mixture,
            components_distribution=component)

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

                d = tfpd.Normal(loc=mu, scale=tf.math.softplus(log_std))
                nll = -d.log_prob(y)[:, 0]

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

            d = tfpd.Normal(loc=mu, scale=tf.math.softplus(log_std))
            nll = -d.log_prob(y)[:, 0]

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


class WeightedVAE(tf.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 vae_optim=tf.keras.optimizers.Adam,
                 vae_lr=0.001,
                 vae_beta=0.01):
        """Build a trainer for an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        encoder: tf.keras.Model
            the encoder neural network that outputs parameters for a gaussian
        decoder: tf.keras.Model
            the decoder neural network that outputs parameters for a gaussian
        vae_optim: __class__
            the optimizer class to use for optimizing the oracle model
        vae_lr: float
            the learning rate for the oracle model optimizer
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optim = vae_optim(learning_rate=vae_lr)
        self.vae_beta = vae_beta

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   X,
                   y,
                   w):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        X: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of training labels shaped like [batch_size, 1]
        w: tf.Tensor
            importance sampling weights shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()
        var_list = []
        var_list.extend(self.encoder.trainable_variables)
        var_list.extend(self.decoder.trainable_variables)

        with tf.GradientTape() as tape:

            mu, log_std = tf.split(
                self.encoder(X, training=True), 2, axis=-1)

            dz = tfpd.MultivariateNormalDiag(
                loc=mu, scale_diag=tf.math.softplus(log_std))

            z = dz.sample()

            mu, log_std = tf.split(
                self.decoder(z, training=True), 2, axis=-1)

            dx = tfpd.MultivariateNormalDiag(
                loc=mu, scale_diag=tf.math.softplus(log_std))

            nll = -dx.log_prob(X)[:, tf.newaxis]

            prior = tfpd.MultivariateNormalDiag(
                loc=tf.zeros_like(mu), scale_diag=tf.ones_like(log_std))

            kl = dx.kl_divergence(prior)[:, tf.newaxis]

            total_loss = tf.reduce_mean(w * (nll + self.vae_beta * kl))

        grads = tape.gradient(total_loss, var_list)
        self.optim.apply_gradients(zip(grads, var_list))

        statistics[f'vae/train/nll'] = nll
        statistics[f'vae/train/kl'] = kl

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

        mu, log_std = tf.split(
            self.encoder(X, training=True), 2, axis=-1)

        dz = tfpd.MultivariateNormalDiag(
            loc=mu, scale_diag=tf.math.softplus(log_std))

        z = dz.sample()

        mu, log_std = tf.split(
            self.decoder(z, training=True), 2, axis=-1)

        dx = tfpd.MultivariateNormalDiag(
            loc=mu, scale_diag=tf.math.softplus(log_std))

        nll = -dx.log_prob(X)[:, tf.newaxis]

        prior = tfpd.MultivariateNormalDiag(
            loc=tf.zeros_like(mu), scale_diag=tf.ones_like(log_std))

        kl = dx.kl_divergence(prior)[:, tf.newaxis]

        statistics[f'vae/validate/nll'] = nll
        statistics[f'vae/validate/kl'] = kl

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

        for X, y, w in dataset:
            for name, tensor in self.train_step(X, y, w).items():
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
