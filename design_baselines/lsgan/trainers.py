from design_baselines.utils import spearman
from design_baselines.utils import add_discrete_noise
from design_baselines.utils import add_continuous_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow as tf


class LSGAN(tf.Module):

    def __init__(self,
                 generator,
                 discriminator,
                 generator_lr=0.001,
                 generator_beta_1=0.5,
                 generator_beta_2=0.999,
                 discriminator_lr=0.001,
                 discriminator_beta_1=0.5,
                 discriminator_beta_2=0.999,
                 is_discrete=False,
                 noise_std=0.0,
                 keep=0.0,
                 temp=0.0):
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
        vae_beta: float
            the variational beta for the oracle model optimizer
        """

        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        # create optimizers for the generator and discriminator
        self.generator_optim = tf.keras.optimizers.Adam(
            learning_rate=generator_lr,
            beta_1=generator_beta_1,
            beta_2=generator_beta_2)
        self.discriminator_optim = tf.keras.optimizers.Adam(
            learning_rate=discriminator_lr,
            beta_1=discriminator_beta_1,
            beta_2=discriminator_beta_2)

        # create machinery for adding noise to the inputs
        self.is_discrete = is_discrete
        self.noise_std = noise_std
        self.keep = keep
        self.temp = temp

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   x):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: tf.Tensor
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

        # corrupt the inputs with noise
        x_real = x \
            if self.is_discrete else add_continuous_noise(x, self.noise_std)

        with tf.GradientTape() as tape:

            # sample designs from the generator
            x_fake = self.generator.sample(x.shape[0], temp=0.001, training=True)
            d_real = self.discriminator.loss(x_real, real=True, training=True)
            d_fake = self.discriminator.loss(x_fake, real=False, training=True)
            penalty = self.discriminator.penalty(x_real, training=False)

            # calculate discriminative accuracy
            acc_real = tf.cast(d_real < 0.25, tf.float32)
            acc_fake = tf.cast(d_fake < 0.25, tf.float32)

            # build the total loss
            total_loss = tf.reduce_mean(
                d_real +
                d_fake +
                10.0 * penalty)

        var_list = self.discriminator.trainable_variables
        grads = tape.gradient(total_loss, var_list)
        self.discriminator_optim.apply_gradients(zip(grads, var_list))

        statistics[f'discriminator/train/d_real'] = d_real
        statistics[f'discriminator/train/d_fake'] = d_fake
        statistics[f'discriminator/train/penalty'] = penalty
        statistics[f'discriminator/train/acc_real'] = acc_real
        statistics[f'discriminator/train/acc_fake'] = acc_fake
        statistics[f'generator/train/x_fake'] = x_fake
        statistics[f'generator/train/x_real'] = x_real

        with tf.GradientTape() as tape:

            # sample designs from the generator
            x_fake = self.generator.sample(x.shape[0], temp=self.temp, training=False)
            d_fake = self.discriminator.loss(x_fake, real=True, training=False)

            # build the total loss
            total_loss = tf.reduce_mean(d_fake)

        var_list = self.generator.trainable_variables
        grads = tape.gradient(total_loss, var_list)
        self.generator_optim.apply_gradients(zip(grads, var_list))

        statistics[f'generator/train/x_fake2'] = x_fake

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      x):
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

        # corrupt the inputs with noise
        x_real = x \
            if self.is_discrete else add_continuous_noise(x, self.noise_std)

        # sample designs from the generator
        x_fake = self.generator.sample(x.shape[0], temp=0.001, training=False)
        d_real = self.discriminator.loss(x_real, real=True, training=False)
        d_fake = self.discriminator.loss(x_fake, real=False, training=False)
        penalty = self.discriminator.penalty(x_real, training=False)

        # calculate discriminative accuracy
        acc_real = tf.cast(d_real < 0.25, tf.float32)
        acc_fake = tf.cast(d_fake < 0.25, tf.float32)

        statistics[f'discriminator/validate/d_real'] = d_real
        statistics[f'discriminator/validate/d_fake'] = d_fake
        statistics[f'discriminator/validate/penalty'] = penalty
        statistics[f'discriminator/validate/acc_real'] = acc_real
        statistics[f'discriminator/validate/acc_fake'] = acc_fake
        statistics[f'generator/validate/x_fake'] = x_fake
        statistics[f'generator/validate/x_real'] = x_real

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
        for X, y in dataset:
            for name, tensor in self.train_step(X).items():
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
            for name, tensor in self.validate_step(X).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def launch(self,
               train_data,
               validate_data,
               logger,
               epochs,
               start_epoch=0,
               header=""):
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

        for e in range(start_epoch, start_epoch + epochs):
            for name, loss in self.train(train_data).items():
                logger.record(header + name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(header + name, loss, e)

    def get_saveables(self):
        """Collects and returns stateful objects that are serializeable
        using the tensorflow checkpoint format

        Returns:

        saveables: dict
            a dict containing stateful objects compatible with checkpoints
        """

        saveables = dict()
        saveables['generator'] = self.generator
        saveables['discriminator'] = self.discriminator
        saveables['generator_optim'] = self.generator_optim
        saveables['discriminator_optim'] = self.discriminator_optim
        return saveables
