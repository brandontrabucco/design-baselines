from design_baselines.utils import spearman
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class ConservativeObjectiveModel(tf.Module):

    def __init__(self, forward_model,
                 forward_model_opt=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001, alpha=1.0,
                 alpha_opt=tf.keras.optimizers.Adam,
                 alpha_lr=0.01, overestimation_limit=0.5,
                 particle_lr=0.05, particle_gradient_steps=50,
                 entropy_coefficient=0.9, noise_std=0.0):
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
        alpha: float
            the initial value of the lagrange multiplier in the
            conservatism objective of the forward model
        alpha_opt: tf.keras.Optimizer
            an optimizer such as the Adam optimizer that defines
            how to update the lagrange multiplier
        alpha_lr: float
            the learning rate for the optimizer used to update the
            lagrange multiplier during training
        overestimation_limit: float
            the degree to which the predictions of the model
            overestimate the true score function
        particle_lr: float
            the learning rate for the gradient ascent optimizer
            used to find adversarial solution particles
        particle_gradient_steps: int
            the number of gradient ascent steps used to find
            adversarial solution particles
        entropy_coefficient: float
            the entropy bonus added to the loss function when updating
            solution particles with gradient ascent
        noise_std: float
            the standard deviation of the gaussian noise added to
            designs when training the forward model
        """

        super().__init__()
        self.forward_model = forward_model
        self.forward_model_opt = \
            forward_model_opt(learning_rate=forward_model_lr)

        # lagrangian dual descent variables
        self.log_alpha = tf.Variable(np.log(alpha).astype(np.float32))
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.math.exp)
        self.alpha_opt = alpha_opt(learning_rate=alpha_lr)

        # algorithm hyper parameters
        self.overestimation_limit = overestimation_limit
        self.particle_lr = particle_lr
        self.particle_gradient_steps = particle_gradient_steps
        self.entropy_coefficient = entropy_coefficient
        self.noise_std = noise_std

    @tf.function(experimental_relax_shapes=True)
    def optimize(self, x, steps, **kwargs):
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

                # shuffle the designs for calculating entropy
                shuffled_xt = tf.gather(
                    xt, tf.random.shuffle(tf.range(tf.shape(xt)[0])))

                # entropy using the gaussian kernel
                entropy = tf.reduce_mean((xt - shuffled_xt) ** 2)

                # the predicted score according to the forward model
                score = self.forward_model(xt, **kwargs)

                # the conservatism of the current set of particles
                loss = self.entropy_coefficient * entropy + score

            # update the particles to maximize the conservatism
            return tf.stop_gradient(
                xt + self.particle_lr * tape.gradient(loss, xt)),

        # use a while loop to perform gradient ascent on the score
        return tf.while_loop(
            lambda xt: True, gradient_step, (x,),
            maximum_iterations=steps)[0]

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
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
        x = x + self.noise_std * tf.random.normal(tf.shape(x))

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
            x_neg = self.optimize(
                x, self.particle_gradient_steps, training=False)

            # calculate the prediction error and accuracy of the model
            d_neg = self.forward_model(x_neg, training=False)
            overestimation = d_neg[:, 0] - d_pos[:, 0]
            statistics[f'train/overestimation'] = overestimation

            # build a lagrangian for dual descent
            alpha_loss = (self.alpha * self.overestimation_limit -
                          self.alpha * overestimation)
            statistics[f'train/alpha'] = self.alpha

            # loss that combines maximum likelihood with a constraint
            model_loss = mse + self.alpha * overestimation
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
    def validate_step(self, x, y):
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
        x_neg = self.optimize(
            x, self.particle_gradient_steps, training=False)

        # calculate the prediction error and accuracy of the model
        d_neg = self.forward_model(x_neg, training=False)
        overestimation = d_neg[:, 0] - d_pos[:, 0]
        statistics[f'validate/overestimation'] = overestimation
        return statistics

    def train(self, dataset):
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

    def validate(self, dataset):
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

    def launch(self, train_data, validate_data, logger, epochs):
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


class VAETrainer(tf.Module):

    def __init__(self,
                 vae,
                 optim=tf.keras.optimizers.Adam,
                 lr=0.001, beta=1.0):
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
        self.vae = vae
        self.beta = beta

        # create optimizers for each model in the ensemble
        self.vae_optim = optim(learning_rate=lr)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   x):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        with tf.GradientTape() as tape:

            latent = self.vae.encode(x, training=True)
            z = latent.mean()
            prediction = self.vae.decode(z)

            nll = -prediction.log_prob(x)

            kld = latent.kl_divergence(
                tfpd.MultivariateNormalDiag(
                    loc=tf.zeros_like(z), scale_diag=tf.ones_like(z)))

            total_loss = tf.reduce_mean(
                nll) + tf.reduce_mean(kld) * self.beta

        variables = self.vae.trainable_variables

        self.vae_optim.apply_gradients(zip(
            tape.gradient(total_loss, variables), variables))

        statistics[f'vae/train/nll'] = nll
        statistics[f'vae/train/kld'] = kld

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      x):
        """Perform a validation step on an ensemble of models
        without using bootstrapping weights

        Args:

        x: tf.Tensor
            a batch of validation inputs shaped like [batch_size, channels]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        latent = self.vae.encode(x, training=True)
        z = latent.mean()
        prediction = self.vae.decode(z)

        nll = -prediction.log_prob(x)

        kld = latent.kl_divergence(
            tfpd.MultivariateNormalDiag(
                loc=tf.zeros_like(z), scale_diag=tf.ones_like(z)))

        statistics[f'vae/validate/nll'] = nll
        statistics[f'vae/validate/kld'] = kld

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
            for name, tensor in self.train_step(x).items():
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
            for name, tensor in self.validate_step(x).items():
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
