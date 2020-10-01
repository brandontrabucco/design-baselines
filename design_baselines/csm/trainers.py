from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
import tensorflow as tf
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import numpy as np


class Ensemble(tf.Module):

    def __init__(self,
                 forward_models,
                 forward_model_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001,
                 is_discrete=False,
                 continuous_noise_std=0.0,
                 discrete_smoothing=0.0):
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
        self.forward_models = forward_models
        self.bootstraps = len(forward_models)

        # create machinery for sampling adversarial examples
        self.is_discrete = is_discrete
        self.noise_std = continuous_noise_std
        self.keep = discrete_smoothing

        # create optimizers for each model in the ensemble
        self.forward_model_optims = [
            forward_model_optim(learning_rate=forward_model_lr)
            for i in range(self.bootstraps)]

    def get_distribution(self,
                         x,
                         **kwargs):
        """Build the mixture distribution implied by the set of oracles
        that are trained in this module

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        distribution: tfpd.Distribution
            the mixture of gaussian distributions implied by the oracles
        """

        # get the distribution parameters for all models
        params = defaultdict(list)
        for fm in self.forward_models:
            for key, val in fm.get_params(x, **kwargs).items():
                params[key].append(val)

        # stack the parameters in a new component axis
        for key, val in params.items():
            params[key] = tf.stack(val, axis=-1)

        # build the mixture distribution using the family of component one
        weights = tf.fill([self.bootstraps], 1 / self.bootstraps)
        return tfpd.MixtureSameFamily(tfpd.Categorical(
            probs=weights), self.forward_models[0].distribution(**params))

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   x,
                   y,
                   b):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: tf.Tensor
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

        # corrupt the inputs with noise
        x0 = soft_noise(x, self.keep) \
            if self.is_discrete else cont_noise(x, self.noise_std)

        for i in range(self.bootstraps):
            fm = self.forward_models[i]
            fm_optim = self.forward_model_optims[i]

            with tf.GradientTape(persistent=True) as tape:

                # calculate the prediction error and accuracy of the model
                d = fm.get_distribution(x0, training=True)
                nll = -d.log_prob(y)[:, 0]

                # evaluate how correct the rank fo the model predictions are
                rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

                # build the total loss and weight by the bootstrap
                total_loss = tf.math.divide_no_nan(
                    tf.reduce_sum(b[:, i] * nll), tf.reduce_sum(b[:, i]))

            grads = tape.gradient(total_loss, fm.trainable_variables)
            fm_optim.apply_gradients(zip(grads, fm.trainable_variables))

            statistics[f'oracle_{i}/train/nll'] = nll
            statistics[f'oracle_{i}/train/rank_corr'] = rank_correlation

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

        # corrupt the inputs with noise
        x0 = soft_noise(x, self.keep) \
            if self.is_discrete else cont_noise(x, self.noise_std)

        for i in range(self.bootstraps):
            fm = self.forward_models[i]

            # calculate the prediction error and accuracy of the model
            d = fm.get_distribution(x0, training=False)
            nll = -d.log_prob(y)[:, 0]

            # evaluate how correct the rank fo the model predictions are
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            statistics[f'oracle_{i}/validate/nll'] = nll
            statistics[f'oracle_{i}/validate/rank_corr'] = rank_correlation

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
        for x, y, b in dataset:
            for name, tensor in self.train_step(x, y, b).items():
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
        for i in range(self.bootstraps):
            saveables[f'forward_model_{i}'] = self.forward_models[i]
            saveables[f'forward_model_optim_{i}'] = self.forward_model_optims[i]
        return saveables


class ConservativeMaximumLikelihood(tf.Module):

    def __init__(self,
                 forward_model,
                 forward_model_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001,
                 target_conservatism=0.5,
                 initial_alpha=1.0,
                 alpha_optim=tf.keras.optimizers.Adam,
                 alpha_lr=0.05,
                 perturbation_lr=0.01,
                 perturbation_steps=50,
                 perturbation_backprop=True,
                 is_discrete=False,
                 continuous_noise_std=0.0,
                 discrete_smoothing=0.6):
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
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_optim(learning_rate=forward_model_lr)

        # lagrangian dual descent
        self.log_alpha = tf.Variable(np.log(initial_alpha).astype(np.float32))
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.math.softplus)
        self.alpha_opt = alpha_optim(learning_rate=alpha_lr)

        # create machinery for sampling adversarial examples
        self.target_conservatism = target_conservatism
        self.perturbation_lr = perturbation_lr
        self.perturbation_steps = perturbation_steps
        self.perturbation_backprop = perturbation_backprop

        # extra parameters for controlling data formats
        self.is_discrete = is_discrete
        self.continuous_noise_std = continuous_noise_std
        self.discrete_smoothing = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def optimize(self,
                 x,
                 **kwargs):
        """Using gradient descent find adversarial versions of x that maximize
        the score predicted by the forward model

        Args:

        x: tf.Tensor
            the original value of the tensor being optimized
        i: int
            the index of the forward model used when back propagating

        Returns:

        optimized_x: tf.Tensor
            the perturbed value of x that maximizes the score function
        """

        # use the forward model to create adversarial examples
        def body(xt):
            with tf.GradientTape() as tape:
                tape.watch(xt)
                solution = tf.math.softmax(xt) if self.is_discrete else xt
                score = self.forward_model.get_distribution(solution, **kwargs).mean()
            return xt + self.perturbation_lr * tape.gradient(score, xt)
        x = tf.math.log(x) if self.is_discrete else x
        x = tf.while_loop(lambda xt: True, body, (x,), swap_memory=True,
                          maximum_iterations=self.perturbation_steps)[0]
        return tf.math.softmax(x) if self.is_discrete else x

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   x,
                   y,
                   b):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of training labels shaped like [batch_size, 1]
        b: tf.Tensor
            bootstrap indicators shaped like [batch_size, num_oracles]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        # corrupt the inputs with noise
        x0 = soft_noise(x, self.discrete_smoothing) \
            if self.is_discrete else cont_noise(x, self.continuous_noise_std)

        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:

            # calculate the prediction error and accuracy of the model
            d = self.forward_model.get_distribution(x0, training=True)
            nll = -d.log_prob(y)
            statistics[f'train/nll'] = nll

            # evaluate how correct the rank fo the model predictions are
            fm_rank_corr = spearman(y[:, 0], d.mean()[:, 0])
            statistics[f'train/rank_corr'] = fm_rank_corr

            # calculate the conservative gap
            perturb = self.optimize(x0, training=False)
            if not self.perturbation_backprop:
                perturb = tf.stop_gradient(perturb)

            # calculate the prediction error and accuracy of the model
            perturb_d = self.forward_model.get_distribution(perturb, training=False)
            conservative_gap = (perturb_d.mean() - d.mean())[:, 0]
            statistics[f'train/conservatism'] = conservative_gap

            # build a lagrangian
            alpha_loss = (self.alpha * self.target_conservatism -
                          self.alpha * conservative_gap)

            # model loss that combines maximum likelihood with a constraint
            model_loss = nll + self.alpha * conservative_gap
            statistics[f'train/alpha_loss'] = alpha_loss

            # build the total and lagrangian losses
            denom = tf.reduce_sum(b)
            total_loss = tf.math.divide_no_nan(
                tf.reduce_sum(b * model_loss), denom)
            alpha_loss = tf.math.divide_no_nan(
                tf.reduce_sum(b * alpha_loss), denom)

        grads = tape.gradient(
            total_loss, self.forward_model.trainable_variables)
        self.forward_model_opt.apply_gradients(
            zip(grads, self.forward_model.trainable_variables))
        self.alpha_opt.apply_gradients([[
            tape.gradient(alpha_loss, self.log_alpha), self.log_alpha]])
        statistics[f'train/alpha'] = self.alpha

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

        # corrupt the inputs with noise
        x0 = soft_noise(x, self.discrete_smoothing) \
            if self.is_discrete else cont_noise(x, self.continuous_noise_std)

        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d = self.forward_model.get_distribution(x0, training=False)
        nll = -d.log_prob(y)
        statistics[f'validate/nll'] = nll

        # evaluate how correct the rank fo the model predictions are
        rank_correlation = spearman(y[:, 0], d.mean()[:, 0])
        statistics[f'validate/rank_corr'] = rank_correlation

        # calculate the conservative gap
        perturb = self.optimize(x0, training=False)

        # calculate the prediction error and accuracy of the model
        perturb_d = self.forward_model.get_distribution(perturb, training=False)

        # build the lagrangian loss
        conservative_gap = (perturb_d.mean() - d.mean())[:, 0]
        statistics[f'validate/conservatism'] = conservative_gap

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
        for i in range(self.bootstraps):
            saveables[f'forward_model'] = self.forward_model
            saveables[f'forward_model_opt'] = self.forward_model_opt
            saveables[f'log_alpha'] = self.log_alpha
            saveables[f'alpha_opt'] = self.alpha_opt
        return saveables
