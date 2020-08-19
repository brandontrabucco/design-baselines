from design_baselines.utils import spearman
from design_baselines.utils import add_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class ConservativeEnsemble(tf.Module):

    def __init__(self,
                 forward_models,
                 forward_model_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001,
                 target_conservative_gap=tf.constant(10.0),
                 initial_alpha=0.0001,
                 alpha_optim=tf.keras.optimizers.Adam,
                 alpha_lr=0.001,
                 perturbation_lr=0.001,
                 perturbation_steps=100,
                 is_discrete=False,
                 input_noise=0.0):
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
        self.target_conservative_gap = target_conservative_gap
        self.bootstraps = len(forward_models)

        self.forward_model_optims = [
            forward_model_optim(learning_rate=forward_model_lr)
            for i in range(self.bootstraps)]

        # create training machinery for lagrange multiplier
        self.log_alphas = [
            tf.Variable(np.log(initial_alpha).astype(np.float32))
            for i in range(self.bootstraps)]

        self.alphas = [
            tfp.util.DeferredTensor(self.log_alphas[i], tf.exp)
            for i in range(self.bootstraps)]

        self.alpha_optims = [
            alpha_optim(learning_rate=alpha_lr)
            for i in range(self.bootstraps)]

        # create machinery for sampling adversarial examples
        self.perturbation_lr = perturbation_lr
        self.perturbation_steps = perturbation_steps
        self.is_discrete = is_discrete
        self.input_noise = input_noise

    def get_distribution(self,
                         x,
                         **kwargs):
        """Build the mixture distribution implied by the set of oracles
        that are trained in this module

        Args:

        X: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        distribution: tfpd.Distribution
            the mixture of gaussian distributions implied by the oracles
        """

        # get the distribution parameters for all models
        params = defaultdict(list)
        for fm in self.forward_models:
            for key, val in fm.get_parameters(x, **kwargs).items():
                params[key].append(val)

        # stack the parameters in a new component axis
        for key, val in params.items():
            params[key] = tf.stack(val, axis=1)

        # build the mixture distribution using the family of component one
        weights = tf.fill([self.bootstraps], 1 / self.bootstraps)
        return tfpd.MixtureSameFamily(tfpd.Categorical(
            probs=weights), self.forward_models[0].distribution(**params))

    def optimize(self,
                 x,
                 fm,
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

        # use the selected forward model to find adversarial examples
        x = tf.math.log(x) if self.is_discrete else x
        for step in range(self.perturbation_steps):
            with tf.GradientTape() as tape:
                tape.watch(x)
                solution = tf.math.softmax(x) if self.is_discrete else x
                score = fm.get_distribution(solution, **kwargs).mean()
            x = x + self.perturbation_lr * tape.gradient(score, x)
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

        statistics = dict()

        for i in range(self.bootstraps):
            fm = self.forward_models[i]
            fm_optim = self.forward_model_optims[i]
            alpha = self.alphas[i]
            log_alpha = self.log_alphas[i]
            alpha_optim = self.alpha_optims[i]

            x0 = add_noise(x, self.input_noise, self.is_discrete)
            with tf.GradientTape(persistent=True) as tape:

                # calculate the prediction error and accuracy of the model
                d = fm.get_distribution(x0, training=True)
                nll = -d.log_prob(y)

                # evaluate how correct the rank fo the model predictions are
                rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

                # calculate the conservative gap
                perturb = tf.stop_gradient(self.optimize(x0, fm))

                # calculate the prediction error and accuracy of the model
                perturb_d = fm.get_distribution(perturb, training=True)

                # build the lagrangian loss
                conservative_gap = (perturb_d.mean() - d.mean())[:, 0]
                gap = (alpha * self.target_conservative_gap -
                       alpha * conservative_gap)

                # model loss that combines maximum likelihood with a constraint
                model_loss = nll + alpha * conservative_gap

                # build the total and lagrangian losses
                denom = tf.reduce_sum(b[:, i])
                total_loss = tf.math.divide_no_nan(
                    tf.reduce_sum(b[:, i] * model_loss), denom)
                alpha_loss = tf.math.divide_no_nan(
                    tf.reduce_sum(b[:, i] * gap), denom)

            grads = tape.gradient(total_loss, fm.trainable_variables)
            fm_optim.apply_gradients(zip(grads, fm.trainable_variables))
            grads = tape.gradient(alpha_loss, log_alpha)
            alpha_optim.apply_gradients([[grads, log_alpha]])

            statistics[f'oracle_{i}/train/nll'] = nll
            statistics[f'oracle_{i}/train/max_logstd'] = fm.max_logstd
            statistics[f'oracle_{i}/train/min_logstd'] = fm.min_logstd
            statistics[f'oracle_{i}/train/alpha'] = alpha
            statistics[f'oracle_{i}/train/alpha_loss'] = gap
            statistics[f'oracle_{i}/train/rank_corr'] = rank_correlation
            statistics[f'oracle_{i}/train/gap'] = conservative_gap

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

        for i in range(self.bootstraps):
            fm = self.forward_models[i]

            # calculate the prediction error and accuracy of the model
            d = fm.get_distribution(x, training=False)
            nll = -d.log_prob(y)

            # evaluate how correct the rank fo the model predictions are
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            # calculate the conservative gap
            perturb = tf.stop_gradient(self.optimize(x, fm))

            # calculate the prediction error and accuracy of the model
            perturb_d = fm.get_distribution(perturb, training=False)

            # build the lagrangian loss
            conservative_gap = (perturb_d.mean() - d.mean())[:, 0]

            statistics[f'oracle_{i}/validate/nll'] = nll
            statistics[f'oracle_{i}/validate/rank_corr'] = rank_correlation
            statistics[f'oracle_{i}/validate/gap'] = conservative_gap

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
            saveables[f'forward_model_{i}'] = self.forward_models[i]
            saveables[f'forward_model_optim_{i}'] = self.forward_model_optims[i]
            saveables[f'log_alpha_{i}'] = self.log_alphas[i]
            saveables[f'alpha_optim_{i}'] = self.alpha_optims[i]
        return saveables
