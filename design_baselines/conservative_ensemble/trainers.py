from design_baselines.utils import spearman
from design_baselines.utils import add_gumbel_noise
from design_baselines.utils import add_continuous_noise
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
                 perturbation_steps=100,
                 perturbation_backprop=False,
                 is_discrete=False,
                 noise_std=0.0,
                 keep=0.0,
                 temp=0.0):
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
        self.fm = forward_model
        self.fm_optim = forward_model_optim(learning_rate=forward_model_lr)
        self.target_conservative_gap = target_conservative_gap

        # create training machinery for lagrange multiplier
        self.log_alpha = tf.Variable(np.log(initial_alpha).astype(np.float32))
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        self.alpha_optim = alpha_optim(learning_rate=alpha_lr)

        # create machinery for sampling adversarial examples
        self.perturbation_lr = perturbation_lr
        self.perturbation_steps = perturbation_steps
        self.perturbation_backprop = perturbation_backprop
        self.is_discrete = is_discrete
        self.noise_std = noise_std
        self.keep = keep
        self.temp = temp

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

        def cond_fn(xt, t):
            return tf.less(t, self.perturbation_steps)

        def body_fn(xt, t):
            with tf.GradientTape() as tape:
                tape.watch(xt)
                solution = tf.math.softmax(xt) if self.is_discrete else xt
                y = self.fm.get_distribution(solution, **kwargs).mean()
            return xt + self.perturbation_lr * tape.gradient(y, xt), t + 1

        # use the forward model to make adversarial examples
        x0 = tf.math.log(x) if self.is_discrete else x
        xn = tf.while_loop(cond_fn, body_fn, (x0, tf.constant(0)))[0]
        return tf.math.softmax(xn) if self.is_discrete else xn

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
        x0 = add_gumbel_noise(x, self.keep, self.temp) \
            if self.is_discrete else add_continuous_noise(x, self.noise_std)

        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:

            # calculate the prediction error and accuracy of the model
            d = self.fm.get_distribution(x0, training=True)
            nll = -d.log_prob(y)

            # evaluate how correct the rank fo the model predictions are
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            # calculate the conservative gap
            perturb = self.optimize(x0, training=False)
            if not self.perturbation_backprop:
                perturb = tf.stop_gradient(perturb)

            # calculate the prediction error and accuracy of the model
            perturb_d = self.fm.get_distribution(perturb, training=False)

            # build the lagrangian loss
            conservative_gap = (perturb_d.mean() - d.mean())[:, 0]
            gap = (self.alpha * self.target_conservative_gap -
                   self.alpha * conservative_gap)

            # model loss that combines maximum likelihood with a constraint
            model_loss = nll + self.alpha * conservative_gap

            # build the total and lagrangian losses
            denom = tf.reduce_sum(b)
            total_loss = tf.math.divide_no_nan(
                tf.reduce_sum(b * model_loss), denom)
            alpha_loss = tf.math.divide_no_nan(
                tf.reduce_sum(b * gap), denom)

        grads = tape.gradient(total_loss, self.fm.trainable_variables)
        self.fm_optim.apply_gradients(zip(grads, self.fm.trainable_variables))
        grads = tape.gradient(alpha_loss, self.log_alpha)
        self.alpha_optim.apply_gradients([[grads, self.log_alpha]])

        statistics[f'train/nll'] = nll
        statistics[f'train/alpha'] = self.alpha
        statistics[f'train/alpha_loss'] = gap
        statistics[f'train/rank_corr'] = rank_correlation
        statistics[f'train/gap'] = conservative_gap

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
        x0 = add_gumbel_noise(x, self.keep, self.temp) \
            if self.is_discrete else add_continuous_noise(x, self.noise_std)

        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d = self.fm.get_distribution(x0, training=False)
        nll = -d.log_prob(y)

        # evaluate how correct the rank fo the model predictions are
        rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

        # calculate the conservative gap
        perturb = self.optimize(x0, training=False)

        # calculate the prediction error and accuracy of the model
        perturb_d = self.fm.get_distribution(perturb, training=False)

        # build the lagrangian loss
        conservative_gap = (perturb_d.mean() - d.mean())[:, 0]

        statistics[f'validate/nll'] = nll
        statistics[f'validate/rank_corr'] = rank_correlation
        statistics[f'validate/gap'] = conservative_gap

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
            saveables[f'forward_model'] = self.fm
            saveables[f'forward_model_optim'] = self.fm_optim
            saveables[f'log_alpha'] = self.log_alpha
            saveables[f'alpha_optim'] = self.alpha_optim
        return saveables
