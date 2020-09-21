from design_baselines.utils import spearman
from design_baselines.utils import gumb_noise
from design_baselines.utils import cont_noise
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class OnlineSolver(tf.Module):

    def __init__(self,
                 forward_model,
                 forward_model_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001,
                 target_conservative_gap=tf.constant(10.0),
                 initial_alpha=0.0001,
                 alpha_optim=tf.keras.optimizers.Adam,
                 alpha_lr=0.001,
                 lookahead_lr=0.001,
                 lookahead_steps=100,
                 lookahead_backprop=False,
                 lookahead_swap=0.1,
                 solver_period=50,
                 solver_warmup=500,
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
        self.is_discrete = is_discrete
        self.noise_std = noise_std
        self.keep = keep
        self.temp = temp

        # parameters for the lookahead optimizer
        self.lookahead_lr = lookahead_lr
        self.lookahead_steps = lookahead_steps
        self.lookahead_backprop = lookahead_backprop
        self.lookahead_swap = lookahead_swap

        # create machinery for storing the best solution
        self.solver_period = solver_period
        self.solver_warmup = solver_warmup
        self.soln = None

    @tf.function(experimental_relax_shapes=True)
    def lookahead(self,
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

        # use the forward model to create adversarial examples
        def body(xt):
            with tf.GradientTape() as tape:
                tape.watch(xt)
                z = tf.math.softmax(xt) if self.is_discrete else xt
                pred = self.fm.get_distribution(z, **kwargs).mean()
            return xt + self.lookahead_lr * tape.gradient(pred, xt)
        x = tf.math.log(x) if self.is_discrete else x
        x = tf.while_loop(lambda xt: True, body, (x,), swap_memory=True,
                          maximum_iterations=self.lookahead_steps)[0]
        return tf.math.softmax(x) if self.is_discrete else x

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   i,
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
        batch_dim = tf.shape(y)[0]

        # corrupt the inputs with noise
        if self.is_discrete:
            x = gumb_noise(x, keep=self.keep, temp=self.temp)
        else:
            x = cont_noise(x, self.noise_std)

        with tf.GradientTape(persistent=True) as tape:

            # calculate the prediction error and accuracy of the model
            d = self.fm.get_distribution(x, training=True)
            nll = -d.log_prob(y)
            statistics[f'train/nll'] = nll

            # evaluate how correct the rank fo the model predictions are
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])
            statistics[f'train/rank_corr'] = rank_correlation

            # select the starting points for lookahead
            xi = tf.where(
                tf.random.uniform([batch_dim, 1]) > self.lookahead_swap,
                self.soln[:batch_dim], x)

            # calculate the conservative gap
            perturb = self.lookahead(xi, training=False)
            if not self.lookahead_backprop:
                perturb = tf.stop_gradient(perturb)

            # calculate the prediction error and accuracy of the model
            perturb_d = self.fm.get_distribution(perturb, training=False)

            # build the lagrangian loss
            conservative_gap = (perturb_d.mean() - d.mean())[:, 0]
            statistics[f'train/gap'] = conservative_gap

            gap = (self.alpha * self.target_conservative_gap -
                   self.alpha * conservative_gap)
            statistics[f'train/alpha_loss'] = gap
            statistics[f'train/alpha'] = self.alpha

            # model loss that combines maximum likelihood with a constraint
            model_loss = nll + self.alpha * conservative_gap

            # build the total and lagrangian losses
            denom = tf.reduce_sum(b)
            total_loss = tf.math.divide_no_nan(tf.reduce_sum(b * model_loss), denom)
            alpha_loss = tf.math.divide_no_nan(tf.reduce_sum(b * gap), denom)

        grads = tape.gradient(total_loss, self.fm.trainable_variables)
        self.fm_optim.apply_gradients(zip(grads, self.fm.trainable_variables))
        grads = tape.gradient(alpha_loss, self.log_alpha)
        self.alpha_optim.apply_gradients([[grads, self.log_alpha]])

        # optimizer only under certain conditions
        if tf.logical_and(
                tf.greater_equal(i, self.solver_warmup),
                tf.equal(tf.math.floormod(i, self.solver_period), 0)):

            # update the current solution using gradient ascent
            with tf.GradientTape() as tape:
                z = tf.nn.softmax(self.soln) if self.is_discrete else self.soln
                score = self.fm.get_distribution(z, training=False).mean()

            # push gradients to the solution
            grads = tape.gradient(score, self.soln)
            self.soln.assign(self.soln + self.lookahead_lr * grads)

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
        batch_dim = tf.shape(y)[0]

        # corrupt the inputs with noise
        if self.is_discrete:
            x = gumb_noise(x, keep=self.keep, temp=self.temp)
        else:
            x = cont_noise(x, self.noise_std)

        # calculate the prediction error and accuracy of the model
        d = self.fm.get_distribution(x, training=False)
        nll = -d.log_prob(y)
        statistics[f'validate/nll'] = nll

        # evaluate how correct the rank fo the model predictions are
        rank_correlation = spearman(y[:, 0], d.mean()[:, 0])
        statistics[f'validate/rank_corr'] = rank_correlation

        # select the starting points for lookahead
        xi = tf.where(
            tf.random.uniform([batch_dim, 1]) > self.lookahead_swap,
            self.soln[:batch_dim], x)

        # calculate the conservative gap
        perturb = self.lookahead(xi, training=False)

        # calculate the prediction error and accuracy of the model
        perturb_d = self.fm.get_distribution(perturb, training=False)

        # build the lagrangian loss
        conservative_gap = (perturb_d.mean() - d.mean())[:, 0]
        statistics[f'validate/gap'] = conservative_gap

        return statistics

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
