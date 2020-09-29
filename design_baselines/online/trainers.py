from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class ConservativeMaximumLikelihood(tf.Module):

    def __init__(self,
                 forward_model,
                 forward_model_opt=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001,
                 initial_alpha=1.0,
                 alpha_opt=tf.keras.optimizers.Adam,
                 alpha_lr=0.05,
                 target_conservatism=1.0,
                 negatives_fraction=0.5,
                 lookahead_steps=50,
                 lookahead_backprop=True,
                 solver_lr=0.01,
                 solver_interval=10,
                 solver_warmup=500,
                 solver_steps=1,
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
        self.forward_model_opt = forward_model_opt(
            learning_rate=forward_model_lr)

        # lagrangian dual descent variables
        log_alpha = np.log(initial_alpha).astype(np.float32)
        self.log_alpha = tf.Variable(log_alpha)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.math.softplus)
        self.alpha_opt = alpha_opt(learning_rate=alpha_lr)

        # parameters for controlling the lagrangian dual descent
        self.target_conservatism = target_conservatism
        self.negatives_fraction = negatives_fraction
        self.lookahead_steps = lookahead_steps
        self.lookahead_backprop = lookahead_backprop

        # parameters for controlling learning rate for negative samples
        self.solver_lr = solver_lr
        self.solver_interval = solver_interval
        self.solver_warmup = solver_warmup
        self.solver_steps = solver_steps

        # extra parameters for controlling data noise
        self.is_discrete = is_discrete
        self.continuous_noise_std = continuous_noise_std
        self.discrete_smoothing = discrete_smoothing

        # save the state of the solution found by the model
        self.step = tf.Variable(tf.constant(0, dtype=tf.int32))
        self.solution = None
        self.done = None

    @tf.function(experimental_relax_shapes=True)
    def lookahead(self,
                  x,
                  steps,
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

        # gradient ascent on the predicted score
        def gradient_step(xt):
            with tf.GradientTape() as tape:
                tape.watch(xt)
                score = self.forward_model.get_distribution(
                    tf.math.softmax(xt)
                    if self.is_discrete else xt, **kwargs).mean()
            return xt + self.solver_lr * tape.gradient(score, xt)

        # use a while loop to perform gradient ascent on the score
        return tf.while_loop(
            lambda xt: True,
            gradient_step, (x,), maximum_iterations=steps)[0]

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

        self.step.assign_add(1)
        statistics = dict()
        batch_dim = tf.shape(y)[0]

        # corrupt the inputs with noise
        x0 = soft_noise(x, self.discrete_smoothing) \
            if self.is_discrete else \
            cont_noise(x, self.continuous_noise_std)

        with tf.GradientTape(persistent=True) as tape:

            # calculate the prediction error and accuracy of the model
            d0 = self.forward_model.get_distribution(x0, training=True)
            nll = -d0.log_prob(y)
            statistics[f'train/nll'] = nll

            # evaluate how correct the rank fo the model predictions are
            rank_corr = spearman(y[:, 0], d0.mean()[:, 0])
            statistics[f'train/rank_corr'] = rank_corr

            # determine which samples to start optimization from
            p0 = tf.math.log(x0) if self.is_discrete else x0
            p0 = tf.where(tf.random.uniform([batch_dim] + [1 for _ in x.shape[1:]])
                          < self.negatives_fraction,
                          p0, self.solution[:batch_dim])

            # calculate the conservative gap
            pt = self.lookahead(p0, self.lookahead_steps, training=False)
            pt = tf.math.softmax(pt) if self.is_discrete else pt
            if not self.lookahead_backprop:
                pt = tf.stop_gradient(pt)

            # calculate the prediction error and accuracy of the model
            dt = self.forward_model.get_distribution(pt, training=False)
            conservatism = (dt.mean() - d0.mean())[:, 0]
            statistics[f'train/conservatism'] = conservatism

            # build a lagrangian for dual descent
            alpha_loss = (self.alpha * self.target_conservatism -
                          self.alpha * conservatism)
            statistics[f'train/alpha'] = self.alpha

            # loss that combines maximum likelihood with a constraint
            model_loss = nll + self.alpha * conservatism
            total_loss = tf.reduce_mean(model_loss)
            alpha_loss = tf.reduce_mean(alpha_loss)

        # take gradient steps on the model
        grads = tape.gradient(
            total_loss, self.forward_model.trainable_variables)
        self.forward_model_opt.apply_gradients(
            zip(grads, self.forward_model.trainable_variables))

        # take gradient steps on alpha
        grads = tape.gradient(alpha_loss, self.log_alpha)
        self.alpha_opt.apply_gradients([[grads, self.log_alpha]])

        # occasionally take gradient ascent steps on the solution
        if tf.logical_and(
                tf.equal(tf.math.mod(self.step, self.solver_interval), 0),
                tf.math.greater_equal(self.step, self.solver_warmup)):

            # only update the solutions that are not frozen
            update = self.lookahead(
                self.solution, self.solver_steps, training=False)
            self.solution.assign(
                tf.where(self.done, self.solution, update))

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
        x0 = soft_noise(x, self.discrete_smoothing) \
            if self.is_discrete else \
            cont_noise(x, self.continuous_noise_std)

        # calculate the prediction error and accuracy of the model
        d0 = self.forward_model.get_distribution(x0, training=False)
        nll = -d0.log_prob(y)
        statistics[f'validate/nll'] = nll

        # evaluate how correct the rank fo the model predictions are
        rank_corr = spearman(y[:, 0], d0.mean()[:, 0])
        statistics[f'validate/rank_corr'] = rank_corr

        # determine which samples to start optimization from
        p0 = tf.math.log(x0) if self.is_discrete else x0
        p0 = tf.where(tf.random.uniform([batch_dim] + [1 for _ in x.shape[1:]])
                      < self.negatives_fraction,
                      p0, self.solution[:batch_dim])

        # calculate the conservative gap
        pt = self.lookahead(p0, self.lookahead_steps, training=False)
        pt = tf.math.softmax(pt) if self.is_discrete else pt
        if not self.lookahead_backprop:
            pt = tf.stop_gradient(pt)

        # calculate the prediction error and accuracy of the model
        dt = self.forward_model.get_distribution(pt, training=False)
        conservatism = (dt.mean() - d0.mean())[:, 0]
        statistics[f'validate/conservatism'] = conservatism
        return statistics
