from design_baselines.utils import spearman
from design_baselines.utils import cont_noise
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow as tf
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
                 solver_beta=0.0,
                 solver_lr=0.01,
                 solver_interval=10,
                 solver_warmup=500,
                 solver_steps=1,
                 constraint_type="mix",
                 entropy_coefficient=0.9,
                 continuous_noise_std=0.0):
        """Build a trainer for an conservative forward model trained using
        an adversarial negative sampling procedure.

        Args:

        forward_model: tf.keras.Model
            a tf.keras.Model that accepts a batch of designs x as input
            and predicts a batch of scalar scores as output
        forward_model_opt: tf.keras.optimizers.Optimizer
            an optimizer that determines how the weights of the forward
            model are updated during training (eg: Adam)
        forward_model_lr: float
            the learning rate passed to the optimizer used to update the
            forward model weights (eg: 0.0003)
        initial_alpha: float
            the initial value for the lagrange multiplier, which is jointly
            optimized with the forward model during training.
        alpha_opt: tf.keras.optimizers.Optimizer
            an optimizer that determines how the lagrange multiplier
            is updated during training (eg: Adam)
        alpha_lr: float
            the learning rate passed to the optimizer used to update the
            lagrange multiplier (eg: 0.05)
        target_conservatism: float
            the degree of overestimation that the forward model is trained
            via dual gradient ascent to have no more than
        negatives_fraction: float
            (deprecated) a deprecated parameter that should be set to 1,
            and will be phased out in future versions
        lookahead_steps: int
            the number of steps of gradient ascent used when finding
            negative samples for training the forward model
        lookahead_backprop: bool
            whether or not to allow gradients to flow back into the
            negative sampler, required for gradients to be unbiased
        solver_beta: float
            the value of beta to use during trust region optimization,
            a value as large as 0.9 typically works
        solver_lr: float
            the learning rate used to update negative samples when
            optimizing them to maximize the forward models predictions
        solver_interval: int
            the number of training steps for the forward model between
            updates to the set of solution particles
        solver_warmup: int
            the number of steps to train the forward model for before updating
            the set of solution particles for the first time
        solver_steps: int
            (deprecated) the number of steps used to update the set of
            solution particles at once, set this to 1
        constraint_type: str in ["dataset", "mix", "solution"]
            (deprecated) a deprecated parameter that should always be set
            equal to "mix" with negatives_fraction = 1.0
        continuous_noise_std: float
            standard deviation of gaussian noise added to the design variable
            x while training the forward model
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
        self.solver_beta = solver_beta
        self.entropy_coefficient = entropy_coefficient

        # extra parameters for controlling data noise
        self.continuous_noise_std = continuous_noise_std
        self.constraint_type = constraint_type

        # save the state of the solution found by the model
        self.step = tf.Variable(tf.constant(0, dtype=tf.int32))
        self.solution = None
        self.particle_loss = None
        self.particle_constraint = None
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
        steps: int
            the number of optimization steps taken

        Returns:

        optimized_x: tf.Tensor
            the perturbed value of x that maximizes the score function
        """

        # gradient ascent on the predicted score
        def gradient_step(xt):
            with tf.GradientTape() as tape:
                tape.watch(xt)
                entropy = tf.reduce_mean(
                    (xt[tf.newaxis] - xt[:, tf.newaxis]) ** 2)
                score = (self.entropy_coefficient * entropy +
                         self.forward_model(xt, **kwargs))
            return xt + self.solver_lr * tape.gradient(score, xt)

        # use a while loop to perform gradient ascent on the score
        return tf.while_loop(
            lambda xt: True, gradient_step, (x,),
            maximum_iterations=steps)[0]

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   x,
                   y):
        """Perform a training step of gradient descent on the loss function
        of a conservative objective model

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
        x = cont_noise(x, self.continuous_noise_std)

        with tf.GradientTape(persistent=True) as tape:

            # calculate the prediction error and accuracy of the model
            d = self.forward_model(x, training=True)
            mse = tf.keras.losses.mean_squared_error(y, d)
            statistics[f'train/mse'] = mse

            # evaluate how correct the rank fo the model predictions are
            rank_corr = spearman(y[:, 0], d[:, 0])
            statistics[f'train/rank_corr'] = rank_corr

            # calculate negative samples starting from the dataset
            x_pos = x
            x_pos = tf.where(tf.random.uniform([batch_dim] + [1 for _ in x.shape[1:]])
                             < self.negatives_fraction, x_pos, self.solution[:batch_dim])
            x_neg = self.lookahead(x_pos, self.lookahead_steps, training=False)
            if not self.lookahead_backprop:
                x_neg = tf.stop_gradient(x_neg)

            # calculate the prediction error and accuracy of the model
            d_pos = self.forward_model(
                {"dataset": x, "mix": x_pos, "solution": self.solution[:batch_dim]}
                [self.constraint_type], training=False)
            d_neg = self.forward_model(x_neg, training=False)
            conservatism = d_neg[:, 0] - d_pos[:, 0]
            statistics[f'train/conservatism'] = conservatism

            # build a lagrangian for dual descent
            alpha_loss = (self.alpha * self.target_conservatism -
                          self.alpha * conservatism)
            statistics[f'train/alpha'] = self.alpha

            multiplier_loss = 0.0
            last_weight = self.forward_model.trainable_variables[-1]
            if tf.shape(tf.reshape(last_weight, [-1]))[0] == 1:
                statistics[f'train/tanh_multipier'] = \
                    self.forward_model.trainable_variables[-1]

            # loss that combines maximum likelihood with a constraint
            model_loss = mse + self.alpha * conservatism + multiplier_loss
            total_loss = tf.reduce_mean(model_loss)
            alpha_loss = tf.reduce_mean(alpha_loss)

        # initialize stateful variables at the first iteration
        if self.particle_loss is None:
            initialization = tf.zeros_like(conservatism)
            self.particle_loss = tf.Variable(initialization)
            self.particle_constraint = tf.Variable(initialization)

        # calculate gradients using the model
        alpha_grads = tape.gradient(alpha_loss, self.log_alpha)
        model_grads = tape.gradient(
            total_loss, self.forward_model.trainable_variables)

        # occasionally take gradient ascent steps on the solution
        if tf.logical_and(
                tf.equal(tf.math.mod(self.step, self.solver_interval), 0),
                tf.math.greater_equal(self.step, self.solver_warmup)):
            with tf.GradientTape() as tape:

                # take gradient steps on the model
                self.alpha_opt.apply_gradients([[alpha_grads, self.log_alpha]])
                self.forward_model_opt.apply_gradients(
                    zip(model_grads, self.forward_model.trainable_variables))

                # calculate the predicted score of the current solution
                current_score_new_model = self.forward_model(
                    self.solution, training=False)[:, 0]

                # look into the future and evaluate future solutions
                future_new_model = self.lookahead(
                    self.solution, self.solver_steps, training=False)
                future_score_new_model = self.forward_model(
                    future_new_model, training=False)[:, 0]

                # evaluate the conservatism of the current solution
                particle_loss = (self.solver_beta * future_score_new_model -
                                 current_score_new_model)
                update = (self.solution - self.solver_lr *
                          tape.gradient(particle_loss, self.solution))

            # if optimizer conservatism passes threshold stop optimizing
            self.solution.assign(tf.where(self.done, self.solution, update))
            self.particle_loss.assign(particle_loss)
            self.particle_constraint.assign(
                future_score_new_model - current_score_new_model)

        else:

            # take gradient steps on the model
            self.alpha_opt.apply_gradients([[alpha_grads, self.log_alpha]])
            self.forward_model_opt.apply_gradients(
                zip(model_grads, self.forward_model.trainable_variables))

        statistics[f'train/done'] = tf.cast(self.done, tf.float32)
        statistics[f'train/particle_loss'] = self.particle_loss
        statistics[f'train/particle_constraint'] = self.particle_constraint

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      x,
                      y):
        """Perform a validation step on the loss function
        of a conservative objective model

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
        x = cont_noise(x, self.continuous_noise_std)

        # calculate the prediction error and accuracy of the model
        d = self.forward_model(x, training=False)
        mse = tf.keras.losses.mean_squared_error(y, d)
        statistics[f'validate/mse'] = mse

        # evaluate how correct the rank fo the model predictions are
        rank_corr = spearman(y[:, 0], d[:, 0])
        statistics[f'validate/rank_corr'] = rank_corr

        # calculate negative samples starting from the dataset
        x_pos = x
        x_pos = tf.where(tf.random.uniform([batch_dim] + [1 for _ in x.shape[1:]])
                         < self.negatives_fraction, x_pos, self.solution[:batch_dim])
        x_neg = self.lookahead(x_pos, self.lookahead_steps, training=False)
        if not self.lookahead_backprop:
            x_neg = tf.stop_gradient(x_neg)

        # calculate the prediction error and accuracy of the model
        d_pos = self.forward_model(
            {"dataset": x, "mix": x_pos, "solution": self.solution[:batch_dim]}
            [self.constraint_type], training=False)
        d_neg = self.forward_model(x_neg, training=False)
        conservatism = d_neg[:, 0] - d_pos[:, 0]
        statistics[f'validate/conservatism'] = conservatism
        return statistics


class TransformedMaximumLikelihood(tf.Module):

    def __init__(self,
                 forward_model,
                 forward_model_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001,
                 logger_prefix="",
                 continuous_noise_std=0.0):
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
        self.logger_prefix = logger_prefix
        self.continuous_noise_std = continuous_noise_std
        self.forward_model = forward_model
        self.forward_model_optim = \
            forward_model_optim(learning_rate=forward_model_lr)

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
        b: tf.Tensor
            bootstrap indicators shaped like [batch_size, num_oracles]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        # corrupt the inputs with noise
        x = cont_noise(x, self.continuous_noise_std)
        statistics = dict()

        with tf.GradientTape() as tape:

            # calculate the prediction error and accuracy of the model
            d = self.forward_model(x, training=True)
            nll = tf.keras.losses.mean_squared_error(y, d)

            # evaluate how correct the rank fo the model predictions are
            rank_correlation = spearman(y[:, 0], d[:, 0])

            multiplier_loss = 0.0
            last_weight = self.forward_model.trainable_variables[-1]
            if tf.shape(tf.reshape(last_weight, [-1]))[0] == 1:
                statistics[f'{self.logger_prefix}/train/tanh_multipier'] = \
                    self.forward_model.trainable_variables[-1]

            # build the total loss and weight by the bootstrap
            total_loss = tf.reduce_mean(nll) + multiplier_loss

        grads = tape.gradient(total_loss,
                              self.forward_model.trainable_variables)
        self.forward_model_optim.apply_gradients(
            zip(grads, self.forward_model.trainable_variables))

        statistics[f'{self.logger_prefix}/train/nll'] = nll
        statistics[f'{self.logger_prefix}/train/rank_corr'] = rank_correlation

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
        x = cont_noise(x, self.continuous_noise_std)
        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d = self.forward_model(x, training=False)
        nll = tf.keras.losses.mean_squared_error(y, d)

        # evaluate how correct the rank fo the model predictions are
        rank_correlation = spearman(y[:, 0], d[:, 0])

        statistics[f'{self.logger_prefix}/validate/nll'] = nll
        statistics[f'{self.logger_prefix}/validate/rank_corr'] = rank_correlation

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
            for name, tensor in self.train_step(x, y).items():
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
