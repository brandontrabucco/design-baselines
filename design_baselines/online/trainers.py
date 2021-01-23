from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class Ensemble(tf.Module):

    def __init__(self,
                 forward_models,
                 forward_model_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001):
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

        for i in range(self.bootstraps):
            fm = self.forward_models[i]
            fm_optim = self.forward_model_optims[i]

            with tf.GradientTape(persistent=True) as tape:

                # calculate the prediction error and accuracy of the model
                d = fm.get_distribution(x, training=True)
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

        for i in range(self.bootstraps):
            fm = self.forward_models[i]

            # calculate the prediction error and accuracy of the model
            d = fm.get_distribution(x, training=False)
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
               start_epoch=0):
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
                logger.record(name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(name, loss, e)

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
                 forward_model_opt=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001,
                 initial_alpha=1.0,
                 initial_beta=0.5,
                 alpha_opt=tf.keras.optimizers.Adam,
                 alpha_lr=0.05,
                 target_conservatism=1.0,
                 negatives_fraction=0.5,
                 lookahead_steps=50,
                 lookahead_backprop=True,
                 solver_conservatism=0.0,
                 solver_lr=0.01,
                 solver_interval=10,
                 solver_warmup=500,
                 solver_steps=1,
                 is_discrete=False,
                 constraint_type="mix",
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
        self.beta = initial_beta
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
        self.solver_conservatism = solver_conservatism

        # extra parameters for controlling data noise
        self.is_discrete = is_discrete
        self.continuous_noise_std = continuous_noise_std
        self.discrete_smoothing = discrete_smoothing
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
            lambda xt: True, gradient_step, (x,),
            maximum_iterations=steps)[0]

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
        x = soft_noise(x, self.discrete_smoothing) \
            if self.is_discrete else \
            cont_noise(x, self.continuous_noise_std)

        with tf.GradientTape(persistent=True) as tape:

            # calculate the prediction error and accuracy of the model
            d = self.forward_model.get_distribution(x, training=True)
            nll = -d.log_prob(y)
            statistics[f'train/nll'] = nll

            # evaluate how correct the rank fo the model predictions are
            rank_corr = spearman(y[:, 0], d.mean()[:, 0])
            statistics[f'train/rank_corr'] = rank_corr

            # calculate negative samples starting from the dataset
            x_pos = tf.math.log(x) if self.is_discrete else x
            x_pos = tf.where(tf.random.uniform([batch_dim] + [1 for _ in x.shape[1:]])
                             < self.negatives_fraction, x_pos, self.solution[:batch_dim])
            x_neg = self.lookahead(x_pos, self.lookahead_steps, training=False)
            if not self.lookahead_backprop:
                x_neg = tf.stop_gradient(x_neg)

            # calculate the prediction error and accuracy of the model
            x_neg = tf.math.softmax(x_neg) if self.is_discrete else x_neg
            d_pos = self.forward_model.get_distribution(
                {"dataset": x, "mix": x_pos, "solution": self.solution[:batch_dim]}
                [self.constraint_type], training=False)
            d_neg = self.forward_model.get_distribution(x_neg, training=False)
            conservatism = d_neg.mean()[:, 0] - d_pos.mean()[:, 0]
            statistics[f'train/conservatism'] = conservatism

            # build a lagrangian for dual descent
            alpha_loss = (self.alpha * self.target_conservatism -
                          self.alpha * conservatism)
            statistics[f'train/alpha'] = self.alpha

            # loss that combines maximum likelihood with a constraint
            model_loss = nll + self.alpha * conservatism
            total_loss = tf.reduce_mean(model_loss)
            alpha_loss = tf.reduce_mean(alpha_loss)

        if self.particle_loss is None:
            self.particle_loss = tf.Variable(tf.zeros_like(conservatism))
        if self.particle_constraint is None:
            self.particle_constraint = tf.Variable(tf.zeros_like(conservatism))

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

            with tf.GradientTape(persistent=True) as tape:

                # calculate the predicted score of the current solution
                current_score = self.forward_model.get_distribution(
                    tf.math.softmax(self.solution)
                    if self.is_discrete else self.solution, training=False).mean()[:, 0]

                # look into the future and evaluate future solutions
                future = self.lookahead(
                    self.solution, self.solver_steps, training=False)
                future_score = self.forward_model.get_distribution(
                    tf.math.softmax(future)
                    if self.is_discrete else future, training=False).mean()[:, 0]

                # evaluate the conservatism of the current solution
                particle_loss = future_score - self.beta * current_score

                # if optimizer conservatism passes threshold stop optimizing
                self.particle_loss.assign(particle_loss)
                self.particle_constraint.assign(future_score - current_score)

                # build a lagrangian for dual descent
                beta_loss = -particle_loss

            # calculate an update to the particles
            update = (self.solution - self.solver_lr *
                      tape.gradient(particle_loss, self.solution))

            # only update solutions that are not frozen
            self.solution.assign(
                tf.where(self.done, self.solution, update))
        statistics[f'train/beta'] = self.beta
        statistics[f'train/done'] = tf.cast(self.done, tf.float32)
        statistics[f'train/particle_loss'] = self.particle_loss
        statistics[f'train/particle_constraint'] = self.particle_constraint

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
        x = soft_noise(x, self.discrete_smoothing) \
            if self.is_discrete else \
            cont_noise(x, self.continuous_noise_std)

        # calculate the prediction error and accuracy of the model
        d = self.forward_model.get_distribution(x, training=False)
        nll = -d.log_prob(y)
        statistics[f'validate/nll'] = nll

        # evaluate how correct the rank fo the model predictions are
        rank_corr = spearman(y[:, 0], d.mean()[:, 0])
        statistics[f'validate/rank_corr'] = rank_corr

        # calculate negative samples starting from the dataset
        x_pos = tf.math.log(x) if self.is_discrete else x
        x_pos = tf.where(tf.random.uniform([batch_dim] + [1 for _ in x.shape[1:]])
                         < self.negatives_fraction, x_pos, self.solution[:batch_dim])
        x_neg = self.lookahead(x_pos, self.lookahead_steps, training=False)
        if not self.lookahead_backprop:
            x_neg = tf.stop_gradient(x_neg)

        # calculate the prediction error and accuracy of the model
        x_neg = tf.math.softmax(x_neg) if self.is_discrete else x_neg
        d_pos = self.forward_model.get_distribution(
            {"dataset": x, "mix": x_pos, "solution": self.solution[:batch_dim]}
            [self.constraint_type], training=False)
        d_neg = self.forward_model.get_distribution(x_neg, training=False)
        conservatism = d_neg.mean()[:, 0] - d_pos.mean()[:, 0]
        statistics[f'validate/conservatism'] = conservatism
        return statistics
