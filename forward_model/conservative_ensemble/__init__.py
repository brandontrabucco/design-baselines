from forward_model.data import StaticGraphTask
from forward_model.logger import Logger
from forward_model.conservative_ensemble.trainers import ConservativeEnsemble
from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np
import os


class FullyConnected(tf.keras.Sequential):
    """A Fully Connected Network with 3 trainable layers"""

    distribution = tfpd.MultivariateNormalDiag

    def __init__(self,
                 inp_size,
                 out_size,
                 hidden=2048,
                 initial_max_std=1.5,
                 initial_min_std=0.5,
                 act=tfkl.ReLU):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        inp_size: int
            the size of the input vector of this network
        out_size: int
            the size of the output vector of this network
        hidden: int
            the global hidden size of the network
        act: function
            a function that returns an activation function such as tfkl.ReLU
        """

        self.max_logstd = tf.Variable(tf.fill([1, out_size], np.log(
            initial_max_std).astype(np.float32)), trainable=True)
        self.min_logstd = tf.Variable(tf.fill([1, out_size], np.log(
            initial_min_std).astype(np.float32)), trainable=True)

        super(FullyConnected, self).__init__([
            tfkl.Dense(hidden, input_shape=(inp_size,)),
            act(),
            tfkl.Dense(hidden),
            act(),
            tfkl.Dense(out_size * 2)])

    def get_parameters(self, inputs, **kwargs):
        """Return a dictionary of parameters for a particular distribution
        family such as the mean and variance of a gaussian

        Args:

        inputs: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        parameters: dict
            a dictionary that contains 'loc' and 'scale_diag' keys
        """

        prediction = super(FullyConnected, self).__call__(inputs, **kwargs)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale_diag": tf.exp(logstd)}

    def get_distribution(self, inputs, **kwargs):
        """Return a distribution over the outputs of this model, for example
        a Multivariate Gaussian Distribution

        Args:

        inputs: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        distribution: tfp.distribution.Distribution
            a tensorflow probability distribution over outputs of the model
        """

        return self.distribution(**self.get_parameters(inputs, **kwargs))


def conservative_ensemble(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, validate_data = task.build(bootstraps=config['bootstraps'])
    logger = Logger(config['logging_dir'])

    # make several keras neural networks with two hidden layers
    forward_models = [FullyConnected(
        task.input_size,
        1,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'],
        act=tfkl.ReLU) for b in range(config['bootstraps'])]

    # create a trainer for a forward model with a conservative objective
    trainer = ConservativeEnsemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        target_conservative_gap=config['target_conservative_gap'],
        initial_alpha=config['initial_alpha'],
        alpha_optim=tf.keras.optimizers.Adam,
        alpha_lr=config['alpha_lr'],
        perturbation_lr=config['perturbation_lr'],
        perturbation_steps=config['perturbation_steps'])

    # create a manager for saving algorithms state to the disk
    manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**trainer.get_saveables()),
        directory=os.path.join(config['logging_dir'], 'ckpt'),
        max_to_keep=1)

    # train the model for an additional number of epochs
    manager.restore_or_initialize()
    trainer.launch(train_data, validate_data, logger, config['epochs'])
    manager.save()

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(task.y[:, 0], k=config['solver_samples'])[1]

    # evaluate the initial design using the oracle and the forward model
    solution = tf.gather(task.x, indices, axis=0)
    score = task.score(solution)
    prediction = trainer.get_distribution(solution).mean()

    # record the prediction and score to the logger
    logger.record("score", score, 0)
    logger.record("prediction", prediction, 0)

    # and keep track of the best design sampled so far
    best_design = None
    best_score = None

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):

        # back propagate through the forward model
        with tf.GradientTape() as tape:
            tape.watch(solution)
            score = trainer.get_distribution(solution).mean()
        grads = tape.gradient(score, solution)
        solution = solution + config['solver_lr'] * grads

        # evaluate the design using the oracle and the forward model
        gradient_norm = tf.linalg.norm(grads, axis=1)
        score = task.score(solution)
        prediction = trainer.get_distribution(solution).mean()

        # record the prediction and score to the logger
        logger.record("gradient_norm", gradient_norm, i)
        logger.record("score", score, i)
        logger.record("prediction", prediction, i)

        # update the best design every iteration
        idx = np.argmax(score.numpy())
        if best_design is None or score[idx] > best_score:
            best_score = score[idx]
            best_design = solution[idx]

    # save the best design to the disk
    np.save(os.path.join(
        config['logging_dir'], 'score.npy'), best_score)
    np.save(os.path.join(
        config['logging_dir'], 'design.npy'), best_design)


def second_model_predictions(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, validate_data = task.build(bootstraps=2)
    logger = Logger(config['logging_dir'])

    # make several keras neural networks with two hidden layers
    forward_models = [FullyConnected(
        task.input_size,
        1,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'],
        act=tfkl.ReLU) for b in range(2)]

    # create a trainer for a forward model with a conservative objective
    trainer = ConservativeEnsemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        target_conservative_gap=config['target_conservative_gap'],
        initial_alpha=config['initial_alpha'],
        alpha_optim=tf.keras.optimizers.Adam,
        alpha_lr=config['alpha_lr'],
        perturbation_lr=config['perturbation_lr'],
        perturbation_steps=config['perturbation_steps'])

    # create a manager for saving algorithms state to the disk
    manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**trainer.get_saveables()),
        directory=os.path.join(config['logging_dir'], 'ckpt'),
        max_to_keep=1)

    # train the model for an additional number of epochs
    manager.restore_or_initialize()
    trainer.launch(train_data, validate_data, logger, config['epochs'])
    manager.save()

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(task.y[:, 0], k=config['solver_samples'])[1]

    # evaluate the initial design using the oracle and the forward model
    solution = tf.gather(task.x, indices, axis=0)
    score = task.score(solution)
    prediction0 = forward_models[0].get_distribution(solution).mean()
    prediction1 = forward_models[1].get_distribution(solution).mean()

    # record the prediction and score to the logger
    logger.record("score", score, 0)
    logger.record("model_0/prediction", prediction0, 0)
    logger.record("model_1/prediction", prediction1, 0)

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):

        # back propagate through the forward model
        with tf.GradientTape() as tape:
            tape.watch(solution)
            score = forward_models[0].get_distribution(solution).mean()
        grads = tape.gradient(score, solution)
        solution = solution + config['solver_lr'] * grads

        # evaluate the design using the oracle and the forward model
        gradient_norm = tf.linalg.norm(grads, axis=1)
        score = task.score(solution)
        prediction0 = forward_models[0].get_distribution(solution).mean()
        prediction1 = forward_models[1].get_distribution(solution).mean()

        # record the prediction and score to the logger
        logger.record("gradient_norm", gradient_norm, i)
        logger.record("score", score, i)
        logger.record("model_0/prediction", prediction0, i)
        logger.record("model_1/prediction", prediction1, i)
