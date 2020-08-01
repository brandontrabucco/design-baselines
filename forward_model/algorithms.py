from forward_model.data import StaticGraphTask
from forward_model.perturbations import GradientAscent
from forward_model.trainers import Conservative
from forward_model.trainers import ModelInversion
from forward_model.trainers import BootstrapEnsemble
from forward_model.trainers import WeightedVAE
from forward_model.nets import ShallowFullyConnected
from forward_model.logger import Logger
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np
import os
import math


def conservative_mbo(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the data set and logger

    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, validate_data = task.build(include_weights=False)
    logger = Logger(config['logging_dir'])

    # create the neural net models and optimizers

    forward_model = ShallowFullyConnected(
        task.input_size, 1,
        hidden=config['hidden_size'],
        act=tfkl.ReLU,
        batch_norm=False)

    perturbation = GradientAscent(
        forward_model,
        learning_rate=config['perturbation_lr'],
        max_steps=config['perturbation_steps'])

    trainer = Conservative(
        forward_model, perturbation,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        target_conservative_gap=config['target_conservative_gap'],
        initial_alpha=config['initial_alpha'],
        alpha_optim=tf.keras.optimizers.Adam,
        alpha_lr=config['alpha_lr'])

    # train and validate the neural network models

    for e in range(config['epochs']):
        for name, loss in trainer.train(train_data).items():
            logger.record(name, loss, e)
        for name, loss in trainer.validate(validate_data).items():
            logger.record(name, loss, e)

    # perform gradient based optimization to find x

    indices = tf.math.top_k(task.y[:, 0], k=config['solver_samples'])[1]
    solution = tf.gather(task.x, indices, axis=0)
    score = task.score(solution)
    prediction = forward_model(solution)

    logger.record(
        "score", score, tf.cast(0, tf.int64))
    logger.record(
        "prediction", prediction, tf.cast(0, tf.int64))
    logger.record(
        "best/score", score[0], tf.cast(0, tf.int64))
    logger.record(
        "best/prediction", prediction[0], tf.cast(0, tf.int64))

    for i in range(1, config['solver_steps'] + 1):

        with tf.GradientTape() as tape:
            tape.watch(solution)
            score = forward_model(solution)
        grads = tape.gradient(score, solution)
        solution = solution + config['solver_lr'] * grads

        gradient_norm = tf.linalg.norm(grads, axis=1)
        score = task.score(solution)
        prediction = forward_model(solution)

        logger.record(
            "gradient_norm", gradient_norm, tf.cast(i, tf.int64))
        logger.record(
            "score", score, tf.cast(i, tf.int64))
        logger.record(
            "prediction", prediction, tf.cast(i, tf.int64))
        logger.record(
            "best/gradient_norm", gradient_norm[0], tf.cast(i, tf.int64))
        logger.record(
            "best/score", score[0], tf.cast(i, tf.int64))
        logger.record(
            "best/prediction", prediction[0], tf.cast(i, tf.int64))


def cbas(config):
    """Optimization code for Conditioning By Adaptive Sampling
    otherwise known as CbAS

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the data set and logger

    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, validate_data = task.build(bootstraps=config['bootstraps'])
    logger = Logger(config['logging_dir'])

    # create the neural net models and optimizers

    oracles = [ShallowFullyConnected(
        task.input_size, 2,
        hidden=config['oracle_hidden_size'],
        act=tfkl.ReLU,
        batch_norm=False) for b in range(config['bootstraps'])]

    ensemble = BootstrapEnsemble(
        oracles,
        oracle_optim=tf.keras.optimizers.Adam,
        oracle_lr=config['oracle_lr'])

    # train and validate the ensemble

    for e in range(config['oracle_epochs']):

        for name, loss in ensemble.train(train_data).items():
            logger.record(name, loss, e)

        for name, loss in ensemble.validate(validate_data).items():
            logger.record(name, loss, e)

    # create the neural net vae components and their optimizer

    p_encoder = ShallowFullyConnected(
        task.input_size, config['latent_size'] * 2,
        hidden=config['vae_hidden_size'],
        act=tfkl.ReLU,
        batch_norm=False)

    p_decoder = ShallowFullyConnected(
        config['latent_size'], task.input_size * 2,
        hidden=config['vae_hidden_size'],
        act=tfkl.ReLU,
        batch_norm=False)

    p_vae = WeightedVAE(
        p_encoder,
        p_decoder,
        vae_optim=tf.keras.optimizers.Adam,
        vae_lr=config['vae_lr'],
        vae_beta=config['vae_beta'])

    train_data, validate_data = task.build(
        importance_weights=np.ones_like(task.y))

    # train and validate the p_vae

    e = 0
    for _ in range(config['offline_vae_epochs']):

        for name, loss in p_vae.train(train_data).items():
            logger.record(name, loss, e)

        for name, loss in p_vae.validate(validate_data).items():
            logger.record(name, loss, e)

        e += 1

    # create the neural net vae components and their optimizer

    q_encoder = ShallowFullyConnected(
        task.input_size, config['latent_size'] * 2,
        hidden=config['vae_hidden_size'],
        act=tfkl.ReLU,
        batch_norm=False)
    q_encoder.set_weights(p_encoder.get_weights())

    q_decoder = ShallowFullyConnected(
        config['latent_size'], task.input_size * 2,
        hidden=config['vae_hidden_size'],
        act=tfkl.ReLU,
        batch_norm=False)
    q_decoder.set_weights(p_decoder.get_weights())

    q_vae = WeightedVAE(
        q_encoder,
        q_decoder,
        vae_optim=tf.keras.optimizers.Adam,
        vae_lr=config['vae_lr'],
        vae_beta=config['vae_beta'])

    @tf.function(experimental_relax_shapes=True)
    def generate_data(dataset_size,
                      batch_size):
        """A function for generating a data set of samples of a particular
        size using two adaptively sampled vaes

        Args:

        dataset_size: int
            the number of samples to include in the final data set
        batch_size: int
            the number of samples to generate all at once using the vae

        Returns:

        xs: tf.Tensor
            the dataset x values sampled from the vaes
        ys: tf.Tensor
            the dataset y values predicted by the ensemble
        ws: tf.Tensor
            the dataset importance weights calculated using the vaes
        """

        xs = []
        ys = []
        ws = []

        num_steps = math.ceil(dataset_size / batch_size)
        for j in range(num_steps):

            num_samples = min(dataset_size - batch_size * j,
                              batch_size)

            z = tf.random.normal([
                num_samples, config['latent_size']])

            mu, log_std = tf.split(
                q_decoder(z, training=False), 2, axis=-1)
            q_dx = tfpd.MultivariateNormalDiag(
                loc=mu, scale_diag=tf.math.softplus(log_std))

            mu, log_std = tf.split(
                p_decoder(z, training=False), 2, axis=-1)
            p_dx = tfpd.MultivariateNormalDiag(
                loc=mu, scale_diag=tf.math.softplus(log_std))

            x = q_dx.sample()
            xs.append(x)

            y = ensemble.get_distribution(x).mean()
            ys.append(y)

            ws.append(tf.math.exp(
                p_dx.log_prob(x) - q_dx.log_prob(x))[:, tf.newaxis])

        return tf.concat(xs, axis=0), \
               tf.concat(ys, axis=0), \
               tf.concat(ws, axis=0)

    @tf.function(experimental_relax_shapes=True)
    def reweight_by_s(x,
                      y,
                      percentile,
                      batch_size):
        """A function for generating the probability that samples x
        satisfy a specification s given by y

        Args:

        x: tf.Tensor
            the dataset x values sampled from the vaes
        y: tf.Tensor
            the dataset y values predicted by the ensemble
        percentile: int
            the percentile to use when calculating importance weights
        batch_size: int
            the number of samples to generate all at once using the vae

        Returns:

        ws: tf.Tensor
            the dataset importance weights calculated using the ensemble
        """

        ws = []
        gamma = tfp.stats.percentile(y, percentile)

        num_steps = math.ceil(x.shape[0] / batch_size)
        for j in range(num_steps):

            num_samples = min(x.shape[0] - batch_size * j,
                              batch_size)

            d = ensemble.get_distribution(
                x[j * batch_size:j * batch_size + num_samples])

            weight = 1.0 - d.cdf(tf.fill([num_samples, 1], gamma))
            ws.append(weight)

        return tf.concat(ws, axis=0)

    # train and validate the q_vae

    for i in range(config['iterations']):

        x, y, w = generate_data(config['online_size'],
                                config['task_kwargs']['batch_size'])

        logger.record("score", task.score(x[:config['solver_samples']]), i)

        w = w * reweight_by_s(x, y, config['percentile'],
                              config['task_kwargs']['batch_size'])

        train_data, validate_data = task.build(
            x=x.numpy(), y=y.numpy(), importance_weights=w.numpy())

        for _ in range(config['online_vae_epochs']):

            for name, loss in q_vae.train(train_data).items():
                logger.record(name, loss, e)

            for name, loss in q_vae.validate(validate_data).items():
                logger.record(name, loss, e)

            e += 1

    x = generate_data(config['solver_samples'],
                      config['task_kwargs']['batch_size'])[0]

    logger.record("score", task.score(x), config['iterations'])


def model_inversion(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the data set and logger

    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, validate_data = task.build(include_weights=True)
    logger = Logger(config['logging_dir'])

    # create the neural net models and optimizers

    latent_size = config['latent_size']
    hdim = config['hidden_size']

    generator = tf.keras.Sequential([
        tfkl.Dense(hdim, use_bias=True, input_shape=(latent_size + 1,)),
        tfkl.ReLU(),
        tfkl.Dense(hdim, use_bias=True),
        tfkl.ReLU(),
        tfkl.Dense(task.input_size, use_bias=True)])

    discriminator = tf.keras.Sequential([
        tfkl.Dense(hdim, use_bias=True, input_shape=(task.input_size + 1,)),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.Dense(hdim, use_bias=True),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.Dense(1, use_bias=True)])

    trainer = ModelInversion(
        generator,
        discriminator,
        latent_size=latent_size,
        optim=tf.keras.optimizers.Adam,
        learning_rate=config['model_lr'],
        beta_1=config['beta_1'],
        beta_2=config['beta_2'])

    # train and validate the neural network models

    for e in range(config['epochs']):
        step = tf.cast(e, tf.int64)
        for name, loss in trainer.train(train_data).items():
            logger.record(name, loss, step)
        for name, loss in trainer.validate(validate_data).items():
            logger.record(name, loss, step)

    # sample for the best y using the generator

    max_y = tf.tile(tf.reduce_max(
        task.y, keepdims=True), [config['solver_samples'], 1])
    max_x = generator(tf.concat([
        tf.random.normal([max_y.shape[0], latent_size]), max_y], 1))

    logger.record(
        "score", task.score(max_x), tf.cast(0, tf.int64))
    logger.record(
        "prediction", max_y, tf.cast(0, tf.int64))
