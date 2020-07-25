from forward_model.data import PolicyWeightsDataset
from forward_model.perturbations import PGD
from forward_model.trainers import Conservative
from forward_model.trainers import ModelInversion
from forward_model.logger import Logger
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import os
import math


def conservative_mbo(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the dataset and logger

    logging_dir = config['logging_dir']
    logger = Logger(logging_dir)

    data = PolicyWeightsDataset(
        obs_dim=11,
        action_dim=3,
        hidden_dim=64,
        val_size=200,
        batch_size=config['batch_size'],
        env_name='Hopper-v2',
        seed=0,
        x_file='hopper_controller_X.txt',
        y_file='hopper_controller_y.txt')

    # create the neural net models and optimizers

    hdim = config['hidden_size']

    forward_model = tf.keras.Sequential([
        tfkl.Dense(hdim, use_bias=True, input_shape=data.input_shape),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.Dense(hdim, use_bias=True),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.Dense(1, use_bias=True)])

    perturbation_distribution = PGD(
        forward_model,
        optim=tf.keras.optimizers.SGD,
        learning_rate=config['perturbation_lr'])

    trainer = Conservative(
        forward_model,
        perturbation_distribution,
        conservative_weight=config['conservative_weight'],
        optim=tf.keras.optimizers.Adam,
        learning_rate=config['forward_model_lr'])

    # train and validate the neural network models

    epochs = config['epochs']
    perturbation_steps = config['perturbation_steps']

    def schedule(epoch):
        return int(math.ceil(
            perturbation_steps * (epoch + 1) / epochs))

    for e in range(epochs):
        step = tf.cast(e, tf.int64)
        for name, loss in trainer.train(
                data.train, num_steps=schedule(e)).items():
            logger.record(name, loss, step)
        for name, loss in trainer.validate(data.validate).items():
            logger.record(name, loss, step)

    # save the trained forward model

    forward_model.save(
        os.path.join(logging_dir, "forward-model"))

    # fit an initialization distribution

    values, indices = tf.math.top_k(
        data.y[:, 0], k=config['solver_samples'])

    x_var = tf.Variable(
        tf.gather(data.x, indices, axis=0))
    optim = tf.keras.optimizers.SGD(
        learning_rate=config['solver_lr'])

    # perform gradient based optimization to find x

    logger.record(
        "score", data.score(
            x_var), tf.cast(0, tf.int64))

    logger.record(
        "prediction", forward_model(
            x_var), tf.cast(0, tf.int64))

    logger.record(
        "best/score", data.score(
            x_var)[0], tf.cast(0, tf.int64))

    logger.record(
        "best/prediction", forward_model(
            x_var)[0], tf.cast(0, tf.int64))

    for i in range(1, config['solver_steps'] + 1):

        with tf.GradientTape() as gradient_tape:
            loss = -tf.reduce_sum(forward_model(x_var))
        grads = gradient_tape.gradient(loss, x_var)
        optim.apply_gradients([[grads, x_var]])

        logger.record(
            "gradient_norm", tf.linalg.norm(
                grads, axis=1), tf.cast(i, tf.int64))

        logger.record(
            "score", data.score(
                x_var), tf.cast(i, tf.int64))

        logger.record(
            "prediction", forward_model(
                x_var), tf.cast(i, tf.int64))

        logger.record(
            "best/gradient_norm", tf.linalg.norm(
                grads[0], axis=1), tf.cast(i, tf.int64))

        logger.record(
            "best/score", data.score(
                x_var)[0], tf.cast(i, tf.int64))

        logger.record(
            "best/prediction", forward_model(
                x_var)[0], tf.cast(i, tf.int64))


def model_inversion(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the dataset and logger

    logging_dir = config['logging_dir']
    logger = Logger(logging_dir)

    data = PolicyWeightsDataset(
        obs_dim=11,
        action_dim=3,
        hidden_dim=64,
        val_size=200,
        batch_size=config['batch_size'],
        env_name='Hopper-v2',
        seed=config['seed'],
        x_file='hopper_controller_X.txt',
        y_file='hopper_controller_y.txt',
        include_weights=True)

    # create the neural net models and optimizers

    latent_size = config['latent_size']
    hdim = config['hidden_size']

    generator = tf.keras.Sequential([
        tfkl.Dense(hdim, use_bias=True, input_shape=(latent_size + 1,)),
        tfkl.ReLU(),
        tfkl.Dense(hdim, use_bias=True),
        tfkl.ReLU(),
        tfkl.Dense(data.input_size, use_bias=True)])

    discriminator = tf.keras.Sequential([
        tfkl.Dense(hdim, use_bias=True, input_shape=(data.input_size + 1,)),
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
        for name, loss in trainer.train(data.train).items():
            logger.record(name, loss, step)
        for name, loss in trainer.validate(data.validate).items():
            logger.record(name, loss, step)

    # save the trained forward model

    generator.save(
        os.path.join(logging_dir, "generator"))
    discriminator.save(
        os.path.join(logging_dir, "discriminator"))

    # sample for the best y using the generator

    max_y = tf.tile(tf.reduce_max(
        data.y, keepdims=True), [config['solver_samples'], 1])
    max_x = generator(tf.concat([
        tf.random.normal([max_y.shape[0], latent_size]), max_y], 1))

    logger.record(
        "score", data.score(max_x), tf.cast(0, tf.int64))
    logger.record(
        "prediction", max_y, tf.cast(0, tf.int64))
