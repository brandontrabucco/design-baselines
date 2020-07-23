from forward_model.data import PolicyWeightsDataset
from forward_model.perturbations import PGD
from forward_model.trainers import Conservative
from forward_model.logger import Logger
from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import os


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
        num_steps=config['perturbation_steps'],
        optim=tf.keras.optimizers.SGD,
        learning_rate=config['perturbation_lr'])

    trainer = Conservative(
        forward_model,
        perturbation_distribution,
        conservative_weight=config['conservative_weight'],
        optim=tf.keras.optimizers.Adam,
        learning_rate=config['forward_model_lr'])

    # train and validate the neural network models

    for e in range(config['epochs']):

        logger.record(
            "train", trainer.train(
                data.train), tf.cast(e, tf.int64))

        logger.record(
            "validate", trainer.validate(
                data.val), tf.cast(e, tf.int64))

    # save the trained forward model

    forward_model.save(
        os.path.join(logging_dir, "forward-model"))

    # fit an initialization distribution

    loc = tf.math.reduce_mean(data.x, axis=0, keepdims=True)
    scale_diag = tf.math.reduce_std(data.x - loc, axis=0)

    d = tfpd.MultivariateNormalDiag(
        loc=loc[0], scale_diag=scale_diag)

    x_var = tf.Variable(d.sample(
        sample_shape=config['solver_samples']))
    optim = tf.keras.optimizers.SGD(
        learning_rate=config['solver_lr'])

    # perform gradient based optimization to find x

    logger.record(
        "score", data.score(
            x_var), tf.cast(0, tf.int64))

    logger.record(
        "prediction", forward_model(
            x_var), tf.cast(0, tf.int64))

    for i in range(1, config['solver_steps'] + 1):

        optim.minimize(
            lambda: -forward_model(x_var), [x_var])

        logger.record(
            "score", data.score(
                x_var), tf.cast(i, tf.int64))

        logger.record(
            "prediction", forward_model(
                x_var), tf.cast(i, tf.int64))
