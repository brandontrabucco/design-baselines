from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.gan.replay_buffer import ReplayBuffer
from design_baselines.gan.trainers import GAN
from design_baselines.gan.nets import Discriminator
from design_baselines.gan.nets import DiscreteGenerator
from design_baselines.gan.nets import ContinuousGenerator
import tensorflow as tf
import numpy as np
import os


def gan(config):
    """Train a GAN to solve a Model-Based Optimization
    problem with a hyper-parameter dict 'config'

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])

    if config['is_discrete']:

        # build a Gumbel-Softmax GAN to sample discrete outputs
        generator = DiscreteGenerator(
            task.input_shape, config['latent_size'],
            hidden=config['hidden_size'])

    else:

        # build an LS-GAN to sample continuous outputs
        generator = ContinuousGenerator(
            task.input_shape, config['latent_size'],
            hidden=config['hidden_size'])

    # build an unconditional discriminator
    discriminator = Discriminator(
        task.input_shape,
        hidden=config['hidden_size'],
        method=config['method'])

    # build a trainer with a generator and discriminator
    gan = GAN(
        generator,
        discriminator,
        ReplayBuffer(config['pool_size'], task.input_shape),
        critic_frequency=config['critic_frequency'],
        flip_frac=config['flip_frac'],
        pool_frac=config['pool_frac'],
        pool_save=config['pool_save'],
        fake_pair_frac=config['fake_pair_frac'],
        penalty_weight=config['penalty_weight'],
        generator_lr=config['generator_lr'],
        generator_beta_1=config['generator_beta_1'],
        generator_beta_2=config['generator_beta_2'],
        discriminator_lr=config['discriminator_lr'],
        discriminator_beta_1=config['discriminator_beta_1'],
        discriminator_beta_2=config['discriminator_beta_2'],
        is_discrete=config['is_discrete'],
        noise_std=config.get('noise_std', 0.0),
        keep=config.get('keep', 1.0),
        start_temp=config.get('start_temp', 5.0),
        final_temp=config.get('final_temp', 1.0))

    # create a manager for saving algorithms state to the disk
    gan_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**gan.get_saveables()),
        os.path.join(config['logging_dir'], 'gan'), 1)

    # restore tha GANS if a checkpoint exists
    gan_manager.restore_or_initialize()

    # save the initial dataset statistics for safe keeping
    x = task.x
    y = task.y

    if config.get('normalize_ys', False):

        # compute normalization statistics for the score
        mu_y = np.mean(y, axis=0, keepdims=True)
        mu_y = mu_y.astype(np.float32)
        y = y - mu_y
        st_y = np.std(y, axis=0, keepdims=True)
        st_y = np.where(np.equal(st_y, 0), 1, st_y)
        st_y = st_y.astype(np.float32)
        y = y / st_y

    else:

        # compute normalization statistics for the score
        mu_y = np.zeros_like(y[:1])
        st_y = np.ones_like(y[:1])

    if config.get('normalize_xs', False) and not config['is_discrete']:

        # compute normalization statistics for the data vectors
        mu_x = np.mean(x, axis=0, keepdims=True)
        mu_x = mu_x.astype(np.float32)
        x = x - mu_x
        st_x = np.std(x, axis=0, keepdims=True)
        st_x = np.where(np.equal(st_x, 0), 1, st_x)
        st_x = st_x.astype(np.float32)
        x = x / st_x

    else:

        # compute normalization statistics for the score
        mu_x = np.zeros_like(x[:1])
        st_x = np.ones_like(x[:1])

    # build a weighted data set using newly collected samples
    train_data, val_data = task.build(
        x=x, y=y, batch_size=config['gan_batch_size'],
        val_size=config['val_size'])

    # train the gan for several epochs
    gan.launch(
        train_data, val_data, logger, config['initial_epochs'],
        header="gan/")

    # sample designs from the GAN and evaluate them
    condition_ys = tf.tile(tf.reduce_max(
        y, keepdims=True), [config['solver_samples'], 1])

    # generate samples for exploitation
    solver_xs = generator.sample(condition_ys, temp=0.001)
    actual_ys = task.score(solver_xs * st_x + mu_x)

    # record score percentiles
    logger.record("gan/condition_ys",
                  condition_ys * st_y + mu_y,
                  0,
                  percentile=True)
    logger.record("gan/actual_ys",
                  actual_ys,
                  0,
                  percentile=True)
