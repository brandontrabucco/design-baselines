from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.lsgan.trainers import LSGAN
from design_baselines.lsgan.nets import Discriminator
from design_baselines.lsgan.nets import DiscreteGenerator
from design_baselines.lsgan.nets import ContinuousGenerator
import tensorflow as tf
import os


def least_squares_gan(config):
    """Optimize a design problem score using the algorithm MINS
    otherwise known as Model Inversion Networks

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])

    if config['is_discrete']:

        # build a Gumbel-Softmax GAN to sample discrete outputs
        exploration_generator = DiscreteGenerator(
            task.input_shape, config['latent_size'],
            hidden=config['hidden_size'])

    else:

        # build an LS-GAN to sample continuous outputs
        exploration_generator = ContinuousGenerator(
            task.input_shape, config['latent_size'],
            hidden=config['hidden_size'])

    # build the neural network GAN components
    exploration_discriminator = Discriminator(
        task.input_shape, hidden=config['hidden_size'])
    exploration_gan = LSGAN(
        exploration_generator, exploration_discriminator,
        generator_lr=config['generator_lr'],
        generator_beta_1=config['generator_beta_1'],
        generator_beta_2=config['generator_beta_2'],
        discriminator_lr=config['discriminator_lr'],
        discriminator_beta_1=config['discriminator_beta_1'],
        discriminator_beta_2=config['discriminator_beta_2'],
        is_discrete=config['is_discrete'],
        noise_std=config.get('noise_std', 0.0),
        keep=config.get('keep', 0.9),
        start_temp=config.get('temp', 5.0),
        final_temp=config.get('temp', 1.0))

    # create a manager for saving algorithms state to the disk
    exploration_gan_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**exploration_gan.get_saveables()),
        os.path.join(config['logging_dir'], 'exploration_gan'), 1)

    # restore tha GANS if a checkpoint exists
    exploration_gan_manager.restore_or_initialize()

    # save the initial dataset statistics for safe keeping
    x = task.x
    y = task.y

    # build a weighted data set using newly collected samples
    train_data, val_data = task.build(
        x=x, y=y,
        batch_size=config['gan_batch_size'],
        val_size=config['val_size'])

    # train the gan for several epochs
    exploration_gan.launch(
        train_data, val_data, logger, config['initial_epochs'],
        header="exploration/")

    # generate samples for exploration
    solver_xs = exploration_generator.sample(config['solver_samples'], temp=0.001)
    actual_ys = task.score(solver_xs)

    # record score percentiles
    logger.record("exploration/actual_ys",
                  actual_ys, 0, percentile=True)
