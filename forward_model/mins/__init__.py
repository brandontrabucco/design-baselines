from forward_model.data import StaticGraphTask
from forward_model.logger import Logger
from forward_model.mins.trainers import Ensemble
from forward_model.mins.trainers import WeightedGAN
from forward_model.mins.nets import ForwardModel
from forward_model.mins.nets import Discriminator
from forward_model.mins.nets import DiscreteGenerator
from forward_model.mins.nets import ContinuousGenerator
from forward_model.mins.utils import get_weights
from forward_model.mins.utils import get_synthetic_data
import tensorflow as tf
import os


def model_inversion(config):
    """Optimize a design problem score using the algorithm MINS
    otherwise known as Model Inversion Networks

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    logger = Logger(config['logging_dir'])

    # create the training task and logger
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, val_data = task.build(bootstraps=config['bootstraps'],
                                      batch_size=config['ensemble_batch_size'],
                                      val_size=config['val_size'])

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        task.input_shape,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for b in range(config['bootstraps'])]

    # create a trainer for a forward model with a conservative objective
    ensemble = Ensemble(forward_models,
                        forward_model_optim=tf.keras.optimizers.Adam,
                        forward_model_lr=config['ensemble_lr'])

    # create a manager for saving algorithms state to the disk
    ensemble_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**ensemble.get_saveables()),
        directory=os.path.join(config['logging_dir'], 'ensemble'),
        max_to_keep=1)

    # train the model for an additional number of epochs
    ensemble_manager.restore_or_initialize()
    ensemble.launch(train_data,
                    val_data,
                    logger,
                    config['ensemble_epochs'])

    # pick the right architecture based on the task
    Generator = DiscreteGenerator if config[
        'is_discrete'] else ContinuousGenerator

    # build the neural network GAN components
    exploration_generator = Generator(
        task.input_shape, config['latent_size'], hidden=config['hidden_size'])
    exploration_discriminator = Discriminator(
        task.input_shape, hidden=config['hidden_size'])
    exploration_gan = WeightedGAN(
        exploration_generator, exploration_discriminator,
        generator_lr=config['generator_lr'],
        generator_beta_1=config['generator_beta_1'],
        generator_beta_2=config['generator_beta_2'],
        discriminator_lr=config['discriminator_lr'],
        discriminator_beta_1=config['discriminator_beta_1'],
        discriminator_beta_2=config['discriminator_beta_2'])

    # create a manager for saving algorithms state to the disk
    exploration_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**exploration_gan.get_saveables()),
        os.path.join(config['logging_dir'], 'exploration_gan'), 1)
    exploration_manager.restore_or_initialize()

    # build the neural network GAN components
    exploitation_generator = Generator(
        task.input_shape, config['latent_size'], hidden=config['hidden_size'])
    exploitation_discriminator = Discriminator(
        task.input_shape, hidden=config['hidden_size'])
    exploitation_gan = WeightedGAN(
        exploitation_generator, exploitation_discriminator,
        generator_lr=config['generator_lr'],
        generator_beta_1=config['generator_beta_1'],
        generator_beta_2=config['generator_beta_2'],
        discriminator_lr=config['discriminator_lr'],
        discriminator_beta_1=config['discriminator_beta_1'],
        discriminator_beta_2=config['discriminator_beta_2'])

    # create a manager for saving algorithms state to the disk
    exploitation_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**exploitation_gan.get_saveables()),
        os.path.join(config['logging_dir'], 'exploitation_gan'), 1)
    exploitation_manager.restore_or_initialize()

    # save the initial dataset statistics for safe keeping
    x = task.x
    y = task.y

    # train the gan using an importance sampled data set
    for iteration in range(config['iterations']):

        # generate synthetic x paired with high performing scores
        tilde_x, tilde_y = get_synthetic_data(
            x, y,
            exploration_samples=config['exploration_samples'],
            exploration_rate=config['exploration_rate'],
            exploration_noise_std=config['exploration_noise_std'])

        # build a weighted data set using newly collected samples
        train_data, val_data = task.build(
            x=tilde_x.numpy(), y=tilde_y.numpy(),
            importance_weights=get_weights(tilde_y.numpy()),
            batch_size=config['gan_batch_size'],
            val_size=config['val_size'])

        # train the gan for several epochs
        exploration_gan.launch(
            train_data, val_data, logger, config['epochs_per_iteration'],
            start_epoch=config['epochs_per_iteration'] * iteration,
            header="exploration/")

        # sample designs from the GAN and evaluate them
        conditioned_ys = tf.tile(tf.reduce_max(
            tilde_y, keepdims=True), [config['thompson_samples'], 1])

        # generate samples and evaluate using an ensemble
        solver_xs = exploration_generator.sample(conditioned_ys)
        if config['fully_offline']:
            actual_ys = ensemble.get_distribution(solver_xs).mean()
        else:
            actual_ys = task.score(solver_xs)

        # record score percentiles
        logger.record("exploration/conditioned_ys", conditioned_ys, iteration)
        logger.record("exploration/actual_ys", actual_ys, iteration)

        # concatenate newly paired samples with the existing data set
        x = tf.concat([x, solver_xs], 0)
        y = tf.concat([y, actual_ys], 0)

        # build a weighted data set using newly collected samples
        train_data, val_data = task.build(
            x=x.numpy(), y=y.numpy(),
            importance_weights=get_weights(y.numpy()),
            batch_size=config['gan_batch_size'],
            val_size=config['val_size'])

        # train the gan for several epochs
        exploitation_gan.launch(
            train_data, val_data, logger, config['epochs_per_iteration'],
            start_epoch=config['epochs_per_iteration'] * iteration,
            header="exploitation/")

        # sample designs from the GAN and evaluate them
        conditioned_ys = tf.tile(tf.reduce_max(
            y, keepdims=True), [config['solver_samples'], 1])

        # generate samples and evaluate using the task
        solver_xs = exploration_generator.sample(conditioned_ys)
        actual_ys = task.score(solver_xs)

        # record score percentiles
        logger.record("exploitation/conditioned_ys", conditioned_ys, iteration)
        logger.record("exploitation/actual_ys", actual_ys, iteration)

    # save every model to the disk
    ensemble_manager.save()
    exploration_manager.save()
    exploitation_manager.save()
