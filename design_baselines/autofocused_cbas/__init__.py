from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.autofocused_cbas.trainers import Ensemble
from design_baselines.autofocused_cbas.trainers import WeightedVAE
from design_baselines.autofocused_cbas.trainers import CBAS
from design_baselines.autofocused_cbas.nets import ForwardModel
from design_baselines.autofocused_cbas.nets import Encoder
from design_baselines.autofocused_cbas.nets import DiscreteDecoder
from design_baselines.autofocused_cbas.nets import ContinuousDecoder
import tensorflow as tf
import numpy as np
import os


def autofocused_cbas(config):
    """Optimize a design problem score using the algorithm CBAS
    otherwise known as Conditioning by Adaptive Sampling

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    logger = Logger(config['logging_dir'])

    # create the training task and logger
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, val_data = task.build(
        bootstraps=config['bootstraps'],
        importance_weights=np.ones_like(task.y),
        batch_size=config['oracle_batch_size'],
        val_size=config['val_size'])

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        task.input_shape,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for _ in range(config['bootstraps'])]

    # create a trainer for a forward model with a conservative objective
    oracle = Ensemble(forward_models,
                      forward_model_optim=tf.keras.optimizers.Adam,
                      forward_model_lr=config['oracle_lr'])

    # create a manager for saving algorithms state to the disk
    oracle_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**oracle.get_saveables()),
        os.path.join(config['logging_dir'], 'oracle'), 1)

    # train the model for an additional number of epochs
    oracle_manager.restore_or_initialize()
    oracle.launch(train_data,
                  val_data,
                  logger,
                  config['oracle_epochs'])

    # determine which arcitecture for the decoder to use
    decoder = DiscreteDecoder \
        if config['is_discrete'] else ContinuousDecoder

    # build the encoder and decoder distribution and the p model
    p_encoder = Encoder(task.input_shape,
                        config['latent_size'],
                        hidden=config['hidden_size'])
    p_decoder = decoder(task.input_shape,
                        config['latent_size'],
                        hidden=config['hidden_size'])
    p_vae = WeightedVAE(p_encoder,
                        p_decoder,
                        vae_optim=tf.keras.optimizers.Adam,
                        vae_lr=config['vae_lr'],
                        vae_beta=config['vae_beta'])

    # create a manager for saving algorithms state to the disk
    p_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**p_vae.get_saveables()),
        os.path.join(config['logging_dir'], 'p_vae'), 1)

    # build a weighted data set
    train_data, val_data = task.build(
        importance_weights=np.ones_like(task.y),
        batch_size=config['vae_batch_size'],
        val_size=config['val_size'])

    # train the initial vae fit to the original data distribution
    p_manager.restore_or_initialize()
    p_vae.launch(train_data,
                 val_data,
                 logger,
                 config['offline_epochs'])

    # build the encoder and decoder distribution and the p model
    q_encoder = Encoder(task.input_shape,
                        config['latent_size'],
                        hidden=config['hidden_size'])
    q_decoder = decoder(task.input_shape,
                        config['latent_size'],
                        hidden=config['hidden_size'])
    q_vae = WeightedVAE(q_encoder,
                        q_decoder,
                        vae_optim=tf.keras.optimizers.Adam,
                        vae_lr=config['vae_lr'],
                        vae_beta=config['vae_beta'])

    # create a manager for saving algorithms state to the disk
    q_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**q_vae.get_saveables()),
        os.path.join(config['logging_dir'], 'q_vae'), 1)

    # create the cbas importance weight generator
    cbas = CBAS(oracle,
                p_vae,
                q_vae,
                latent_size=config['latent_size'])

    # train and validate the q_vae using online samples
    q_encoder.set_weights(p_encoder.get_weights())
    q_decoder.set_weights(p_decoder.get_weights())
    for i in range(config['iterations']):

        # generate an importance weighted dataset
        x, y, w = cbas.generate_data(config['online_batches'],
                                     config['vae_batch_size'],
                                     config['percentile'])

        # evaluate the sampled designs
        score = task.score(x[:config['solver_samples']])
        logger.record("score", score, i, percentile=True)

        # build a weighted data set
        train_data, val_data = task.build(
            x=x.numpy(),
            y=y.numpy(),
            importance_weights=w.numpy(),
            batch_size=config['vae_batch_size'],
            val_size=config['val_size'])

        # train a vae fit using weighted maximum likelihood
        start_epoch = config['online_epochs'] * i + \
            config['offline_epochs']
        q_vae.launch(train_data,
                     val_data,
                     logger,
                     config['online_epochs'],
                     start_epoch=start_epoch)

        # autofocus the forward model using importance weights
        v = cbas.autofocus_weights(
            task.x, batch_size=config['oracle_batch_size'])
        train_data, val_data = task.build(
            x=task.x,
            y=task.y,
            bootstraps=config['bootstraps'],
            importance_weights=v.numpy(),
            batch_size=config['oracle_batch_size'],
            val_size=config['val_size'])

        # train a vae fit using weighted maximum likelihood
        start_epoch = config['autofocus_epochs'] * i + \
            config['oracle_epochs']
        oracle.launch(train_data,
                      val_data,
                      logger,
                      config['autofocus_epochs'],
                      start_epoch=start_epoch)

    # save every model to the disk
    oracle_manager.save()
    p_manager.save()
    q_manager.save()

    # sample designs from the prior
    q_dx = q_decoder.get_distribution(tf.random.normal([
        config['solver_samples'],
        config['latent_size']]), training=False)
    score = task.score(q_dx.sample())
    logger.record("score", score,
                  config['iterations'], percentile=True)
