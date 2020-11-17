from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.cbas.trainers import Ensemble
from design_baselines.cbas.trainers import WeightedVAE
from design_baselines.cbas.trainers import CBAS
from design_baselines.cbas.nets import ForwardModel
from design_baselines.cbas.nets import Encoder
from design_baselines.cbas.nets import DiscreteDecoder
from design_baselines.cbas.nets import ContinuousDecoder
import tensorflow as tf
import numpy as np
import os


def cbas(config):
    """Optimize a design problem score using the algorithm CBAS
    otherwise known as Conditioning by Adaptive Sampling

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    x = task.x
    y = task.y

    if config['normalize_ys']:

        # compute normalization statistics for the score
        mu_y = np.mean(y, axis=0, keepdims=True)
        mu_y = mu_y.astype(np.float32)
        y = y - mu_y
        st_y = np.std(y, axis=0, keepdims=True)
        st_y = np.where(np.equal(st_y, 0), 1, st_y)
        st_y = st_y.astype(np.float32)
        y = y / st_y

    else:

        # compute normalization statistics for the data vectors
        mu_y = np.zeros_like(y[:1])
        st_y = np.ones_like(y[:1])

    if config['normalize_xs'] and not config['is_discrete']:

        # compute normalization statistics for the data vectors
        mu_x = np.mean(x, axis=0, keepdims=True)
        mu_x = mu_x.astype(np.float32)
        x = x - mu_x
        st_x = np.std(x, axis=0, keepdims=True)
        st_x = np.where(np.equal(st_x, 0), 1, st_x)
        st_x = st_x.astype(np.float32)
        x = x / st_x

    else:

        # compute normalization statistics for the data vectors
        mu_x = np.zeros_like(x[:1])
        st_x = np.ones_like(x[:1])

    # create the training task and logger
    train_data, val_data = task.build(
        x=x, y=y, bootstraps=config['bootstraps'],
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
    ensemble = Ensemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['ensemble_lr'])

    # create a manager for saving algorithms state to the disk
    ensemble_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**ensemble.get_saveables()),
        os.path.join(config['logging_dir'], 'ensemble'), 1)

    # train the model for an additional number of epochs
    ensemble_manager.restore_or_initialize()
    ensemble.launch(train_data,
                    val_data,
                    logger,
                    config['ensemble_epochs'])
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
        x=x, y=y, importance_weights=np.ones_like(task.y),
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
        directory=os.path.join(config['logging_dir'], 'q_vae'),
        max_to_keep=1)

    # create the cbas importance weight generator
    cbas = CBAS(ensemble,
                p_vae,
                q_vae,
                latent_size=config['latent_size'])

    # train and validate the q_vae using online samples
    q_encoder.set_weights(p_encoder.get_weights())
    q_decoder.set_weights(p_decoder.get_weights())
    for i in range(config['iterations']):

        # generate an importance weighted dataset
        x, y, w = cbas.generate_data(
            config['online_batches'],
            config['vae_batch_size'],
            config['percentile'])

        # evaluate the sampled designs
        score = task.score(x[:config['solver_samples']] * st_x + mu_x)
        logger.record("score",
                      score,
                      i,
                      percentile=True)

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

    # save every model to the disk
    ensemble_manager.save()
    p_manager.save()
    q_manager.save()

    # sample designs from the prior
    z = tf.random.normal([config['solver_samples'], config['latent_size']])
    q_dx = q_decoder.get_distribution(z, training=False)
    score = task.score(q_dx.sample() * st_x + mu_x)
    logger.record("score",
                  score,
                  config['iterations'],
                  percentile=True)
