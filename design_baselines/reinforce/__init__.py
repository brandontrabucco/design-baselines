from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.reinforce.trainers import Ensemble
from design_baselines.reinforce.nets import ForwardModel
from design_baselines.reinforce.nets import DiscreteMarginal
from design_baselines.reinforce.nets import ContinuousMarginal
from design_baselines.utils import render_video
import tensorflow as tf
import numpy as np
import os


def reinforce(config):
    """Optimizes over designs x in an offline optimization problem
    using the CMA Evolution Strategy

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
        forward_model_lr=config['ensemble_lr'],
        is_discrete=config['is_discrete'],
        continuous_noise_std=config.get('continuous_noise_std', 0.0),
        discrete_keep=config.get('discrete_keep', 1.0))

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

    rl_opt = tf.keras.optimizers.Adam(
        learning_rate=config['reinforce_lr'])

    # select the top 1 initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=config['solver_samples'])[1]
    initial_x = tf.gather(x, indices, axis=0)

    if config['is_discrete']:
        logits = tf.reduce_mean(initial_x, axis=0)
        logits = tf.maximum(logits, 0.001)
        logits = tf.math.log(logits)
        sampler = DiscreteMarginal(logits)

    else:
        mean = tf.reduce_mean(initial_x, axis=0)
        std = tf.maximum(tf.math.reduce_std(initial_x, axis=0), 0.001)
        logstd = tf.math.log(std)
        sampler = ContinuousMarginal(mean, logstd)

    for iteration in range(config['iterations']):

        with tf.GradientTape() as tape:
            td = sampler.get_distribution()
            tx = td.sample(sample_shape=config['reinforce_batch_size'])
            ty = ensemble.get_distribution(tx).mean()
            loss = td.log_prob(tx) * ty

        logger.record("reinforce/prediction",
                      ty, iteration, percentile=True)
        logger.record("reinforce/loss",
                      loss, iteration, percentile=True)
        print(f"[Iteration {iteration}] "
              f"Average Score = {tf.reduce_mean(ty)}")

        grads = tape.gradient(
            loss, sampler.trainable_variables)

        rl_opt.apply_gradients(zip(
            grads, sampler.trainable_variables))

        td = sampler.get_distribution()
        solution = td.sample(sample_shape=config['solver_samples'])

        # evaluate the found solution and record a video
        score = task.score(solution * st_x + mu_x)
        logger.record(
            "score", score, iteration, percentile=True)

        # render a video of the best solution found at the end
        if iteration == config['iterations'] - 1:
            render_video(config, task, (
                solution * st_x + mu_x)[np.argmax(np.reshape(score, [-1]))])
