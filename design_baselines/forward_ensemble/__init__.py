from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import gumb_noise
from design_baselines.forward_ensemble.trainers import Ensemble
from design_baselines.forward_ensemble.nets import ForwardModel
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np
import os


def forward_ensemble(config):
    """Train a forward model and perform model based optimization
    using a bootstrap ensemble

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, validate_data = task.build(
        bootstraps=config['bootstraps'],
        batch_size=config['batch_size'],
        val_size=config['val_size'])

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        task.input_shape,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'],
        act=tfkl.LeakyReLU) for b in range(config['bootstraps'])]

    # create a trainer for a forward model ensemble
    trainer = Ensemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        is_discrete=config['is_discrete'],
        noise_std=config.get('noise_std', 0.0),
        keep=config.get('keep', 0.9),
        temp=config.get('temp', 1.0))

    # create a manager for saving algorithms state to the disk
    manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**trainer.get_saveables()),
        os.path.join(config['logging_dir'], 'ckpt'), 1)

    # train the model for an additional number of epochs
    manager.restore_or_initialize()
    trainer.launch(train_data, validate_data, logger, config['epochs'])

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(task.y[:, 0], k=config['solver_samples'])[1]
    x = tf.gather(task.x, indices, axis=0)
    x = tf.math.log(gumb_noise(
        x, config.get('keep', 0.9), config.get('temp', 1.0))) \
        if config['is_discrete'] else x

    # evaluate the starting point
    solution = tf.math.softmax(x) if config['is_discrete'] else x
    score = task.score(solution)
    model = trainer.get_distribution(solution).mean()

    # record the prediction and score to the logger
    logger.record("score", score, 0, percentile=True)
    logger.record("model", model, 0)
    logger.record(f"rank_corr/model_to_real",
                  spearman(model[:, 0], score[:, 0]), 0)

    # and keep track of the best design sampled so far
    best_solution = None
    best_score = None

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):
        # back propagate through the forward model ensemble
        with tf.GradientTape() as tape:
            tape.watch(x)
            solution = tf.math.softmax(x) if config['is_discrete'] else x
            score = trainer.get_distribution(solution).mean()
        grads = tape.gradient(score, x)

        # use the conservative optimizer to update the solution
        x = x + config['solver_lr'] * grads
        solution = tf.math.softmax(x) if config['is_discrete'] else x

        # evaluate the design using the oracle and the forward model
        score = task.score(solution)
        model = trainer.get_distribution(solution).mean()

        # record the prediction and score to the logger
        logger.record("score", score, i, percentile=True)
        logger.record("model", model, i)
        logger.record(f"rank_corr/model_to_real",
                      spearman(model[:, 0], score[:, 0]), i)
        logger.record(f"grad_norm", tf.linalg.norm(
            tf.reshape(grads, [-1, task.input_size]), axis=-1), i)

        # update the best design every iteration
        idx = np.argmax(score)
        if best_solution is None or score[idx] > best_score:
            best_score = score[idx]
            best_solution = solution[idx]

    # save the best design to the disk
    np.save(os.path.join(
        config['logging_dir'], 'score.npy'), best_score)
    np.save(os.path.join(
        config['logging_dir'], 'solution.npy'), best_solution)
