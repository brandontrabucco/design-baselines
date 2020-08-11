from forward_model.data import StaticGraphTask
from forward_model.logger import Logger
from forward_model.noisy_conservative_ensemble.trainers import NoisyConservativeEnsemble
from forward_model.noisy_conservative_ensemble.nets import ForwardModel
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np
import os


def noisy_conservative_ensemble(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, validate_data = task.build(
        bootstraps=config['bootstraps'],
        bootstraps_noise=config['bootstraps_noise'],
        batch_size=config['batch_size'],
        val_size=config['val_size'])

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        task.input_shape,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'],
        act=tfkl.LeakyReLU) for b in range(config['bootstraps'])]

    # create a trainer for a forward model with a conservative objective
    trainer = NoisyConservativeEnsemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        target_conservative_gap=config['target_conservative_gap'],
        initial_alpha=config['initial_alpha'],
        alpha_optim=tf.keras.optimizers.Adam,
        alpha_lr=config['alpha_lr'],
        perturbation_lr=config['perturbation_lr'],
        perturbation_steps=config['perturbation_steps'])

    # create a manager for saving algorithms state to the disk
    manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**trainer.get_saveables()),
        os.path.join(config['logging_dir'], 'ckpt'), 1)

    # train the model for an additional number of epochs
    manager.restore_or_initialize()
    trainer.launch(train_data, validate_data, logger, config['epochs'])
    manager.save()

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(task.y[:, 0], k=config['solver_samples'])[1]

    # evaluate the initial design using the oracle and the forward model
    solution = tf.gather(task.x, indices, axis=0)
    score = task.score(solution)
    prediction = trainer.get_distribution(solution).mean()

    # record the prediction and score to the logger
    logger.record("score", score, 0)
    logger.record("prediction", prediction, 0)

    # and keep track of the best design sampled so far
    best_design = None
    best_score = None

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):

        # back propagate through the forward model
        with tf.GradientTape() as tape:
            tape.watch(solution)
            score = trainer.get_distribution(solution).sample()
        grads = tape.gradient(score, solution)
        solution = solution + config['solver_lr'] * grads

        # evaluate the design using the oracle and the forward model
        gradient_norm = tf.linalg.norm(grads, axis=1)
        score = task.score(solution)
        prediction = trainer.get_distribution(solution).sample()

        # record the prediction and score to the logger
        logger.record("gradient_norm", gradient_norm, i)
        logger.record("score", score, i)
        logger.record("prediction", prediction, i)

        # update the best design every iteration
        idx = np.argmax(score.numpy())
        if best_design is None or score[idx] > best_score:
            best_score = score[idx]
            best_design = solution[idx]

    # save the best design to the disk
    np.save(os.path.join(
        config['logging_dir'], 'score.npy'), best_score)
    np.save(os.path.join(
        config['logging_dir'], 'design.npy'), best_design)


def noisy_conservative_ensemble_predictions(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, validate_data = task.build(
        bootstraps=2,
        bootstraps_noise=config['bootstraps_noise'],
        batch_size=config['batch_size'],
        val_size=config['val_size'])

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        task.input_shape,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'],
        act=tfkl.LeakyReLU) for b in range(config['bootstraps'])]

    # create a trainer for a forward model with a conservative objective
    trainer = NoisyConservativeEnsemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        target_conservative_gap=config['target_conservative_gap'],
        initial_alpha=config['initial_alpha'],
        alpha_optim=tf.keras.optimizers.Adam,
        alpha_lr=config['alpha_lr'],
        perturbation_lr=config['perturbation_lr'],
        perturbation_steps=config['perturbation_steps'])

    # create a manager for saving algorithms state to the disk
    manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**trainer.get_saveables()),
        os.path.join(config['logging_dir'], 'ckpt'), 1)

    # train the model for an additional number of epochs
    manager.restore_or_initialize()
    trainer.launch(train_data, validate_data, logger, config['epochs'])
    manager.save()

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(task.y[:, 0], k=config['solver_samples'])[1]

    # evaluate the initial design using the oracle and the forward model
    solution = tf.gather(task.x, indices, axis=0)
    score = task.score(solution)
    prediction0 = forward_models[0].get_distribution(solution).sample()
    prediction1 = forward_models[1].get_distribution(solution).sample()

    # record the prediction and score to the logger
    logger.record("score", score, 0)
    logger.record("model_0/prediction", prediction0, 0)
    logger.record("model_1/prediction", prediction1, 0)

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):

        # back propagate through the forward model
        with tf.GradientTape() as tape:
            tape.watch(solution)
            score = forward_models[0].get_distribution(solution).sample()
        grads = tape.gradient(score, solution)
        solution = solution + config['solver_lr'] * grads

        # evaluate the design using the oracle and the forward model
        gradient_norm = tf.linalg.norm(grads, axis=1)
        score = task.score(solution)
        prediction0 = forward_models[0].get_distribution(solution).sample()
        prediction1 = forward_models[1].get_distribution(solution).sample()

        # record the prediction and score to the logger
        logger.record("gradient_norm", gradient_norm, i)
        logger.record("score", score, i)
        logger.record("model_0/prediction", prediction0, i)
        logger.record("model_1/prediction", prediction1, i)
