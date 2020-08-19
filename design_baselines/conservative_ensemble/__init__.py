from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.conservative_ensemble.trainers import ConservativeEnsemble
from design_baselines.conservative_ensemble.nets import ForwardModel
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os


CONSERVATIVE_ENSEMBLE_PARAMS = {
    "logging_dir": "data",
    "task": "Superconductor-v0",
    "task_kwargs": {},
    "is_discrete": False,
    "val_size": 200,
    "batch_size": 128,
    "epochs": 50,
    "activations": (('relu', 'relu'), ('tanh', 'tanh')),
    "hidden_size": 2048,
    "initial_max_std": 0.2,
    "initial_min_std": 0.1,
    "forward_model_lr": 0.001,
    "target_conservative_gap": 0.0,
    "initial_alpha": 0.001,
    "alpha_lr": 0.0,
    "perturbation_lr": 1.0,
    "perturbation_steps": 100,
    "solver_samples": 128,
    "solver_lr": 1.0,
    "solver_steps": 100
}


def conservative_ensemble(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(
        config['task'],
        normalize_x=not config['is_discrete'],
        normalize_y=True,
        **config['task_kwargs'])
    train_data, validate_data = task.build(
        bootstraps=len(config['activations']),
        batch_size=config['batch_size'],
        val_size=config['val_size'])

    # make several keras neural networks with different architectures
    forward_models = [ForwardModel(
        task.input_shape,
        activations=activations,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for activations in config['activations']]

    # create a trainer for a forward model with a conservative objective
    trainer = ConservativeEnsemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        target_conservative_gap=config['target_conservative_gap'],
        initial_alpha=config['initial_alpha'],
        alpha_optim=tf.keras.optimizers.Adam,
        alpha_lr=config['alpha_lr'],
        perturbation_lr=config['perturbation_lr'],
        perturbation_steps=config['perturbation_steps'],
        is_discrete=config['is_discrete'])

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
    x = tf.gather(task.x, indices, axis=0)
    x = tf.math.log(x) if config['is_discrete'] else x

    # evaluate the starting point
    solution = tf.math.softmax(x) if config['is_discrete'] else x
    score = task.score(solution)
    prediction = trainer.get_distribution(solution).mean()

    # record the prediction and score to the logger
    logger.record("score", task.denormalize_y(score), 0)
    logger.record("prediction", task.denormalize_y(prediction), 0)

    # and keep track of the best design sampled so far
    best_solution = None
    best_score = None

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):
        # back propagate through the forward model
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
        prediction = trainer.get_distribution(solution).mean()
        gradient_norm = tf.linalg.norm(
            tf.reshape(grads, [-1, task.input_size]), axis=-1)

        # record the prediction and score to the logger
        logger.record("score", task.denormalize_y(score), i)
        logger.record("prediction", task.denormalize_y(prediction), i)
        logger.record("gradient_norm", gradient_norm, i)

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


def conservative_ensemble_predictions(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(
        config['task'],
        normalize_x=not config['is_discrete'],
        normalize_y=True,
        **config['task_kwargs'])
    train_data, validate_data = task.build(
        bootstraps=len(config['activations']),
        batch_size=config['batch_size'],
        val_size=config['val_size'])

    # make several keras neural networks with different architectures
    forward_models = [ForwardModel(
        task.input_shape,
        activations=activations,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for activations in config['activations']]

    # create a trainer for a forward model with a conservative objective
    trainer = ConservativeEnsemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        target_conservative_gap=config['target_conservative_gap'],
        initial_alpha=config['initial_alpha'],
        alpha_optim=tf.keras.optimizers.Adam,
        alpha_lr=config['alpha_lr'],
        perturbation_lr=config['perturbation_lr'],
        perturbation_steps=config['perturbation_steps'],
        is_discrete=config['is_discrete'])

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
    x = tf.gather(task.x, indices, axis=0)
    x = tf.math.log(x) if config['is_discrete'] else x

    # evaluate the starting point
    solution = tf.math.softmax(x) if config['is_discrete'] else x
    score = task.score(solution)
    preds = [fm.get_distribution(
        solution).mean() for fm in forward_models]

    # record the prediction and score to the logger
    logger.record("score", task.denormalize_y(score), 0)
    for n, prediction_i in enumerate(preds):
        logger.record(f"oracle_{n}/prediction",
                      task.denormalize_y(prediction_i), 0)
        logger.record(f"rank_corr/{n}_to_real",
                      spearman(prediction_i[:, 0], score[:, 0]), 0)
        if n > 0:
            logger.record(f"rank_corr/0_to_{n}",
                          spearman(preds[0][:, 0], prediction_i[:, 0]), 0)

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):
        # back propagate through the forward model
        grads = []
        for fm in forward_models:
            with tf.GradientTape() as tape:
                tape.watch(x)
                solution = tf.math.softmax(x) if config['is_discrete'] else x
                score = fm.get_distribution(solution).mean()
            grads.append(tape.gradient(score, x))

        # use the conservative optimizer to update the solution
        x = x + config['solver_lr'] * grads[0]
        solution = tf.math.softmax(x) if config['is_discrete'] else x

        # evaluate the design using the oracle and the forward model
        score = task.score(solution)
        preds = [fm.get_distribution(
            solution).mean() for fm in forward_models]

        # record the prediction and score to the logger
        logger.record("score", task.denormalize_y(score), i)
        for n, prediction_i in enumerate(preds):
            logger.record(f"oracle_{n}/prediction",
                          task.denormalize_y(prediction_i), i)
            logger.record(f"oracle_{n}/grad_norm", tf.linalg.norm(
                tf.reshape(grads[n], [-1, task.input_size]), axis=-1), i)
            logger.record(f"rank_corr/{n}_to_real",
                          spearman(prediction_i[:, 0], score[:, 0]), i)
            if n > 0:
                logger.record(f"rank_corr/0_to_{n}",
                              spearman(preds[0][:, 0], prediction_i[:, 0]), i)
                logger.record(f"grad_corr/0_to_{n}", tfp.stats.correlation(
                    grads[0], grads[n], sample_axis=0, event_axis=None), i)
