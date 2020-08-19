from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.conservative_ensemble.trainers import ConservativeEnsemble
from design_baselines.conservative_ensemble.nets import ForwardModel
import tensorflow.keras.layers as tfkl
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
    "bootstraps": 1,
    "epochs": 50,
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
    task = StaticGraphTask(config['task'],
                           normalize_x=not config['is_discrete'],
                           normalize_y=True,
                           **config['task_kwargs'])
    train_data, validate_data = task.build(bootstraps=config['bootstraps'],
                                           batch_size=config['batch_size'],
                                           val_size=config['val_size'])
    logger = Logger(config['logging_dir'])

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        task.input_shape,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'],
        act=tfkl.LeakyReLU) for b in range(config['bootstraps'])]

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
    task = StaticGraphTask(config['task'],
                           normalize_x=not config['is_discrete'],
                           normalize_y=True,
                           **config['task_kwargs'])
    train_data, validate_data = task.build(bootstraps=config['bootstraps'],
                                           batch_size=config['batch_size'],
                                           val_size=config['val_size'])
    logger = Logger(config['logging_dir'])

    # make several keras neural networks with two hidden layers
    forward_models_0 = [ForwardModel(
        task.input_shape,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'],
        act=tfkl.LeakyReLU) for b in range(config['bootstraps'])]

    # create a trainer for a forward model with a conservative objective
    trainer_0 = ConservativeEnsemble(
        forward_models_0,
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
    manager_0 = tf.train.CheckpointManager(
        tf.train.Checkpoint(**trainer_0.get_saveables()),
        os.path.join(config['logging_dir'], 'model_0'), 1)

    # train the model for an additional number of epochs
    manager_0.restore_or_initialize()
    trainer_0.launch(train_data, validate_data, logger, config['epochs'],
                     header='model_0/')
    manager_0.save()

    # make several keras neural networks with two hidden layers
    forward_models_1 = [ForwardModel(
        task.input_shape,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'],
        act=lambda: tfkl.Activation('tanh')) for b in range(config['bootstraps'])]

    # create a trainer for a forward model with a conservative objective
    trainer_1 = ConservativeEnsemble(
        forward_models_1,
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
    manager_1 = tf.train.CheckpointManager(
        tf.train.Checkpoint(**trainer_1.get_saveables()),
        os.path.join(config['logging_dir'], 'model_1'), 1)

    # train the model for an additional number of epochs
    manager_1.restore_or_initialize()
    trainer_1.launch(train_data, validate_data, logger, config['epochs'],
                     header='model_1/')
    manager_1.save()

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(task.y[:, 0], k=config['solver_samples'])[1]
    x = tf.gather(task.x, indices, axis=0)
    x = tf.math.log(x) if config['is_discrete'] else x

    # evaluate the starting point
    solution = tf.math.softmax(x) if config['is_discrete'] else x
    score = task.score(solution)
    prediction_0 = trainer_0.get_distribution(solution).mean()
    prediction_1 = trainer_1.get_distribution(solution).mean()

    # record the prediction and score to the logger
    logger.record("score", task.denormalize_y(score), 0)
    logger.record("model_0/prediction",
                  task.denormalize_y(prediction_0), 0)
    logger.record("model_1/prediction",
                  task.denormalize_y(prediction_1), 0)
    logger.record("rank_corr/0_to_1",
                  spearman(prediction_0[:, 0], prediction_1[:, 0]), 0)
    logger.record("rank_corr/0_to_real",
                  spearman(prediction_0[:, 0], score[:, 0]), 0)
    logger.record("rank_corr/1_to_real",
                  spearman(prediction_1[:, 0], score[:, 0]), 0)

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):
        # back propagate through the conservative model
        with tf.GradientTape() as tape:
            tape.watch(x)
            solution = tf.math.softmax(x) if config['is_discrete'] else x
            score = trainer_0.get_distribution(solution).mean()
        grads_0 = tape.gradient(score, solution)

        # back propagate through the vanilla model
        with tf.GradientTape() as tape:
            tape.watch(x)
            solution = tf.math.softmax(x) if config['is_discrete'] else x
            score = trainer_1.get_distribution(solution).mean()
        grads_1 = tape.gradient(score, solution)

        # use the conservative optimizer to update the solution
        x = x + config['solver_lr'] * grads_0
        solution = tf.math.softmax(x) if config['is_discrete'] else x

        # calculate the element-wise gradient correlation
        gradient_corr = tfp.stats.correlation(
            grads_0, grads_1, sample_axis=0, event_axis=None)
        gradient_norm = tf.linalg.norm(
            tf.reshape(grads_0, [-1, task.input_size]), axis=-1)

        # evaluate the design using the oracle and the forward model
        score = task.score(solution)
        prediction_0 = trainer_0.get_distribution(solution).mean()
        prediction_1 = trainer_1.get_distribution(solution).mean()

        # record the prediction and score to the logger
        logger.record("gradient_corr", gradient_corr, i)
        logger.record("gradient_norm", gradient_norm, i)
        logger.record("score", task.denormalize_y(score), i)
        logger.record("model_0/prediction",
                      task.denormalize_y(prediction_0), i)
        logger.record("model_1/prediction",
                      task.denormalize_y(prediction_1), i)
        logger.record("rank_corr/0_to_1",
                      spearman(prediction_0[:, 0], prediction_1[:, 0]), i)
        logger.record("rank_corr/0_to_real",
                      spearman(prediction_0[:, 0], score[:, 0]), i)
        logger.record("rank_corr/1_to_real",
                      spearman(score[:, 0], prediction_1[:, 0]), i)
