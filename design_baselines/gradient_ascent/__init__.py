from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.gradient_ascent.trainers import MaximumLikelihood
from design_baselines.gradient_ascent.trainers import Ensemble
from design_baselines.gradient_ascent.nets import ForwardModel
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os
import glob


def gradient_ascent(config):
    """Train a Score Function to solve a Model-Based Optimization
    using gradient ascent on the input design

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])

    # make several keras neural networks with different architectures
    forward_models = [ForwardModel(
        task.input_shape,
        activations=activations,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for activations in config['activations']]

    # save the initial dataset statistics for safe keeping
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

        # compute normalization statistics for the score
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

        # compute normalization statistics for the score
        mu_x = np.zeros_like(x[:1])
        st_x = np.ones_like(x[:1])

    # scale the learning rate based on the number of channels in x
    config['solver_lr'] *= np.sqrt(np.prod(x.shape[1:]))

    # make a neural network to predict scores
    held_out_models = [ForwardModel(
        task.input_shape,
        activations=['relu' if i == j else 'tanh' for j in range(8)],
        hidden=256,
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for i in range(8)]

    # create a trainer for a forward model with a conservative objective
    held_out_trainer = Ensemble(
        held_out_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=0.001,
        is_discrete=config['is_discrete'],
        continuous_noise_std=config.get('continuous_noise_std', 0.0),
        discrete_smoothing=config.get('discrete_smoothing', 0.6))

    # create a bootstrapped data set
    held_out_train_data, held_out_validate_data = task.build(
        x=x, y=y,
        batch_size=config['batch_size'],
        val_size=config['val_size'],
        bootstraps=len(held_out_models))

    # train a held-out model on the validation set
    for e in range(100):

        statistics = defaultdict(list)
        for xi, yi, bi in held_out_train_data:
            for name, tensor in held_out_trainer.train_step(xi, yi, bi).items():
                statistics["held_out/" + name].append(tensor)

        for name in statistics.keys():
            logger.record(
                name, tf.concat(statistics[name], axis=0), e)

        statistics = defaultdict(list)
        for xi, yi in held_out_validate_data:
            for name, tensor in held_out_trainer.validate_step(xi, yi).items():
                statistics["held_out/" + name].append(tensor)

        for name in statistics.keys():
            logger.record(
                name, tf.concat(statistics[name], axis=0), e)

    trs = []
    for i, fm in enumerate(forward_models):

        # create a bootstrapped data set
        train_data, validate_data = task.build(
            x=x, y=y, batch_size=config['batch_size'],
            val_size=config['val_size'], bootstraps=1)

        # create a trainer for a forward model with a conservative objective
        trainer = MaximumLikelihood(
            fm,
            forward_model_optim=tf.keras.optimizers.Adam,
            forward_model_lr=config['forward_model_lr'],
            is_discrete=config['is_discrete'],
            continuous_noise_std=config.get('continuous_noise_std', 0.0),
            discrete_smoothing=config.get('discrete_smoothing', 0.6))

        # train the model for an additional number of epochs
        trs.append(trainer)
        trainer.launch(train_data, validate_data, logger,
                       config['epochs'], header=f'oracle_{i}/')

    # select the top k initial designs from the dataset
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    indices = tf.math.top_k(y[:, 0], k=config['solver_samples'])[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = tf.math.log(soft_noise(initial_x,
                               config.get('discrete_smoothing', 0.6))) \
        if config['is_discrete'] else initial_x

    # evaluate the starting point
    solution = tf.math.softmax(x) if config['is_discrete'] else x
    score = task.score(solution * st_x + mu_x)
    preds = [fm.get_distribution(
        solution).mean() * st_y + mu_y for fm in forward_models]

    # record the prediction and score to the logger
    logger.record("score", score, 0, percentile=True)
    logger.record("distance/travelled", tf.linalg.norm(solution - initial_x), 0)
    logger.record("distance/from_mean", tf.linalg.norm(solution - mean_x), 0)
    for n, prediction_i in enumerate(preds):
        logger.record(f"oracle_{n}/prediction", prediction_i, 0)
        logger.record(f"rank_corr/{n}_to_real",
                      spearman(prediction_i[:, 0], score[:, 0]), 0)
        if n > 0:
            logger.record(f"rank_corr/0_to_{n}",
                          spearman(preds[0][:, 0], prediction_i[:, 0]), 0)

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):
        # back propagate through the forward model
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = []
            for fm in forward_models:
                solution = tf.math.softmax(x) if config['is_discrete'] else x
                predictions.append(fm.get_distribution(solution).mean())
            if config['aggregation_method'] == 'mean':
                score = tf.reduce_min(predictions, axis=0)
            if config['aggregation_method'] == 'min':
                score = tf.reduce_min(predictions, axis=0)
            if config['aggregation_method'] == 'random':
                score = predictions[np.random.randint(len(predictions))]
        grads = tape.gradient(score, x)

        # use the conservative optimizer to update the solution
        x = x + config['solver_lr'] * grads
        solution = tf.math.softmax(x) if config['is_discrete'] else x

        # evaluate the design using the oracle and the forward model
        score = task.score(solution * st_x + mu_x)
        preds = [fm.get_distribution(
            solution).mean() * st_y + mu_y for fm in forward_models]

        held_out_m = [m.get_distribution(solution).mean()
                      for m in held_out_models]
        held_out_s = [m.get_distribution(solution).stddev()
                      for m in held_out_models]

        max_of_mean = tf.reduce_max(held_out_m, axis=0)
        max_of_stddev = tf.reduce_max(held_out_s, axis=0)
        min_of_mean = tf.reduce_min(held_out_m, axis=0)
        min_of_stddev = tf.reduce_min(held_out_s, axis=0)
        mean_of_mean = tf.reduce_mean(held_out_m, axis=0)
        mean_of_stddev = tf.reduce_mean(held_out_s, axis=0)
        logger.record(f"held_out/max_of_mean",
                      max_of_mean, i)
        logger.record(f"held_out/max_of_stddev",
                      max_of_stddev, i)
        logger.record(f"held_out/min_of_mean",
                      min_of_mean, i)
        logger.record(f"held_out/min_of_stddev",
                      min_of_stddev, i)
        logger.record(f"held_out/mean_of_mean",
                      mean_of_mean, i)
        logger.record(f"held_out/mean_of_stddev",
                      mean_of_stddev, i)

        # record the prediction and score to the logger
        logger.record("score", score, i, percentile=True)
        logger.record("distance/travelled", tf.linalg.norm(solution - initial_x), i)
        logger.record("distance/from_mean", tf.linalg.norm(solution - mean_x), i)
        for n, prediction_i in enumerate(preds):
            logger.record(f"oracle_{n}/prediction", prediction_i, i)
            logger.record(f"oracle_{n}/grad_norm", tf.linalg.norm(
                tf.reshape(grads[n], [-1, task.input_size]), axis=-1), i)
            logger.record(f"rank_corr/{n}_to_real",
                          spearman(prediction_i[:, 0], score[:, 0]), i)
            if n > 0:
                logger.record(f"rank_corr/0_to_{n}",
                              spearman(preds[0][:, 0], prediction_i[:, 0]), i)
                logger.record(f"grad_corr/0_to_{n}", tfp.stats.correlation(
                    grads[0], grads[n], sample_axis=0, event_axis=None), i)

        # save the best design to the disk
        np.save(os.path.join(
            config['logging_dir'], f'score_{i}.npy'), score)
        np.save(os.path.join(
            config['logging_dir'], f'solution_{i}.npy'), solution)


def ablate_architecture(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])

    # save the initial dataset statistics for safe keeping
    x = task.x
    y = task.y

    if config['normalize_ys']:
        # compute normalization statistics for the score
        mu_y = np.mean(y, axis=0, keepdims=True).astype(np.float32)
        y = y - mu_y
        st_y = np.std(y, axis=0, keepdims=True).astype(np.float32)
        st_y = np.where(np.equal(st_y, 0), 1, st_y)
        y = y / st_y
    else:
        # compute normalization statistics for the score
        mu_y = np.zeros_like(y[:1])
        st_y = np.ones_like(y[:1])

    if config['normalize_xs'] and not config['is_discrete']:
        # compute normalization statistics for the data vectors
        mu_x = np.mean(x, axis=0, keepdims=True).astype(np.float32)
        x = x - mu_x
        st_x = np.std(x, axis=0, keepdims=True).astype(np.float32)
        st_x = np.where(np.equal(st_x, 0), 1, st_x)
        x = x / st_x
    else:
        # compute normalization statistics for the score
        mu_x = np.zeros_like(x[:1])
        st_x = np.ones_like(x[:1])

    # make a neural network to predict scores
    held_out_models = [ForwardModel(
        task.input_shape,
        activations=config['activations'][0],
        hidden=256,
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for i in range(8)]

    # create a trainer for a forward model with a conservative objective
    held_out_trainer = Ensemble(
        held_out_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=0.001)

    # create a bootstrapped data set
    held_out_train_data, held_out_validate_data = task.build(
        x=x, y=y,
        batch_size=config['batch_size'],
        val_size=config['val_size'],
        bootstraps=len(held_out_models))

    # train a held-out model on the validation set
    for e in range(100):

        statistics = defaultdict(list)
        for xi, yi, bi in held_out_train_data:
            for name, tensor in held_out_trainer.train_step(xi, yi, bi).items():
                statistics["held_out2/" + name].append(tensor)

        for name in statistics.keys():
            logger.record(
                name, tf.concat(statistics[name], axis=0), e)

        statistics = defaultdict(list)
        for xi, yi in held_out_validate_data:
            for name, tensor in held_out_trainer.validate_step(xi, yi).items():
                statistics["held_out2/" + name].append(tensor)

        for name in statistics.keys():
            logger.record(
                name, tf.concat(statistics[name], axis=0), e)

    def evaluate_solution(solution, evaluations):

        # evaluate the design using the oracle and the forward model
        held_out_m = [m.get_distribution(solution).mean()
                      for m in held_out_models]
        held_out_s = [m.get_distribution(solution).stddev()
                      for m in held_out_models]

        max_of_mean = tf.reduce_max(held_out_m, axis=0)
        max_of_stddev = tf.reduce_max(held_out_s, axis=0)
        min_of_mean = tf.reduce_min(held_out_m, axis=0)
        min_of_stddev = tf.reduce_min(held_out_s, axis=0)
        mean_of_mean = tf.reduce_mean(held_out_m, axis=0)
        mean_of_stddev = tf.reduce_mean(held_out_s, axis=0)

        logger.record(f"oracle/same_architecture/max_of_mean",
                      max_of_mean, evaluations)
        logger.record(f"oracle/same_architecture/max_of_stddev",
                      max_of_stddev, evaluations)
        logger.record(f"oracle/same_architecture/min_of_mean",
                      min_of_mean, evaluations)
        logger.record(f"oracle/same_architecture/min_of_stddev",
                      min_of_stddev, evaluations)
        logger.record(f"oracle/same_architecture/mean_of_mean",
                      mean_of_mean, evaluations)
        logger.record(f"oracle/same_architecture/mean_of_stddev",
                      mean_of_stddev, evaluations)

    # train a held-out model on the validation set

    for file_name in glob.glob(
            os.path.join(config['logging_dir'], f'solution_*.npy')):

        iteration = file_name.split('_')[-1].split('.')[0]
        solution = np.load(file_name)
        evaluate_solution(solution, int(iteration))
