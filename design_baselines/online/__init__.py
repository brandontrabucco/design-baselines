from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.online.trainers import ConservativeMaximumLikelihood
from design_baselines.online.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf
import numpy as np


def online(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])

    # make a neural network to predict scores
    forward_model = ForwardModel(
        task.input_shape,
        activations=config['activations'],
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])

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

    # create a bootstrapped data set
    train_data, validate_data = task.build(
        x=x, y=y,
        batch_size=config['batch_size'],
        val_size=config['val_size'])

    # create a trainer for a forward model with a conservative objective
    trainer = ConservativeMaximumLikelihood(
        forward_model,
        forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        initial_alpha=config['initial_alpha'],
        alpha_opt=tf.keras.optimizers.Adam,
        alpha_lr=config['alpha_lr'],
        target_conservatism=config['target_conservatism'],
        negatives_fraction=config['negatives_fraction'],
        lookahead_steps=config['lookahead_steps'],
        lookahead_backprop=config['lookahead_backprop'],
        solver_lr=config['solver_lr'] * np.sqrt(np.prod(x.shape[1:])),
        solver_interval=config['solver_interval'],
        solver_warmup=config['solver_warmup'],
        solver_steps=config['solver_steps'],
        is_discrete=config['is_discrete'],
        continuous_noise_std=config.get('continuous_noise_std', 0.0),
        discrete_smoothing=config.get('discrete_smoothing', 0.6))

    # select the top k initial designs from the dataset
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    indices = tf.math.top_k(y[:, 0], k=config['batch_size'])[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = tf.math.log(soft_noise(
        initial_x, config.get('discrete_smoothing', 0.6))) \
        if config['is_discrete'] else initial_x

    # create the starting point for the optimizer
    evaluations = 0
    trainer.solution = tf.Variable(x)
    trainer.done = tf.Variable(
        tf.fill([config['batch_size']] + [1 for _ in x.shape[1:]], False))

    def evaluate_solution(xt):
        nonlocal evaluations

        # evaluate the design using the oracle and the forward model
        with tf.GradientTape() as tape:
            tape.watch(xt)
            solution = tf.math.softmax(xt) if config['is_discrete'] else xt
            model = forward_model.get_distribution(solution).mean()

        # evaluate the predictions and gradient norm
        score = task.score(solution * st_x + mu_x)
        grads = tape.gradient(model, xt)
        model = model * st_y + mu_y
        evaluations += 1

        # record the prediction and score to the logger
        logger.record("score",
                      score, evaluations, percentile=True)
        logger.record("distance/travelled",
                      tf.linalg.norm(solution - initial_x), evaluations)
        logger.record("distance/from_mean",
                      tf.linalg.norm(solution - mean_x), evaluations)
        logger.record(f"oracle/prediction",
                      model, evaluations)
        logger.record(f"oracle/grad_norm", tf.linalg.norm(
            tf.reshape(grads, [-1, task.input_size]), axis=-1), evaluations)
        logger.record(f"rank_corr/model_to_real",
                      spearman(model[:, 0], score[:, 0]), evaluations)

    # keep track of when to record performance
    interval = trainer.solver_interval
    warmup = trainer.solver_warmup

    # train model for many epochs with conservatism
    for e in range(config['epochs']):

        statistics = defaultdict(list)
        for x, y in train_data:
            for name, tensor in trainer.train_step(x, y).items():
                statistics[name].append(tensor)

            # evaluate the current solution
            if tf.logical_and(
                    tf.equal(tf.math.mod(trainer.step, interval), 0),
                    tf.math.greater_equal(trainer.step, warmup)):
                evaluate_solution(trainer.solution)

        for name in statistics.keys():
            logger.record(
                name, tf.concat(statistics[name], axis=0), e)

        statistics = defaultdict(list)
        for x, y in validate_data:
            for name, tensor in trainer.validate_step(x, y).items():
                statistics[name].append(tensor)

        for name in statistics.keys():
            logger.record(
                name, tf.concat(statistics[name], axis=0), e)
