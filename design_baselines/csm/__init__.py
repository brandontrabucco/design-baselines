from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.csm.trainers import ConservativeMaximumLikelihood
from design_baselines.csm.nets import ForwardModel
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os


def csm(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

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

    # make several keras neural networks with different architectures
    vanilla_models = [ForwardModel(
        task.input_shape,
        activations=activations,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for activations in config['activations']]

    # sync corresponding models
    for model_a, model_b in zip(
            forward_models, vanilla_models):
        model_b.set_weights(model_a.get_weights())

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
    config['perturbation_lr'] *= np.sqrt(np.prod(x.shape[1:]))
    config['solver_lr'] *= np.sqrt(np.prod(x.shape[1:]))

    trs = []
    for i, fm in enumerate(forward_models):

        # create a bootstrapped data set
        train_data, validate_data = task.build(
            x=x, y=y, batch_size=config['batch_size'],
            val_size=config['val_size'], bootstraps=1)

        # create a trainer for a forward model with a conservative objective
        trainer = ConservativeMaximumLikelihood(
            fm, vanilla_models[i],
            forward_model_optim=tf.keras.optimizers.Adam,
            forward_model_lr=config['forward_model_lr'],
            target_conservative_gap=config['target_conservative_gap'],
            target_rank_corr_gap=config['target_rank_corr_gap'],
            initial_alpha=config['initial_alpha'],
            alpha_optim=tf.keras.optimizers.Adam,
            alpha_lr=config['alpha_lr'],
            perturbation_lr=config['perturbation_lr'],
            perturbation_steps=config['perturbation_steps'],
            perturbation_backprop=config['perturbation_backprop'],
            is_discrete=config['is_discrete'],
            noise_std=config.get('noise_std', 0.0),
            keep=config.get('keep', 1.0),
            temp=config.get('temp', 0.001))

        # train the model for an additional number of epochs
        trs.append(trainer)
        trainer.launch(train_data, validate_data, logger,
                       config['epochs'], header=f'oracle_{i}/')

    # select the top k initial designs from the dataset
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    indices = tf.math.top_k(y[:, 0], k=config['solver_samples'])[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = tf.math.log(soft_noise(initial_x,
                               config.get('keep', 0.999),
                               config.get('temp', 0.001))) \
        if config['is_discrete'] else initial_x

    # evaluate the starting point
    solution = tf.math.softmax(x) if config['is_discrete'] else x
    score = task.score(solution * st_x + mu_x)
    preds = [fm.get_distribution(
        solution).mean() * st_y + mu_y for fm in forward_models]

    # evaluate the conservative gap for every model
    perturb_preds = [tr.fm.get_distribution(
        tr.optimize(solution)).mean() * st_y + mu_y for tr in trs]
    perturb_gap = [
        b - a for a, b in zip(preds, perturb_preds)]

    # record the prediction and score to the logger
    logger.record("score", score, 0, percentile=True)
    logger.record("distance/travelled",
                  tf.linalg.norm(solution - initial_x), 0)
    logger.record("distance/from_mean",
                  tf.linalg.norm(solution - mean_x), 0)
    for n, prediction_i in enumerate(preds):
        logger.record(f"oracle_{n}/gap", perturb_gap[n], 0)
        logger.record(f"oracle_{n}/prediction", prediction_i, 0)
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
        x = x + config['solver_lr'] * grads[np.random.randint(len(grads))]
        solution = tf.math.softmax(x) if config['is_discrete'] else x

        # evaluate the design using the oracle and the forward model
        score = task.score(solution * st_x + mu_x)
        preds = [fm.get_distribution(
            solution).mean() * st_y + mu_y for fm in forward_models]

        # evaluate the conservative gap for every model
        perturb_preds = [tr.fm.get_distribution(
            tr.optimize(solution)).mean() * st_y + mu_y for tr in trs]
        perturb_gap = [
            b - a for a, b in zip(preds, perturb_preds)]

        # record the prediction and score to the logger
        logger.record("score", score, i, percentile=True)
        logger.record("distance/travelled",
                      tf.linalg.norm(solution - initial_x), i)
        logger.record("distance/from_mean",
                      tf.linalg.norm(solution - mean_x), i)
        for n, prediction_i in enumerate(preds):
            logger.record(f"oracle_{n}/gap", perturb_gap[n], i)
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
