from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import add_gumbel_noise
from design_baselines.conservative_ensemble.trainers import Conservative
from design_baselines.conservative_ensemble.nets import ForwardModel
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os


def conservative_ensemble(config):
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

    trs = []
    for i, fm in enumerate(forward_models):

        # create a bootstrapped data set
        train_data, validate_data = task.build(batch_size=config['batch_size'],
                                               val_size=config['val_size'],
                                               bootstraps=1)

        # create a trainer for a forward model with a conservative objective
        trainer = Conservative(
            fm,
            forward_model_optim=tf.keras.optimizers.Adam,
            forward_model_lr=config['forward_model_lr'],
            target_conservative_gap=config['target_conservative_gap'],
            initial_alpha=config['initial_alpha'],
            alpha_optim=tf.keras.optimizers.Adam,
            alpha_lr=config['alpha_lr'],
            perturbation_lr=config['perturbation_lr'],
            perturbation_steps=config['perturbation_steps'],
            is_discrete=config['is_discrete'],
            noise_std=config.get('noise_std', 0.0),
            keep=config.get('keep', 1.0),
            temp=config.get('temp', 0.001))

        # train the model for an additional number of epochs
        trs.append(trainer)
        trainer.launch(train_data, validate_data, logger,
                       config['epochs'], header=f'oracle_{i}/')

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(task.y[:, 0], k=config['solver_samples'])[1]
    x = tf.gather(task.x, indices, axis=0)
    x = tf.math.log(add_gumbel_noise(
        x, config.get('keep', 0.001), config.get('temp', 0.001))) \
        if config['is_discrete'] else x

    # evaluate the starting point
    solution = tf.math.softmax(x) if config['is_discrete'] else x
    score = task.score(solution)
    preds = [fm.get_distribution(
        solution).mean() for fm in forward_models]

    # evaluate the conservative gap for every model
    perturb_preds = [tr.fm.get_distribution(
        tr.optimize(solution)).mean() for tr in trs]
    perturb_gap = [
        b - a for a, b in zip(preds, perturb_preds)]

    # record the prediction and score to the logger
    logger.record("score", score, 0, percentile=True)
    for n, prediction_i in enumerate(preds):
        logger.record(f"oracle_{n}/gap", perturb_gap[n], 0)
        logger.record(f"oracle_{n}/prediction", prediction_i, 0)
        logger.record(f"rank_corr/{n}_to_real",
                      spearman(prediction_i[:, 0], score[:, 0]), 0)
        if n > 0:
            logger.record(f"rank_corr/0_to_{n}",
                          spearman(preds[0][:, 0], prediction_i[:, 0]), 0)

    # and keep track of the best design sampled so far
    best_solution = None
    best_score = None

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
        score = task.score(solution)
        preds = [fm.get_distribution(
            solution).mean() for fm in forward_models]

        # evaluate the conservative gap for every model
        perturb_preds = [tr.fm.get_distribution(
            tr.optimize(solution)).mean() for tr in trs]
        perturb_gap = [
            b - a for a, b in zip(preds, perturb_preds)]

        # record the prediction and score to the logger
        logger.record("score", score, i, percentile=True)
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
