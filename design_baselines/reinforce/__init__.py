from design_baselines.data import StaticGraphTask, build_pipeline
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
    using the REINFORCE policy gradient method

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    if task.is_discrete:
        task.map_to_integers()

    if config['normalize_ys']:
        task.map_normalize_y()
    if config['normalize_xs']:
        task.map_normalize_x()

    x = task.x
    y = task.y

    # create the training task and logger
    train_data, val_data = build_pipeline(
        x=x, y=y, bootstraps=config['bootstraps'],
        batch_size=config['ensemble_batch_size'],
        val_size=config['val_size'])

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        task,
        embedding_size=config['embedding_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for b in range(config['bootstraps'])]

    # create a trainer for a forward model with a conservative objective
    ensemble = Ensemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['ensemble_lr'])

    # train the model for an additional number of epochs
    ensemble.launch(train_data,
                    val_data,
                    logger,
                    config['ensemble_epochs'])

    rl_opt = tf.keras.optimizers.Adam(
        learning_rate=config['reinforce_lr'])

    # select the top 1 initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=config['solver_samples'])[1]
    initial_x = tf.gather(x, indices, axis=0)

    if task.is_discrete:
        logits = tf.pad(task.to_logits(initial_x), [[0, 0], [0, 0], [1, 0]])
        probs = tf.math.softmax(logits / 1e-5)
        logits = tf.math.log(tf.reduce_mean(probs, axis=0))
        sampler = DiscreteMarginal(logits)

    else:
        mean = tf.reduce_mean(initial_x, axis=0)
        logstd = tf.math.log(tf.ones_like(mean) * config['exploration_std'])
        sampler = ContinuousMarginal(mean, logstd)

    for iteration in range(config['iterations']):

        with tf.GradientTape() as tape:
            td = sampler.get_distribution()
            tx = td.sample(sample_shape=config['reinforce_batch_size'])
            if config['optimize_ground_truth']:
                ty = task.predict(tx)
            else:  # use the surrogate model for optimization
                ty = ensemble.get_distribution(tx).mean()

            mean_y = tf.reduce_mean(ty)
            standard_dev_y = tf.math.reduce_std(ty - mean_y)
            log_probs = td.log_prob(tf.stop_gradient(tx))
            loss = tf.reduce_mean(-log_probs[:, tf.newaxis] *
                                  tf.stop_gradient(
                                      (ty - mean_y) / standard_dev_y))

        print(f"[Iteration {iteration}] "
              f"Average Prediction = {tf.reduce_mean(ty)}")

        logger.record("reinforce/prediction",
                      ty, iteration, percentile=True)
        logger.record("reinforce/loss",
                      loss, iteration, percentile=True)

        grads = tape.gradient(
            loss, sampler.trainable_variables)

        rl_opt.apply_gradients(zip(
            grads, sampler.trainable_variables))

    td = sampler.get_distribution()
    solution = td.sample(sample_shape=config['solver_samples'])

    # save the current solution to the disk
    np.save(os.path.join(config["logging_dir"],
                         f"solution.npy"), solution.numpy())
    if config["do_evaluation"]:

        # evaluate the found solution and record a video
        score = task.predict(solution)
        if config['normalize_ys']:
            score = task.denormalize_y(score)
        logger.record(
            "score", score, config['iterations'], percentile=True)
