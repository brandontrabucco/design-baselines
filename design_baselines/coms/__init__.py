from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.coms.trainers import ConservativeObjectiveModel
from design_baselines.coms.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
import os


def coms(config):
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

    if config['is_discrete']:

        # clip the distribution probabilities to a max of discrete_clip
        p = np.full_like(x, 1 / float(x.shape[-1]))
        discrete_clip = config.get('discrete_clip', 5.0)
        x = discrete_clip * x + (1.0 - discrete_clip) * p

        # map the distribution probabilities to logits
        x = np.log(x)
        x = x[:, :, 1:] - x[:, :, :1]

    if config['normalize_ys']:

        # remove the mean from the score values
        mu_y = np.mean(y, axis=0,
                       keepdims=True).astype(np.float32)
        y = y - mu_y

        # standardize the variance of the score values
        st_y = np.std(y, axis=0,
                      keepdims=True).astype(np.float32).clip(1e-6, 1e9)
        y = y / st_y

    else:

        # create placeholder normalization statistics
        mu_y = 0.0
        st_y = 1.0

    if config['normalize_xs']:

        # remove the mean from the data vectors
        mu_x = np.mean(x, axis=0,
                       keepdims=True).astype(np.float32)
        x = x - mu_x

        # standardize the variance of the data vectors
        st_x = np.std(x, axis=0,
                      keepdims=True).astype(np.float32).clip(1e-6, 1e9)
        x = x / st_x

    else:

        # create placeholder normalization statistics
        mu_x = 0.0
        st_x = 1.0

    # record the inputs shape of the forward model
    input_shape = list(task.input_shape)
    if config['is_discrete']:
        input_shape[-1] = input_shape[-1] - 1

    # compute the normalized learning rate of the model
    inner_lr = config['inner_lr'] * np.sqrt(np.prod(input_shape))
    outer_lr = config['outer_lr'] * np.sqrt(np.prod(input_shape))

    # make a neural network to predict scores
    forward_model = ForwardModel(
        input_shape, activations=config['activations'],
        hidden=config['hidden'],
        max_std=config['max_std'], min_std=config['min_std'])

    # make a trainer for the forward model
    trainer = ConservativeObjectiveModel(
        forward_model, forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        initial_alpha=config['initial_alpha'],
        alpha_opt=tf.keras.optimizers.Adam, alpha_lr=config['alpha_lr'],
        target_conservatism=config['target_conservatism'],
        inner_lr=inner_lr, outer_lr=outer_lr,
        inner_gradient_steps=config['inner_gradient_steps'],
        outer_gradient_steps=config['outer_gradient_steps'],
        beta=config['beta'],
        continuous_noise_std=config['continuous_noise_std'])

    # create a data set
    train_data, validate_data = task.build(
        x=x, y=y, batch_size=config['batch_size'],
        val_size=config['val_size'])

    # train the forward model
    trainer.launch(train_data,
                   validate_data,
                   logger,
                   config["epochs"])

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=config['batch_size'])[1]
    initial_x = tf.gather(x, indices, axis=0)
    xt = initial_x

    scores = []
    predictions = []

    for step in range(4 * config['outer_gradient_steps']):

        xt = trainer.outer_optimize(xt, 1, training=False)
        prediction = forward_model(
            xt, training=False).mean().numpy()

        next_xt = trainer.inner_optimize(xt, training=False)
        next_prediction = forward_model(
            next_xt, training=False).mean().numpy()

        final_xt = trainer.outer_optimize(
            xt, config['outer_gradient_steps'], training=False)
        final_prediction = forward_model(
            final_xt, training=False).mean().numpy()

        # record the prediction and score to the logger
        logger.record("solver/distance",
                      tf.linalg.norm(xt - initial_x), step)
        logger.record(f"solver/prediction",
                      prediction, step)
        logger.record(f"solver/beta_conservatism",
                      prediction - config["beta"] * next_prediction, step)
        logger.record(f"solver/conservatism",
                      prediction - final_prediction, step)

        solution = xt * st_x + mu_x
        if config['is_discrete']:
            solution = tf.math.softmax(tf.pad(
                solution, [[0, 0], [0, 0], [1, 0]]) / 0.001)

        score = task.score(solution)
        logger.record("score", score, step, percentile=True)
        logger.record(f"solver/model_to_real",
                      spearman(prediction[:, 0], score[:, 0]), step)

        scores.append(score)
        predictions.append(prediction)

    # save the model predictions and scores to be aggregated later
    np.save(os.path.join(config['logging_dir'], "scores.npy"),
            np.concatenate(scores, axis=1))
    np.save(os.path.join(config['logging_dir'], "predictions.npy"),
            np.stack(predictions, axis=1))
