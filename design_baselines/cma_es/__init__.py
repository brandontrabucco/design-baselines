from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import soft_noise
from design_baselines.cma_es.trainers import Ensemble
from design_baselines.cma_es.nets import ForwardModel
from design_baselines.utils import render_video
import tensorflow as tf
import numpy as np
import os


def cma_es(config):
    """Optimizes over designs x in an offline optimization problem
    using the CMA Evolution Strategy

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
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

        # compute normalization statistics for the data vectors
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

        # compute normalization statistics for the data vectors
        mu_x = np.zeros_like(x[:1])
        st_x = np.ones_like(x[:1])

    # create the training task and logger
    train_data, val_data = task.build(
        x=x, y=y, bootstraps=config['bootstraps'],
        batch_size=config['ensemble_batch_size'],
        val_size=config['val_size'])

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        task.input_shape,
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for b in range(config['bootstraps'])]

    # create a trainer for a forward model with a conservative objective
    ensemble = Ensemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['ensemble_lr'],
        is_discrete=config['is_discrete'],
        continuous_noise_std=config.get('continuous_noise_std', 0.0),
        discrete_keep=config.get('discrete_keep', 1.0))

    # create a manager for saving algorithms state to the disk
    ensemble_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**ensemble.get_saveables()),
        os.path.join(config['logging_dir'], 'ensemble'), 1)

    # train the model for an additional number of epochs
    ensemble_manager.restore_or_initialize()
    ensemble.launch(train_data,
                    val_data,
                    logger,
                    config['ensemble_epochs'])

    # select the top 1 initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=config['solver_samples'])[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = tf.math.log(soft_noise(initial_x,
                               config.get('discrete_smoothing', 0.6))) \
        if config['is_discrete'] else initial_x

    # create a fitness function for optimizing the expected task score
    def fitness(input_x):
        input_x = tf.reshape(input_x, task.input_shape)[tf.newaxis]
        return (-ensemble.get_distribution(
            input_x).mean()[0].numpy()).tolist()[0]

    import cma
    result = []
    for i in range(config['solver_samples']):
        xi = x[i].numpy().flatten().tolist()
        es = cma.CMAEvolutionStrategy(xi, config['cma_sigma'])
        step = 0
        while not es.stop() and step < config['cma_max_iterations']:
            solutions = es.ask()
            es.tell(solutions, [fitness(x) for x in solutions])
            step += 1
        result.append(
            tf.reshape(es.result.xbest, task.input_shape))
        print(f"CMA: {i + 1} / {config['solver_samples']}")

    # convert the solution found by CMA-ES to a tensor
    x = tf.stack(result, axis=0)
    solution = tf.math.softmax(x) if config['is_discrete'] else x

    # evaluate the found solution and record a video
    score = task.score(solution * st_x + mu_x)
    logger.record("score", score, 0, percentile=True)

    # render a video of the best solution found at the end
    render_video(config, task, (
        solution * st_x + mu_x)[np.argmax(np.reshape(score, [-1]))])
