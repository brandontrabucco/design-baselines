from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.cma_es.trainers import Ensemble
from design_baselines.cma_es.nets import ForwardModel
import tensorflow as tf
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
    if task.is_discrete:
        task.map_to_logits()

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
    x = initial_x

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
    solution = x

    # evaluate the found solution
    score = task.predict(solution)
    if task.is_normalized_y:
        score = task.denormalize_y(score)
    logger.record("score", score, 0, percentile=True)
