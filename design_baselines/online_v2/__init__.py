from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import gumb_noise
from design_baselines.online_v2.trainers import OnlineSolver
from design_baselines.online_v2.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf


def online_ensemble(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(task.y[:, 0], k=config['batch_size'])[1]
    x = tf.gather(task.x, indices, axis=0)
    x = tf.math.log(gumb_noise(
        x, config.get('keep', 0.001), config.get('temp', 0.001))) \
        if config['is_discrete'] else x

    # make several keras neural networks with different architectures
    forward_model = ForwardModel(
        task.input_shape,
        activations=config['activations'],
        hidden=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])

    # create a trainer for a forward model with a conservative objective
    trainer = OnlineSolver(
        forward_model,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        target_conservative_gap=config['target_conservative_gap'],
        initial_alpha=config['initial_alpha'],
        alpha_optim=tf.keras.optimizers.Adam,
        alpha_lr=config['alpha_lr'],
        lookahead_lr=config['lookahead_lr'],
        lookahead_steps=config['lookahead_steps'],
        lookahead_backprop=config['lookahead_backprop'],
        lookahead_swap=config['lookahead_swap'],
        solver_period=config['solver_period'],
        solver_warmup=config['solver_warmup'],
        is_discrete=config['is_discrete'],
        noise_std=config.get('noise_std', 0.0),
        keep=config.get('keep', 1.0),
        temp=config.get('temp', 0.001))

    trainer.soln = tf.Variable(x)

    # create a bootstrapped data set
    train_data, validate_data = task.build(
        batch_size=config['batch_size'],
        val_size=config['val_size'],
        bootstraps=1)

    # optimize the solution online
    iteration = -1
    for epoch in range(config['epochs']):

        for x, y, b in train_data:
            iteration += 1
            i = tf.convert_to_tensor(iteration)

            for name, tensor in trainer.train_step(i, x, y, b).items():
                logger.record(f"oracle_{0}/" + name, tensor, iteration)

            if tf.logical_and(
                    tf.greater_equal(i, config['solver_warmup']),
                    tf.equal(tf.math.floormod(i, config['solver_period']), 0)):

                # evaluate the current solution
                solution = tf.math.softmax(trainer.soln) \
                    if config['is_discrete'] else trainer.soln

                # evaluate the design using the oracle and the forward model
                score = task.score(solution)
                preds = forward_model.get_distribution(solution).mean()

                # evaluate the current solution
                logger.record("score", score, iteration, percentile=True)
                logger.record(f"oracle_{0}/prediction", preds, iteration)
                logger.record(f"rank_corr/{0}_to_real",
                              spearman(preds[:, 0], score[:, 0]), iteration)

        statistics = defaultdict(list)
        for x, y in validate_data:
            for name, tensor in trainer.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            logger.record(f"oracle_{0}/" + name,
                          tf.concat(statistics[name], axis=0),
                          iteration)
