from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.conservative.trainers import Conservative
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import os


class ForwardModel(tf.keras.Sequential):
    """A Fully Connected Network with 3 trainable layers"""

    def __init__(self,
                 input_shape,
                 hidden=2048,
                 act=tfkl.LeakyReLU):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        inp_size: int
            the size of the input vector of this network
        out_size: int
            the size of the output vector of this network
        hidden: int
            the global hidden size of the network
        act: function
            a function that returns an activation function such as tfkl.ReLU
        """

        super(ForwardModel, self).__init__([
            tfkl.Flatten(input_shape=input_shape),
            tfkl.Dense(hidden),
            act(),
            tfkl.Dense(hidden),
            act(),
            tfkl.Dense(1)])


def conservative(config):
    """Train a forward model and perform model based optimization
    using a conservative objective function

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, validate_data = task.build(batch_size=config['batch_size'],
                                           val_size=config['val_size'])
    logger = Logger(config['logging_dir'])

    # make a keras neural network with two hidden layers
    forward_model = ForwardModel(
        task.input_shape,
        hidden=config['hidden_size'],
        act=tfkl.LeakyReLU)

    # create a trainer for a forward model with a conservative objective
    trainer = Conservative(
        forward_model,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['forward_model_lr'],
        target_conservative_gap=config['target_conservative_gap'],
        initial_alpha=config['initial_alpha'],
        alpha_optim=tf.keras.optimizers.Adam,
        alpha_lr=config['alpha_lr'],
        perturbation_lr=config['perturbation_lr'],
        perturbation_steps=config['perturbation_steps'])

    # create a manager for saving algorithms state to the disk
    manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**trainer.get_saveables()),
        directory=os.path.join(config['logging_dir'], 'ckpt'),
        max_to_keep=1)

    # train the model for an additional number of epochs
    manager.restore_or_initialize()
    trainer.launch(train_data, validate_data, logger, config['epochs'])
    manager.save()

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(task.y[:, 0], k=config['solver_samples'])[1]

    # evaluate the initial design using the oracle and the forward model
    solution = tf.gather(task.x, indices, axis=0)
    score = task.score(solution)
    prediction = forward_model(solution)

    # record the prediction and score to the logger
    logger.record("score", score, 0)
    logger.record("prediction", prediction, 0)

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):

        # back propagate through the forward model
        with tf.GradientTape() as tape:
            tape.watch(solution)
            score = forward_model(solution)
        grads = tape.gradient(score, solution)
        solution = solution + config['solver_lr'] * grads

        # evaluate the design using the oracle and the forward model
        gradient_norm = tf.linalg.norm(grads, axis=1)
        score = task.score(solution)
        prediction = forward_model(solution)

        # record the prediction and score to the logger
        logger.record("gradient_norm", gradient_norm, i)
        logger.record("score", score, i)
        logger.record("prediction", prediction, i)
