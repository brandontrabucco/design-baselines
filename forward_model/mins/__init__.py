from forward_model.data import StaticGraphTask
from forward_model.logger import Logger
from forward_model.mins.trainers import Ensemble
from forward_model.mins.trainers import WeightedGAN
from forward_model.mins.nets import ForwardModel
from forward_model.mins.nets import Discriminator
from forward_model.mins.nets import DiscreteGenerator
from forward_model.mins.nets import ContinuousGenerator
import tensorflow as tf
import numpy as np
import os


def adaptive_temp_v2(scores_np):
    """Calculate an adaptive temperature value based on the
    statistics of the scores array

    Args:

    scores_np: np.ndarray
        an array that represents the vectorized scores per data point

    Returns:

    temp: np.ndarray
        the scalar 90th percentile of scores in the dataset
    """

    inverse_arr = scores_np
    max_score = inverse_arr.max()
    scores_new = inverse_arr - max_score
    quantile_ninety = np.quantile(scores_new, q=0.9)
    return np.abs(quantile_ninety)


def softmax(arr,
            temp=1.0):
    """Calculate the softmax using numpy by normalizing a vector
    to have entries that sum to one

    Args:

    arr: np.ndarray
        the array which will be normalized using a tempered softmax
    temp: float
        a temperature parameter for the softmax

    Returns:

    normalized: np.ndarray
        the normalized input array which sums to one
    """

    max_arr = arr.max()
    arr_new = arr - max_arr
    exp_arr = np.exp(arr_new / temp)
    return exp_arr / np.sum(exp_arr)


def get_weights(scores):
    """Calculate weights used for training a model inversion
    network with a per-sample reweighted objective

    Args:

    scores: np.ndarray
        scores which correspond to the value of data points in the dataset

    Returns:

    weights: np.ndarray
        an array with the same shape as scores that reweights samples
    """

    scores_np = scores[:, 0]
    hist, bin_edges = np.histogram(scores_np, bins=20)
    hist = hist / np.sum(hist)

    base_temp = adaptive_temp_v2(scores_np)
    softmin_prob = softmax(bin_edges[1:], temp=base_temp)

    provable_dist = softmin_prob * (hist / (hist + 1e-3))
    provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)

    bin_indices = np.digitize(scores_np, bin_edges[1:])
    hist_prob = hist[np.minimum(bin_indices, 19)]

    weights = provable_dist[
        np.minimum(bin_indices, 19)] / (hist_prob + 1e-7)
    weights = np.clip(weights, a_min=0.0, a_max=5.0)
    return weights.astype(np.float32)


def model_inversion(config):
    """Optimize a design problem score using the algorithm MINS
    otherwise known as Model Inversion Networks

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    logger = Logger(config['logging_dir'])

    # create the training task and logger
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    train_data, val_data = task.build(bootstraps=config['bootstraps'],
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
    ensemble = Ensemble(forward_models,
                        forward_model_optim=tf.keras.optimizers.Adam,
                        forward_model_lr=config['ensemble_lr'])

    # create a manager for saving algorithms state to the disk
    ensemble_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**ensemble.get_saveables()),
        directory=os.path.join(config['logging_dir'], 'ensemble'),
        max_to_keep=1)

    # train the model for an additional number of epochs
    ensemble_manager.restore_or_initialize()
    ensemble.launch(train_data,
                    val_data,
                    logger,
                    config['ensemble_epochs'])

    # pick the right architecture based on the task
    Generator = DiscreteGenerator if config[
        'is_discrete'] else ContinuousGenerator

    # build the neural network GAN components
    generator = Generator(task.input_shape, config['latent_size'],
                          hidden=config['hidden_size'])
    disc = Discriminator(task.input_shape,
                         hidden=config['hidden_size'])
    gan = WeightedGAN(generator, disc,
                      g_lr=config['generator_lr'],
                      g_beta_1=config['generator_beta_1'],
                      g_beta_2=config['generator_beta_2'],
                      d_lr=config['discriminator_lr'],
                      d_beta_1=config['discriminator_beta_1'],
                      d_beta_2=config['discriminator_beta_2'])

    # create a manager for saving algorithms state to the disk
    gan_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**gan.get_saveables()),
        directory=os.path.join(config['logging_dir'], 'gan'),
        max_to_keep=1)

    # build a weighted data set
    train_data, val_data = task.build(importance_weights=get_weights(task.y),
                                      batch_size=config['gan_batch_size'],
                                      val_size=config['val_size'])

    # train the initial vae fit to the original data distribution
    gan_manager.restore_or_initialize()
    gan.launch(train_data,
               val_data,
               logger,
               config['offline_epochs'])

    # save every model to the disk
    ensemble_manager.save()
    gan_manager.save()

    # sample designs from the GAN and evaluate them
    y = tf.tile(tf.reduce_max(
        task.y, keepdims=True), [config['solver_samples'], 1])
    logger.record("score", task.score(
        generator.sample(y)), config['offline_epochs'])
