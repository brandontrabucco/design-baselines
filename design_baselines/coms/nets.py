from tensorflow_probability import distributions as tfpd
from tensorflow_probability import layers as tfpl
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np


def ForwardModel(input_shape,
                 activations=('relu', 'relu'),
                 hidden=2048,
                 max_std=0.2,
                 min_std=0.1):
    """Creates a tensorflow model that outputs a probability distribution
    specifying the score corresponding to an input x.

    Args:

    input_shape: tuple[int]
        the shape of input tensors to the model
    activations: tuple[str]
        the name of activation functions for every hidden layer
    hidden: int
        the global hidden size of the network
    max_std: float
        the upper bound of the learned standard deviation
    min_std: float
        the lower bound of the learned standard deviation
    """

    max_log_std = np.log(max_std).astype(np.float32)
    min_log_std = np.log(min_std).astype(np.float32)

    def create_d(prediction):
        mean, log_std = tf.split(prediction, 2, axis=-1)
        log_std = max_log_std - tf.nn.softplus(max_log_std - log_std)
        log_std = min_log_std + tf.nn.softplus(log_std - min_log_std)
        return tfpd.Normal(loc=mean, scale=tf.math.exp(log_std))

    activations = [tfkl.LeakyReLU if act == 'leaky_relu' else
                   tfkl.Activation(tf.math.cos) if act == 'cos' else
                   act for act in activations]

    layers = [tfkl.Flatten(input_shape=input_shape)]
    for act in activations:
        layers.extend([tfkl.Dense(hidden), tfkl.Activation(act)
                       if isinstance(act, str) else act()])
    layers.extend([tfkl.Dense(2), tfpl.DistributionLambda(create_d)])
    return tf.keras.Sequential(layers)
