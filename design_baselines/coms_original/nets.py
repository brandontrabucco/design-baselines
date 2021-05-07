import tensorflow.keras.layers as tfkl
import tensorflow as tf


class TanhMultiplier(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(TanhMultiplier, self).__init__(**kwargs)
        w_init = tf.constant_initializer(1.0)
        self.multiplier = tf.Variable(initial_value=w_init(
            shape=(1,), dtype="float32"), trainable=True)

    def call(self, inputs, **kwargs):
        exp_multiplier = tf.math.exp(self.multiplier)
        return tf.math.tanh(inputs / exp_multiplier) * exp_multiplier


def ForwardModel(input_shape,
                 activations=('relu', 'relu'),
                 hidden=2048,
                 final_tanh=False):
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

    activations = [tfkl.LeakyReLU if act == 'leaky_relu' else
                   tfkl.Activation(tf.math.cos) if act == 'cos' else
                   act for act in activations]

    layers = [tfkl.Flatten(input_shape=input_shape)]
    for act in activations:
        layers.extend([tfkl.Dense(hidden), tfkl.Activation(act)
                       if isinstance(act, str) else act()])
    layers.extend([tfkl.Dense(1)])
    if final_tanh:
        layers.extend([TanhMultiplier()])
    return tf.keras.Sequential(layers)
