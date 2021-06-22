from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


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
                 hidden_size=2048,
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
                   act for act in activations]

    layers = [tfkl.Flatten(input_shape=input_shape)]
    for act in activations:
        layers.extend([tfkl.Dense(hidden_size), tfkl.Activation(act)
                       if isinstance(act, str) else act()])
    layers.extend([tfkl.Dense(1)])
    if final_tanh:
        layers.extend([TanhMultiplier()])
    return tf.keras.Sequential(layers)


class SequentialVAE(tf.keras.Model):

    def __init__(self, task, hidden_size=64, latent_size=256,
                 activation='relu', kernel_size=3, num_blocks=4):

        super(SequentialVAE, self).__init__()

        input_shape = task.input_shape
        shape_before_flat = [input_shape[0] // (
                2 ** (num_blocks - 1)), hidden_size]

        """

        DEFINE AN ENCODER MODEL THAT DOWNSAMPLES

        """

        # the input layer of a keras model
        x = input_layer = keras.Input(shape=input_shape)

        # build a model with an input layer and optional embedding
        x = tfkl.Embedding(task.num_classes, hidden_size)(x)

        # the exponent of a positional embedding
        inverse_frequency = 1.0 / (10000.0 ** (tf.range(
            0.0, hidden_size, 2.0) / hidden_size))[tf.newaxis]

        # calculate a positional embedding to break symmetry
        pos = tf.range(0.0, tf.shape(x)[1], 1.0)[:, tf.newaxis]
        positional_embedding = tf.concat([
            tf.math.sin(pos * inverse_frequency),
            tf.math.cos(pos * inverse_frequency)], axis=1)[tf.newaxis]

        # add the positional encoding
        x = tfkl.Add()([x, positional_embedding])
        x = tfkl.LayerNormalization()(x)

        # add several residual blocks to the model
        for i in range(num_blocks):

            if i > 0:
                # downsample the input sequence by 2
                x = tf.keras.layers.AveragePooling1D(pool_size=2,
                                                     padding='same')(x)

            # first convolution layer in a residual block
            h = tfkl.Conv1D(hidden_size, kernel_size,
                            padding='same', activation=None)(x)
            h = tfkl.LayerNormalization()(h)
            h = tfkl.Activation(activation)(h)

            # second convolution layer in a residual block
            h = tfkl.Conv1D(hidden_size, kernel_size,
                            padding='same', activation=None)(h)
            h = tfkl.LayerNormalization()(h)
            h = tfkl.Activation(activation)(h)

            # add a residual connection to the model
            x = tfkl.Add()([x, h])

        # flatten the result and predict the params of a gaussian
        flattened_x = tfkl.Flatten()(x)
        latent_mean = tfkl.Dense(latent_size)(flattened_x)
        latent_standard_dev = tfkl.Dense(
            latent_size, activation=tf.exp)(flattened_x)

        # save the encoder as a keras model
        self.encoder_cnn = keras.Model(
            inputs=input_layer,
            outputs=[latent_mean, latent_standard_dev])

        """

        DEFINE A DECODER THAT UPSAMPLES

        """

        # the input layer of a keras model
        x = input_layer = keras.Input(shape=[latent_size])
        x = tfkl.Dense(np.prod(shape_before_flat))(x)
        x = tfkl.Reshape(shape_before_flat)(x)

        # add several residual blocks to the model
        for i in reversed(range(num_blocks)):

            if i > 0:
                # up-sample the sequence
                x = tf.pad(tf.repeat(x, 2, axis=1), [[0, 0], [
                    0,
                    (input_shape[0] // (2 ** (i - 1))) % 2
                ], [0, 0]], mode="SYMMETRIC")

            # the exponent of a positional embedding
            inverse_frequency = 1.0 / (10000.0 ** (tf.range(
                0.0, hidden_size, 2.0) / hidden_size))[tf.newaxis]

            # calculate a positional embedding to break symmetry
            pos = tf.range(0.0, tf.shape(x)[1], 1.0)[:, tf.newaxis]
            positional_embedding = tf.concat([
                tf.math.sin(pos * inverse_frequency),
                tf.math.cos(pos * inverse_frequency)], axis=1)[tf.newaxis]

            # add the positional encoding
            h = tfkl.Add()([x, positional_embedding])
            h = tfkl.LayerNormalization()(h)

            # first convolution layer in a residual block
            h = tfkl.Conv1D(hidden_size, kernel_size,
                            padding='same', activation=None)(h)
            h = tfkl.LayerNormalization()(h)
            h = tfkl.Activation(activation)(h)

            # second convolution layer in a residual block
            h = tfkl.Conv1D(hidden_size, kernel_size,
                            padding='same', activation=None)(h)
            h = tfkl.LayerNormalization()(h)
            h = tfkl.Activation(activation)(h)

            # add a residual connection to the model
            x = tfkl.Add()([x, h])

        # flatten the result and predict the params of a gaussian
        logits = tfkl.Dense(task.num_classes)(x)

        # save the encoder as a keras model
        self.decoder_cnn = keras.Model(
            inputs=input_layer, outputs=logits)

    def encode(self, x_batch, training=False):
        mean, standard_dev = self.encoder_cnn(x_batch, training=training)
        return tfpd.MultivariateNormalDiag(loc=mean, scale_diag=standard_dev)

    def decode(self, z, training=False):
        logits = self.decoder_cnn(z, training=training)
        return tfpd.Categorical(logits=logits)

    def generate(self, z, training=False):
        logits = self.decoder_cnn(z, training=training)
        return tf.argmax(logits, axis=2, output_type=tf.int32)
