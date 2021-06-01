from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


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
                # up-sample the sequence and handle
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


class ForwardModel(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    distribution = tfpd.Normal

    def __init__(self, input_shape, hidden_size=50,
                 num_layers=1, initial_max_std=1.5, initial_min_std=0.5):
        """Create a fully connected architecture using keras that can process
        designs and predict a gaussian distribution over scores

        Args:

        task: StaticGraphTask
            a model-based optimization task
        embedding_size: int
            the size of the embedding matrix for discrete tasks
        hidden_size: int
            the global hidden size of the neural network
        num_layers: int
            the number of hidden layers
        initial_max_std: float
            the starting upper bound of the standard deviation
        initial_min_std: float
            the starting lower bound of the standard deviation

        """

        self.max_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_max_std).astype(np.float32)), trainable=True)
        self.min_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_min_std).astype(np.float32)), trainable=True)

        layers = []
        layers.append(tfkl.Flatten(input_shape=input_shape)
                      if len(layers) == 0 else tfkl.Flatten())
        for i in range(num_layers):
            layers.extend([tfkl.Dense(hidden_size), tfkl.LeakyReLU()])

        layers.append(tfkl.Dense(2))
        super(ForwardModel, self).__init__(layers)

    def get_params(self, inputs, **kwargs):
        """Return a dictionary of parameters for a particular distribution
        family such as the mean and variance of a gaussian

        Args:

        inputs: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        parameters: dict
            a dictionary that contains 'loc' and 'scale_diag' keys
        """

        prediction = super(ForwardModel, self).__call__(inputs, **kwargs)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale": tf.math.exp(logstd)}

    def get_distribution(self, inputs, **kwargs):
        """Return a distribution over the outputs of this model, for example
        a Multivariate Gaussian Distribution

        Args:

        inputs: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        distribution: tfp.distribution.Distribution
            a tensorflow probability distribution over outputs of the model
        """

        return self.distribution(**self.get_params(inputs, **kwargs))
