import tensorflow as tf
import tensorflow.keras.layers as tfkl


class DenseForwardModel(tf.keras.Model):

    def __init__(self,
                 input_sizes,
                 latent_size=32,
                 hidden_size=256,
                 output_size=1):
        super(DenseForwardModel, self).__init__()
        self.input_sizes = input_sizes
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.first = []
        for size in input_sizes:
            self.first.append(
                tfkl.Dense(latent_size, input_shape=(size,)))

        self.act = tfkl.LeakyReLU(alpha=0.2)

        self.h0 = tfkl.Dense(
            hidden_size, input_shape=(
                latent_size * len(input_sizes),))

        self.h1 = tfkl.Dense(
            hidden_size, input_shape=(hidden_size,))
        self.h2 = tfkl.Dense(
            hidden_size, input_shape=(hidden_size,))
        self.h3 = tfkl.Dense(
            output_size, input_shape=(hidden_size,))

    def call(self, inputs, training=False, **kwargs):
        x = tf.concat([
            layer(xi, training=training, **kwargs)
            for xi, layer in zip(
                tf.split(inputs, self.input_sizes), self.first)], axis=1)
        x = self.h0(self.act(x, training=training, **kwargs),
                    training=training, **kwargs)
        x = self.h1(self.act(x, training=training, **kwargs),
                    training=training, **kwargs)
        x = self.h2(self.act(x, training=training, **kwargs),
                    training=training, **kwargs)
        x = self.h3(self.act(x, training=training, **kwargs),
                    training=training, **kwargs)
        return x
