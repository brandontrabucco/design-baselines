import tensorflow as tf
import tensorflow.keras.layers as tfkl


class ShallowFullyConnected(tf.keras.Model):
    """A Fully Connected Network with 5 trainable layers"""

    def __init__(self,
                 inp_size,
                 out_size,
                 hidden=2048,
                 act=tfkl.ReLU,
                 batch_norm=False):
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
        batch_norm: bool
            whether to use batch normalization or remove it
        """

        super(ShallowFullyConnected, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size

        self.module = tf.keras.Sequential([
            tfkl.Dense(hidden, input_shape=(inp_size,)),
            tfkl.BatchNormalization(), act(),
            tfkl.Dense(hidden), tfkl.BatchNormalization(), act(),
            tfkl.Dense(out_size)]) if batch_norm else tf.keras.Sequential([
                tfkl.Dense(hidden, input_shape=(inp_size,)), act(),
                tfkl.Dense(hidden), act(),
                tfkl.Dense(out_size)])

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        """Pass a vector input through the network that is composed of
        several streams of inputs and outputs

        Args:

        inputs: tf.Tensor
            a tensor that has a batch axis and a channels axis
        training: bool
            whether the network is in training of evaluation mode

        Returns:

        outputs: tf.Tensor
            a tensor that has a batch axis and a channels axis
        """

        return self.module(inputs, training=training, **kwargs)


class FullyConnected(tf.keras.Model):
    """A Fully Connected Network with 5 trainable layers"""

    def __init__(self,
                 inp_sizes,
                 out_sizes,
                 merged=32,
                 hidden=256,
                 act=tfkl.ReLU,
                 batch_norm=True):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        inp_sizes: List[int]
            a list of sizes for every parallel input stream
        out_sizes: List[int]
            a list of sizes for every parallel output stream
        merged: int
            the hidden size before streams are concatenated or split
        hidden: int
            the global hidden size of the network
        act: function
            a function that returns an activation function such as tfkl.ReLU
        batch_norm: bool
            whether to use batch normalization or remove it
        """

        super(FullyConnected, self).__init__()
        self.inp_sizes = inp_sizes
        self.out_sizes = out_sizes
        inp_size = merged * len(inp_sizes)
        out_size = merged * len(out_sizes)

        self.inp = []
        for size in inp_sizes:
            self.inp.append(tfkl.Dense(merged))
            self.inp[-1].build((size,))
        self.out = []
        for size in out_sizes:
            self.out.append(tfkl.Dense(size))
            self.out[-1].build((merged,))

        self.module = tf.keras.Sequential([
            tfkl.BatchNormalization(input_shape=(inp_size,)),
            act(), tfkl.Dense(hidden), tfkl.BatchNormalization(),
            act(), tfkl.Dense(hidden), tfkl.BatchNormalization(),
            act(), tfkl.Dense(out_size), tfkl.BatchNormalization(),
            act()]) if batch_norm else tf.keras.Sequential([
                act(), tfkl.Dense(hidden, input_shape=(inp_size,)),
                act(), tfkl.Dense(hidden),
                act(), tfkl.Dense(out_size), act()])

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        """Pass a vector input through the network that is composed of
        several streams of inputs and outputs

        Args:

        inputs: tf.Tensor
            a tensor that has a batch axis and a channels axis
        training: bool
            whether the network is in training of evaluation mode

        Returns:

        outputs: tf.Tensor
            a tensor that has a batch axis and a channels axis
        """

        x = tf.concat([
            layer(xi, training=training, **kwargs)
            for xi, layer in zip(
                tf.split(inputs, self.inp_sizes, axis=1), self.inp)], axis=1)
        x = self.module(x, training=training, **kwargs)
        return tf.concat([
            layer(xi, training=training, **kwargs)
            for xi, layer in zip(
                tf.split(x, len(self.out_sizes), axis=1), self.out)], axis=1)
