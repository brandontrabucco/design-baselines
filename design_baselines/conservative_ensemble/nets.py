from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np


class ForwardModel(tf.keras.Sequential):
    """A Fully Connected Network with 3 trainable layers"""

    distribution = tfpd.MultivariateNormalDiag

    def __init__(self,
                 input_shape,
                 hidden=2048,
                 initial_max_std=1.5,
                 initial_min_std=0.5,
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

        self.max_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_max_std).astype(np.float32)), trainable=True)
        self.min_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_min_std).astype(np.float32)), trainable=True)

        p = hidden // 4

        super(ForwardModel, self).__init__([
            tfkl.Flatten(input_shape=input_shape),
            tfkl.Dense(hidden + np.random.randint(p * 2) - p),
            act(),
            tfkl.Dense(hidden + np.random.randint(p * 2) - p),
            act(),
            tfkl.Dense(2)])

    def get_parameters(self, inputs, **kwargs):
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
        return {"loc": mean, "scale_diag": tf.exp(logstd)}

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

        return self.distribution(**self.get_parameters(inputs, **kwargs))
