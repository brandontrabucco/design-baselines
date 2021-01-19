from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np


class ForwardModel(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    distribution = tfpd.Normal

    def __init__(self,
                 input_shape,
                 hidden=50,
                 initial_max_std=1.5,
                 initial_min_std=0.5):
        """Create a fully connected architecture using keras that can process
        designs and predict a gaussian distribution over scores

        Args:

        input_shape: List[int]
            the shape of a single tensor input
        hidden: int
            the global hidden size of the neural network
        initial_max_std: float
            the starting upper bound of the standard deviation
        initial_min_std: float
            the starting lower bound of the standard deviation
        """

        self.max_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_max_std).astype(np.float32)), trainable=True)
        self.min_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_min_std).astype(np.float32)), trainable=True)

        super(ForwardModel, self).__init__([
            tfkl.Flatten(input_shape=input_shape),
            tfkl.Dense(hidden),
            tfkl.LeakyReLU(),
            tfkl.Dense(2)])

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
        return {"loc": mean, "scale": tf.math.softplus(logstd)}

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


class ContinuousMarginal(tf.Module):

    distribution = tfpd.Normal

    def __init__(self,
                 initial_mean,
                 initial_logstd):
        """Create a trainable gaussian distribution with diagonal covariance

        Args:

        input_shape: List[int]
            the shape of a single tensor input
        """

        self.mean = tf.Variable(initial_mean, trainable=True)
        self.logstd = tf.Variable(initial_logstd, trainable=True)
        super(ContinuousMarginal, self).__init__()

    def get_params(self):
        """Return a dictionary of parameters for a particular distribution
        family such as the mean and variance of a gaussian

        Returns:

        parameters: dict
            a dictionary that contains 'loc' and 'scale_diag' keys
        """

        return {"loc": self.mean,
                "scale": tf.math.softplus(self.logstd)}

    def get_distribution(self):
        """Return a distribution over the outputs of this model, for example
        a Multivariate Gaussian Distribution

        Returns:

        distribution: tfp.distribution.Distribution
            a tensorflow probability distribution over outputs of the model
        """

        return self.distribution(**self.get_params())


class DiscreteMarginal(tf.Module):

    distribution = tfpd.OneHotCategorical

    def __init__(self,
                 initial_logits):
        """Create a trainable one hot categorical distribution

        Args:

        input_shape: List[int]
            the shape of a single tensor input
        """

        self.logits = tf.Variable(initial_logits, trainable=True)
        super(DiscreteMarginal, self).__init__()

    def get_params(self):
        """Return a dictionary of parameters for a particular distribution
        family such as the one hot categorical distribution

        Returns:

        parameters: dict
            a dictionary that contains 'logits' key
        """

        return {"logits": tf.math.log_softmax(self.logits)}

    def get_distribution(self):
        """Return a distribution over the outputs of this model, for example
        a one hot categorical distribution

        Returns:

        distribution: tfp.distribution.Distribution
            a tensorflow probability distribution over outputs of the model
        """

        return self.distribution(**self.get_params())
