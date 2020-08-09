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


class Discriminator(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    def __init__(self,
                 design_shape,
                 hidden=50):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        inp_size: int
            the size of the input vector of this network
        out_size: int
            the size of the output vector of this network
        hidden: int
            the global hidden size of the network
        """

        self.design_shape = design_shape
        super(Discriminator, self).__init__([
            tfkl.Dense(hidden, input_shape=(np.prod(design_shape) + 1,)),
            tfkl.LeakyReLU(),
            tfkl.Dense(1)])

    def loss(self, x, y, real=True, **kwargs):
        """Use a neural network to discriminate the log probability that a
        sampled design X has score y

        Args:

        X: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design
        y: tf.Tensor
            a batch of scalar scores wherein the generator is trained to
            produce designs that have score y
        real: bool
            a boolean that whether the probability is taken with
            respect to X being real of fake

        Args:

        log_p: tf.Tensor
            a tensor that represents the log probability of either X being
            real of X being fake depending on the value of 'real'
        """

        x = tf.reshape(x, [tf.shape(y)[0], np.prod(self.design_shape)])
        inputs = tf.concat([x, y], 1)
        mu = super(Discriminator, self).__call__(inputs, **kwargs)
        return (mu - 1.0 if real else 0.0) ** 2


class DiscreteGenerator(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    def __init__(self,
                 design_shape,
                 latent_size,
                 hidden=50):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        inp_size: int
            the size of the input vector of this network
        out_size: int
            the size of the output vector of this network
        hidden: int
            the global hidden size of the network
        """

        self.latent_size = latent_size
        super(DiscreteGenerator, self).__init__([
            tfkl.Dense(hidden, input_shape=(latent_size + 1,)),
            tfkl.LeakyReLU(),
            tfkl.Dense(np.prod(design_shape)),
            tfkl.Reshape(design_shape),
            tf.keras.layers.Softmax(axis=-1)])

    def sample(self, y, **kwargs):
        """Generate samples of designs X that have a score y where y is
        the score that the generator conditions on

        Args:

        y: tf.Tensor
            a batch of scalar scores wherein the generator is trained to
            produce designs that have score y

        Returns:

        x_fake: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design
        """

        z = tf.random.normal([tf.shape(y)[0], self.latent_size])
        inputs = tf.concat([z, y], 1)
        return super(DiscreteGenerator, self).__call__(inputs, **kwargs)


class ContinuousGenerator(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    def __init__(self,
                 design_shape,
                 latent_size,
                 hidden=50):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        inp_size: int
            the size of the input vector of this network
        out_size: int
            the size of the output vector of this network
        hidden: int
            the global hidden size of the network
        """

        self.latent_size = latent_size
        super(ContinuousGenerator, self).__init__([
            tfkl.Dense(hidden, input_shape=(latent_size + 1,)),
            tfkl.LeakyReLU(),
            tfkl.Dense(np.prod(design_shape) * 2),
            tfkl.Reshape(design_shape)])

    def sample(self, y, **kwargs):
        """Generate samples of designs X that have a score y where y is
        the score that the generator conditions on

        Args:

        y: tf.Tensor
            a batch of scalar scores wherein the generator is trained to
            produce designs that have score y

        Returns:

        x_fake: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design
        """

        z = tf.random.normal([tf.shape(y)[0], self.latent_size])
        inputs = tf.concat([z, y], 1)
        return super(ContinuousGenerator, self).__call__(inputs, **kwargs)
