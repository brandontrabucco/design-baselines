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


class Discriminator(tf.keras.Model):
    """A Fully Connected Network conditioned on a score"""

    def __init__(self,
                 design_shape,
                 hidden=50):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        design_shape: List[int]
            a list of tuple of integers that represents the shape of a
            single design for a particular task
        hidden: int
            the number of hidden units in every layer of the
            discriminator neural network
        """

        super(Discriminator, self).__init__()
        self.design_shape = design_shape

        # define a layer of the neural net with two pathways
        self.score_0 = tfkl.Dense(hidden)
        self.score_0.build((1,))
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((np.prod(design_shape),))
        self.bn_0 = tfkl.BatchNormalization()

        # define a layer of the neural net with two pathways
        self.score_1 = tfkl.Dense(hidden)
        self.score_1.build((1,))
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((hidden,))
        self.bn_1 = tfkl.BatchNormalization()

        # define a layer of the neural net with two pathways
        self.score_2 = tfkl.Dense(hidden)
        self.score_2.build((1,))
        self.dense_2 = tfkl.Dense(hidden)
        self.dense_2.build((hidden,))
        self.bn_2 = tfkl.BatchNormalization()

        # define a layer of the neural net with two pathways
        self.score_3 = tfkl.Dense(1)
        self.score_3.build((1,))
        self.dense_3 = tfkl.Dense(1)
        self.dense_3.build((hidden,))

    def penalty(self, h, y, **kwargs):
        """Calculate a gradient penalty for the discriminator and return
        a loss that will be minimized

        Args:

        X: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design
        y: tf.Tensor
            a batch of scalar scores wherein the generator is trained to
            produce designs that have score y

        Args:

        penalty: tf.Tensor
            a tensor that represents the penalty for gradients of the
            discriminator for regularizing the discriminator
        """

        h = tf.cast(h, tf.float32)
        y = tf.cast(y, tf.float32)
        h = tf.reshape(h, [tf.shape(y)[0], np.prod(self.design_shape)])
        with tf.GradientTape() as tape:
            tape.watch(h)
            x = self.dense_0(h, **kwargs) * self.score_0(y, **kwargs)
            x = tf.nn.leaky_relu(self.bn_0(x, **kwargs), alpha=0.2)
            x = self.dense_1(x, **kwargs) * self.score_1(y, **kwargs)
            x = tf.nn.leaky_relu(self.bn_1(x, **kwargs), alpha=0.2)
            x = self.dense_2(x, **kwargs) * self.score_2(y, **kwargs)
            x = tf.nn.leaky_relu(self.bn_2(x, **kwargs), alpha=0.2)
            x = self.dense_3(x, **kwargs) * self.score_3(y, **kwargs)
        grads = tape.gradient(x, h)
        norm = tf.linalg.norm(grads, axis=-1, keepdims=True)
        return (1. - norm) ** 2

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

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        x = tf.reshape(x, [tf.shape(y)[0], np.prod(self.design_shape)])
        x = self.dense_0(x, **kwargs) * self.score_0(y, **kwargs)
        x = tf.nn.leaky_relu(self.bn_0(x, **kwargs), alpha=0.2)
        x = self.dense_1(x, **kwargs) * self.score_1(y, **kwargs)
        x = tf.nn.leaky_relu(self.bn_1(x, **kwargs), alpha=0.2)
        x = self.dense_2(x, **kwargs) * self.score_2(y, **kwargs)
        x = tf.nn.leaky_relu(self.bn_2(x, **kwargs), alpha=0.2)
        x = self.dense_3(x, **kwargs) * self.score_3(y, **kwargs)
        return (x - (1.0 if real else 0.0)) ** 2


class DiscreteGenerator(tf.keras.Model):
    """A Fully Connected Network conditioned on a score"""

    def __init__(self,
                 design_shape,
                 latent_size,
                 hidden=50):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        design_shape: List[int]
            a list of tuple of integers that represents the shape of a
            float design for a particular task
        latent_size: int
            the number of hidden units in the latent space used to
            condition the neural network generator
        hidden: int
            the number of hidden units in every layer of the
            discriminator neural network
        """

        super(DiscreteGenerator, self).__init__()
        self.design_shape = design_shape
        self.latent_size = latent_size

        # define a layer of the neural net with two pathways
        self.score_0 = tfkl.Dense(hidden)
        self.score_0.build((1,))
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((latent_size,))
        self.bn_0 = tfkl.BatchNormalization()

        # define a layer of the neural net with two pathways
        self.score_1 = tfkl.Dense(hidden)
        self.score_1.build((1,))
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((hidden,))
        self.bn_1 = tfkl.BatchNormalization()

        # define a layer of the neural net with two pathways
        self.score_2 = tfkl.Dense(hidden)
        self.score_2.build((1,))
        self.dense_2 = tfkl.Dense(hidden)
        self.dense_2.build((hidden,))
        self.bn_2 = tfkl.BatchNormalization()

        # define a layer of the neural net with two pathways
        self.score_3 = tfkl.Dense(np.prod(design_shape))
        self.score_3.build((1,))
        self.dense_3 = tfkl.Dense(np.prod(design_shape))
        self.dense_3.build((hidden,))

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

        temp = kwargs.pop("temp", 1.0)
        z = tf.random.normal([tf.shape(y)[0], self.latent_size])
        x = tf.cast(z, tf.float32)
        y = tf.cast(y, tf.float32)
        x = self.dense_0(x, **kwargs) * self.score_0(y, **kwargs)
        x = tf.nn.relu(self.bn_0(x, **kwargs))
        x = self.dense_1(x, **kwargs) * self.score_1(y, **kwargs)
        x = tf.nn.relu(self.bn_1(x, **kwargs))
        x = self.dense_2(x, **kwargs) * self.score_2(y, **kwargs)
        x = tf.nn.relu(self.bn_2(x, **kwargs))
        x = self.dense_3(x, **kwargs) * self.score_3(y, **kwargs)
        logits = tf.reshape(x, [tf.shape(y)[0], *self.design_shape])
        return tfpd.RelaxedOneHotCategorical(
            temp, logits=tf.math.log_softmax(logits)).sample()


class ContinuousGenerator(tf.keras.Model):
    """A Fully Connected Network conditioned on a score"""

    def __init__(self,
                 design_shape,
                 latent_size,
                 hidden=50):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        design_shape: List[int]
            a list of tuple of integers that represents the shape of a
            single design for a particular task
        latent_size: int
            the number of hidden units in the latent space used to
            condition the neural network generator
        hidden: int
            the number of hidden units in every layer of the
            discriminator neural network
        """

        super(ContinuousGenerator, self).__init__()
        self.design_shape = design_shape
        self.latent_size = latent_size

        # define a layer of the neural net with two pathways
        self.score_0 = tfkl.Dense(hidden)
        self.score_0.build((1,))
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((latent_size,))
        self.bn_0 = tfkl.BatchNormalization()

        # define a layer of the neural net with two pathways
        self.score_1 = tfkl.Dense(hidden)
        self.score_1.build((1,))
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((hidden,))
        self.bn_1 = tfkl.BatchNormalization()

        # define a layer of the neural net with two pathways
        self.score_2 = tfkl.Dense(hidden)
        self.score_2.build((1,))
        self.dense_2 = tfkl.Dense(hidden)
        self.dense_2.build((hidden,))
        self.bn_2 = tfkl.BatchNormalization()

        # define a layer of the neural net with two pathways
        self.score_3 = tfkl.Dense(np.prod(design_shape))
        self.score_3.build((1,))
        self.dense_3 = tfkl.Dense(np.prod(design_shape))
        self.dense_3.build((hidden,))

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

        kwargs.pop("temp", 1.0)
        z = tf.random.normal([tf.shape(y)[0], self.latent_size])
        x = tf.cast(z, tf.float32)
        y = tf.cast(y, tf.float32)
        x = self.dense_0(x, **kwargs) * self.score_0(y, **kwargs)
        x = tf.nn.relu(self.bn_0(x, **kwargs))
        x = self.dense_1(x, **kwargs) * self.score_1(y, **kwargs)
        x = tf.nn.relu(self.bn_1(x, **kwargs))
        x = self.dense_2(x, **kwargs) * self.score_2(y, **kwargs)
        x = tf.nn.relu(self.bn_2(x, **kwargs))
        x = self.dense_3(x, **kwargs) * self.score_3(y, **kwargs)
        return tf.reshape(x, [tf.shape(y)[0], *self.design_shape])
