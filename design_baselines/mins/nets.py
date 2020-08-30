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
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((np.prod(design_shape) + 1,))
        self.bn_0 = tfkl.LayerNormalization()

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((hidden + 1,))
        self.bn_1 = tfkl.LayerNormalization()

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(1)
        self.dense_3.build((hidden + 1,))

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
            x = self.dense_0(tf.concat([h, y], 1), **kwargs)
            x = tf.nn.leaky_relu(self.bn_0(x, **kwargs), alpha=0.2)
            x = self.dense_1(tf.concat([x, y], 1), **kwargs)
            x = tf.nn.leaky_relu(self.bn_1(x, **kwargs), alpha=0.2)
            x = self.dense_3(tf.concat([x, y], 1), **kwargs)
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
        x = self.dense_0(tf.concat([x, y], 1), **kwargs)
        x = tf.nn.leaky_relu(self.bn_0(x, **kwargs), alpha=0.2)
        x = self.dense_1(tf.concat([x, y], 1), **kwargs)
        x = tf.nn.leaky_relu(self.bn_1(x, **kwargs), alpha=0.2)
        x = self.dense_3(tf.concat([x, y], 1), **kwargs)
        return (x - (1.0 if real else 0.0)) ** 2


class DiscriminatorConv(tf.keras.Model):
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

        super(DiscriminatorConv, self).__init__()
        shape = (None, *design_shape)

        # define a layer of the neural net with two pathways
        self.conv_0 = tfkl.Conv1D(32, 5, strides=2)
        shape = self.conv_0.compute_output_shape(shape)
        self.bn_0 = tfkl.LayerNormalization()
        shape = self.bn_0.compute_output_shape(shape)

        # define a layer of the neural net with two pathways
        self.conv_1 = tfkl.Conv1D(32, 5, strides=2)
        shape = self.conv_1.compute_output_shape(shape)
        self.bn_1 = tfkl.LayerNormalization()
        shape = self.bn_1.compute_output_shape(shape)

        # define a layer of the neural net with two pathways
        self.conv_2 = tfkl.Conv1D(32, 5, strides=2)
        shape = self.conv_2.compute_output_shape(shape)
        self.bn_2 = tfkl.LayerNormalization()
        shape = self.bn_2.compute_output_shape(shape)

        # compute the flat shape of the conv-dense connecting node
        self.flat = np.prod(shape[1:])
        self.input_size = np.prod(design_shape)

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(hidden)
        self.dense_3.compute_output_shape((None, self.flat + 1,))
        self.bn_3 = tfkl.LayerNormalization()
        self.dense_3.compute_output_shape((None, hidden,))

        # define a layer of the neural net with two pathways
        self.dense_4 = tfkl.Dense(1)
        self.dense_4.compute_output_shape((None, hidden + 1,))

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

        with tf.GradientTape() as tape:
            tape.watch(h)
            x = self.__call__(h, y, **kwargs)
        return (1.0 - tf.linalg.norm(tf.reshape(tape.gradient(
            x, h), [-1, self.input_size]), axis=-1, keepdims=True)) ** 2

    def __call__(self, x, y, **kwargs):
        """Use a neural network to discriminate the log probability that a
        sampled design X has score y

        Args:

        X: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design
        y: tf.Tensor
            a batch of scalar scores wherein the generator is trained to
            produce designs that have score y

        Args:

        log_p: tf.Tensor
            a tensor that represents the log probability of either X being
            real of X being fake depending on the value of 'real'
        """

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        x = self.conv_0(x, **kwargs)
        x = tf.nn.leaky_relu(self.bn_0(x, **kwargs), alpha=0.2)
        x = self.conv_1(x, **kwargs)
        x = tf.nn.leaky_relu(self.bn_1(x, **kwargs), alpha=0.2)
        x = self.conv_2(x, **kwargs)
        x = tf.nn.leaky_relu(self.bn_2(x, **kwargs), alpha=0.2)
        x = tf.reshape(x, [tf.shape(x)[0], self.flat])
        x = self.dense_3(tf.concat([x, y], 1), **kwargs)
        x = tf.nn.leaky_relu(self.bn_3(x, **kwargs), alpha=0.2)
        return self.dense_4(tf.concat([x, y], 1), **kwargs)

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

        # return the discriminator least squares loss function
        return (self.__call__(
            x, y, **kwargs) - (1.0 if real else 0.0)) ** 2


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
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((latent_size + 1,))
        self.bn_0 = tfkl.LayerNormalization()

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((hidden + 1,))
        self.bn_1 = tfkl.LayerNormalization()

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(np.prod(design_shape))
        self.dense_3.build((hidden + 1,))

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
        x = self.dense_0(tf.concat([x, y], 1), **kwargs)
        x = tf.nn.leaky_relu(self.bn_0(x, **kwargs), alpha=0.2)
        x = self.dense_1(tf.concat([x, y], 1), **kwargs)
        x = tf.nn.leaky_relu(self.bn_1(x, **kwargs), alpha=0.2)
        x = self.dense_3(tf.concat([x, y], 1), **kwargs)
        logits = tf.reshape(x, [tf.shape(y)[0], *self.design_shape])
        return tfpd.RelaxedOneHotCategorical(
            temp, logits=tf.math.log_softmax(logits)).sample()


class DiscreteGenConv(tf.keras.Model):
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
        hidden: int
            the number of hidden units in every layer of the
            discriminator neural network
        """

        super(DiscreteGenConv, self).__init__()
        self.latent_size = latent_size

        def get_len(size, i):
            return int(np.ceil((get_len(size, i - 1) - 4) / 2)) \
                   if i > 0 else size

        len0 = get_len(design_shape[0], 3)
        len1 = get_len(design_shape[0], 2)
        len2 = get_len(design_shape[0], 1)
        self.len0 = len0

        # define a layer of the neural net with two pathways
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.compute_output_shape((None, self.latent_size + 1,))
        self.bn_0 = tfkl.LayerNormalization()
        self.dense_0.compute_output_shape((None, hidden,))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(self.len0 * 32)
        self.dense_1.compute_output_shape((None, hidden + 1,))
        self.bn_1 = tfkl.LayerNormalization()
        self.dense_1.compute_output_shape((None, self.len0 * 32,))

        # define a layer of the neural net with two pathways
        self.conv_2 = tfkl.Conv1DTranspose(32, 5, strides=2)
        shape = self.conv_2.compute_output_shape((None, self.len0, 32))
        self.bn_2 = tfkl.LayerNormalization()
        self.bn_2.compute_output_shape(shape)

        # define a layer of the neural net with two pathways
        self.conv_3 = tfkl.Conv1DTranspose(32, 5, strides=2)
        shape = self.conv_3.compute_output_shape((None, len1, 32))
        self.bn_3 = tfkl.LayerNormalization()
        self.bn_3.compute_output_shape(shape)

        # define a layer of the neural net with two pathways
        self.conv_4 = tfkl.Conv1DTranspose(design_shape[-1], 5, strides=2)
        self.conv_4.compute_output_shape((None, len2, 32))

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
        x = self.dense_0(tf.concat([x, y], 1), **kwargs)
        x = tf.nn.leaky_relu(self.bn_0(x, **kwargs), alpha=0.2)
        x = self.dense_1(tf.concat([x, y], 1), **kwargs)
        x = tf.nn.leaky_relu(self.bn_1(x, **kwargs), alpha=0.2)
        x = tf.reshape(x, [tf.shape(y)[0], self.len0, 32])
        x = self.conv_2(x, **kwargs)
        x = tf.nn.leaky_relu(self.bn_2(x, **kwargs), alpha=0.2)
        x = self.conv_3(x, **kwargs)
        x = tf.nn.leaky_relu(self.bn_3(x, **kwargs), alpha=0.2)
        x = self.conv_4(x, **kwargs)
        return tfpd.RelaxedOneHotCategorical(
            temp, logits=tf.math.log_softmax(x)).sample()


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
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((latent_size + 1,))
        self.bn_0 = tfkl.LayerNormalization()

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((hidden + 1,))
        self.bn_1 = tfkl.LayerNormalization()

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(np.prod(design_shape))
        self.dense_3.build((hidden + 1,))

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
        x = self.dense_0(tf.concat([x, y], 1), **kwargs)
        x = tf.nn.leaky_relu(self.bn_0(x, **kwargs), alpha=0.2)
        x = self.dense_1(tf.concat([x, y], 1), **kwargs)
        x = tf.nn.leaky_relu(self.bn_1(x, **kwargs), alpha=0.2)
        x = self.dense_3(tf.concat([x, y], 1), **kwargs)
        return tf.reshape(x, [tf.shape(y)[0], *self.design_shape])
