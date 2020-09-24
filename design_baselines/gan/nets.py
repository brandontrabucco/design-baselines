from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np


class Discriminator(tf.keras.Model):
    """A Fully Connected Network conditioned on a score"""

    def __init__(self,
                 design_shape,
                 hidden=50,
                 method='wasserstein'):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        design_shape: List[int]
            a list of tuple of integers that represents the shape of a
            single design for a particular task
        hidden: int
            the number of hidden units in every layer of the
            discriminator neural network
        method: str
            the type of loss function used by the discriminator to
            regress to real and fake samples
        """

        super(Discriminator, self).__init__()
        assert method in ['wasserstein',
                          'least_squares',
                          'binary_cross_entropy']
        self.method = method
        self.input_size = np.prod(design_shape)

        # define a layer of the neural net with two pathways
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((None, self.input_size))
        self.ln_0 = tfkl.LayerNormalization()
        self.ln_0.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((None, hidden))
        self.ln_1 = tfkl.LayerNormalization()
        self.ln_1.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_2 = tfkl.Dense(hidden)
        self.dense_2.build((None, hidden))
        self.ln_2 = tfkl.LayerNormalization()
        self.ln_2.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(1)
        self.dense_3.build((None, hidden))

    def __call__(self,
                 x,
                 **kwargs):
        """Use a neural network to discriminate the log probability that a
        sampled design X has score y

        Args:

        X: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design

        Args:

        log_p: tf.Tensor
            a tensor that represents the log probability of either X being
            real of X being fake depending on the value of 'real'
        """

        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, [tf.shape(x)[0], self.input_size])
        x = self.dense_0(x, **kwargs)
        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)
        x = self.dense_1(x, **kwargs)
        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)
        x = self.dense_2(x, **kwargs)
        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)
        return self.dense_3(x, **kwargs)

    def penalty(self,
                h,
                **kwargs):
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
            x = self.__call__(h, **kwargs)
        g = tf.reshape(tape.gradient(x, h), [-1, self.input_size])
        return (1.0 - tf.linalg.norm(g, axis=-1, keepdims=True)) ** 2

    def loss(self,
             x,
             labels,
             **kwargs):
        """Use a neural network to discriminate the log probability that a
        sampled design X has score y

        Args:

        X: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design
        labels: tf.Tensor
            a binary indicator tensor that represents targets labels
            for computing the least squares loss

        Args:

        log_p: tf.Tensor
            a tensor that represents the log probability of either X being
            real of X being fake depending on the value of 'labels'
        accuracy: tf.Tensor
            a tensor that represents the accuracy of the predictions made
            by the discriminator given the targets labels
        """

        # Implements the Wasserstein GAN loss function
        if self.method == 'wasserstein':
            x = self.__call__(x, **kwargs)
            p = tf.where(labels > 0.5, -x, x)
            return x, p, tf.where(labels > 0.5, tf.cast(
                x > 0.0, tf.float32), tf.cast(x < 0.0, tf.float32))

        # Implements the Least-Squares GAN loss function
        elif self.method == 'least_squares':
            x = self.__call__(x, **kwargs)
            p = 0.5 * tf.math.squared_difference(x, labels)
            return x, p, tf.where(labels > 0.5, tf.cast(
                x > 0.5, tf.float32), tf.cast(x < 0.5, tf.float32))

        # Implements the Binary-Cross-Entropy GAN loss function
        elif self.method == 'binary_cross_entropy':
            x = tf.math.sigmoid(self.__call__(x, **kwargs))
            p = tf.keras.losses.binary_crossentropy(labels, x)[..., tf.newaxis]
            return x, p, tf.where(labels > 0.5, tf.cast(
                x > 0.5, tf.float32), tf.cast(x < 0.5, tf.float32))


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
        self.dense_0.build((None, latent_size))
        self.ln_0 = tfkl.LayerNormalization()
        self.ln_0.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((None, hidden))
        self.ln_1 = tfkl.LayerNormalization()
        self.ln_1.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_2 = tfkl.Dense(hidden)
        self.dense_2.build((None, hidden))
        self.ln_2 = tfkl.LayerNormalization()
        self.ln_2.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(np.prod(design_shape))
        self.dense_3.build((None, hidden))

    def sample(self,
               batch_size,
               **kwargs):
        """Generate samples of designs X that have a score y where y is
        the score that the generator conditions on

        Args:

        batch_size: tf.Tensor
            a tf.int32 tensor that represents the scalar batch size
            used to determine how many GAN samples to dray

        Returns:

        x_fake: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design
        """

        temp = kwargs.pop("temp", 1.0)
        z = tf.random.normal([batch_size, self.latent_size])
        x = tf.cast(z, tf.float32)
        x = self.dense_0(x, **kwargs)
        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)
        x = self.dense_1(x, **kwargs)
        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)
        x = self.dense_2(x, **kwargs)
        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)
        x = self.dense_3(x, **kwargs)
        logits = tf.reshape(x, [batch_size, *self.design_shape])
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
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((None, latent_size))
        self.ln_0 = tfkl.LayerNormalization()
        self.ln_0.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((None, hidden))
        self.ln_1 = tfkl.LayerNormalization()
        self.ln_1.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_2 = tfkl.Dense(hidden)
        self.dense_2.build((None, hidden))
        self.ln_2 = tfkl.LayerNormalization()
        self.ln_2.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(np.prod(design_shape))
        self.dense_3.build((None, hidden))

    def sample(self,
               batch_size,
               **kwargs):
        """Generate samples of designs X that have a score y where y is
        the score that the generator conditions on

        Args:

        batch_size: tf.Tensor
            a tf.int32 tensor that represents the scalar batch size
            used to determine how many GAN samples to dray

        Returns:

        x_fake: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design
        """

        kwargs.pop("temp", 1.0)
        z = tf.random.normal([batch_size, self.latent_size])
        x = tf.cast(z, tf.float32)
        x = self.dense_0(x, **kwargs)
        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)
        x = self.dense_1(x, **kwargs)
        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)
        x = self.dense_2(x, **kwargs)
        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)
        x = self.dense_3(x, **kwargs)
        return tf.reshape(x, [batch_size, *self.design_shape])
