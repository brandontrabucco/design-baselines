from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np


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
        self.embed_0 = tfkl.Dense(hidden)
        self.embed_0.build((None, 1))

        # define a layer of the neural net with two pathways
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((None, self.input_size + hidden))
        self.ln_0 = tfkl.LayerNormalization()
        self.ln_0.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((None, hidden + hidden))
        self.ln_1 = tfkl.LayerNormalization()
        self.ln_1.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_2 = tfkl.Dense(hidden)
        self.dense_2.build((None, hidden + hidden))
        self.ln_2 = tfkl.LayerNormalization()
        self.ln_2.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(1)
        self.dense_3.build((None, hidden + hidden))

    def __call__(self,
                 x,
                 y,
                 **kwargs):
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
        x = tf.reshape(x, [tf.shape(x)[0], self.input_size])
        y_embed = self.embed_0(y, **kwargs)

        x = self.dense_0(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)
        x = self.dense_1(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)

        x = self.dense_2(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)
        return self.dense_3(
            tf.concat([x, y_embed], 1), **kwargs)

    def penalty(self,
                h,
                y,
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
            x = self.__call__(h, y, **kwargs)
        g = tf.reshape(tape.gradient(x, h), [-1, self.input_size])
        return (1.0 - tf.linalg.norm(g, axis=-1, keepdims=True)) ** 2

    def loss(self,
             x,
             y,
             labels,
             **kwargs):
        """Use a neural network to discriminate the log probability that a
        sampled design X has score y

        Args:

        X: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design
        y: tf.Tensor
            a batch of scalar scores wherein the generator is trained to
            produce designs that have score y
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
            x = self.__call__(x, y, **kwargs)
            p = tf.where(labels > 0.5, -x, x)
            return x, p, tf.where(labels > 0.5, tf.cast(
                x > 0.0, tf.float32), tf.cast(x < 0.0, tf.float32))

        # Implements the Least-Squares GAN loss function
        elif self.method == 'least_squares':
            x = self.__call__(x, y, **kwargs)
            p = 0.5 * tf.math.squared_difference(x, labels)
            return x, p, tf.where(labels > 0.5, tf.cast(
                x > 0.5, tf.float32), tf.cast(x < 0.5, tf.float32))

        # Implements the Binary-Cross-Entropy GAN loss function
        elif self.method == 'binary_cross_entropy':
            x = tf.math.sigmoid(self.__call__(x, y, **kwargs))
            p = tf.keras.losses.binary_crossentropy(labels, x)[..., tf.newaxis]
            return x, p, tf.where(labels > 0.5, tf.cast(
                x > 0.5, tf.float32), tf.cast(x < 0.5, tf.float32))


class ConvDiscriminator(tf.keras.Model):
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

        super(ConvDiscriminator, self).__init__()
        assert method in ['wasserstein',
                          'least_squares',
                          'binary_cross_entropy']
        self.method = method
        self.embed_0 = tfkl.Dense(hidden)
        self.embed_0.build((None, 1))

        # define a layer of the neural net with two pathways
        self.dense_0 = tfkl.Conv1D(hidden, 3, strides=2, padding="same")
        self.dense_0.build((None, None, design_shape[1] + hidden))
        self.ln_0 = tfkl.LayerNormalization()
        self.ln_0.build((None, None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Conv1D(hidden, 3, strides=2, padding="same")
        self.dense_1.build((None, None, hidden + hidden))
        self.ln_1 = tfkl.LayerNormalization()
        self.ln_1.build((None, None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_2 = tfkl.Conv1D(hidden, 3, strides=2, padding="same")
        self.dense_2.build((None, None, hidden + hidden))
        self.ln_2 = tfkl.LayerNormalization()
        self.ln_2.build((None, None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(1)
        self.dense_3.build((None, hidden + hidden))

    def __call__(self,
                 x,
                 y,
                 **kwargs):
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
        y_embed = self.embed_0(y, **kwargs)

        x = self.dense_0(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)

        x = self.dense_1(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)

        x = self.dense_2(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)
        x = tf.reduce_mean(x, axis=1)
        return self.dense_3(
            tf.concat([x, y_embed], 1), **kwargs)

    def penalty(self,
                h,
                y,
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
            x = self.__call__(h, y, **kwargs)
        g = tf.reshape(tape.gradient(x, h), [tf.shape(x)[0], -1])
        return (1.0 - tf.linalg.norm(g, axis=-1, keepdims=True)) ** 2

    def loss(self,
             x,
             y,
             labels,
             **kwargs):
        """Use a neural network to discriminate the log probability that a
        sampled design X has score y

        Args:

        X: tf.Tensor
            a design the generator is trained to sample from a distribution
            conditioned on the score y achieved by that design
        y: tf.Tensor
            a batch of scalar scores wherein the generator is trained to
            produce designs that have score y
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
            x = self.__call__(x, y, **kwargs)
            p = tf.where(labels > 0.5, -x, x)
            return x, p, tf.where(labels > 0.5, tf.cast(
                x > 0.0, tf.float32), tf.cast(x < 0.0, tf.float32))

        # Implements the Least-Squares GAN loss function
        elif self.method == 'least_squares':
            x = self.__call__(x, y, **kwargs)
            p = 0.5 * tf.math.squared_difference(x, labels)
            return x, p, tf.where(labels > 0.5, tf.cast(
                x > 0.5, tf.float32), tf.cast(x < 0.5, tf.float32))

        # Implements the Binary-Cross-Entropy GAN loss function
        elif self.method == 'binary_cross_entropy':
            x = tf.math.sigmoid(self.__call__(x, y, **kwargs))
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
        self.embed_0 = tfkl.Dense(hidden)
        self.embed_0.build((None, 1))

        # define a layer of the neural net with two pathways
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((None, latent_size + hidden))
        self.ln_0 = tfkl.LayerNormalization()
        self.ln_0.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((None, hidden + hidden))
        self.ln_1 = tfkl.LayerNormalization()
        self.ln_1.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_2 = tfkl.Dense(hidden)
        self.dense_2.build((None, hidden + hidden))
        self.ln_2 = tfkl.LayerNormalization()
        self.ln_2.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(np.prod(design_shape))
        self.dense_3.build((None, hidden + hidden))

    def sample(self,
               y,
               **kwargs):
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

        y_embed = self.embed_0(y, **kwargs)
        x = self.dense_0(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)
        x = self.dense_1(tf.concat([x, y_embed], 1), **kwargs)

        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)
        x = self.dense_2(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)
        x = self.dense_3(tf.concat([x, y_embed], 1), **kwargs)

        logits = tf.reshape(x, [tf.shape(y)[0], *self.design_shape])
        return tfpd.RelaxedOneHotCategorical(
            temp, logits=tf.math.log_softmax(logits)).sample()


class DiscreteConvGenerator(tf.keras.Model):
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

        super(DiscreteConvGenerator, self).__init__()
        self.design_shape = design_shape
        self.latent_size = latent_size
        self.embed_0 = tfkl.Dense(hidden)
        self.embed_0.build((None, 1))

        # define a layer of the neural net with two pathways
        self.dense_0 = tfkl.Conv1D(hidden, 3, strides=1, padding="same")
        self.dense_0.build((None, None, latent_size + hidden))
        self.ln_0 = tfkl.LayerNormalization()
        self.ln_0.build((None, None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Conv1D(hidden, 3, strides=1, padding="same")
        self.dense_1.build((None, None, hidden + hidden))
        self.ln_1 = tfkl.LayerNormalization()
        self.ln_1.build((None, None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_2 = tfkl.Conv1D(hidden, 3, strides=1, padding="same")
        self.dense_2.build((None, None, hidden + hidden))
        self.ln_2 = tfkl.LayerNormalization()
        self.ln_2.build((None, None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Conv1D(design_shape[1], 3, strides=1, padding="same")
        self.dense_3.build((None, None, hidden + hidden))

    def sample(self,
               y,
               **kwargs):
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
        z = tf.random.normal([tf.shape(y)[0], self.design_shape[0], self.latent_size])
        x = tf.cast(z, tf.float32)
        y = tf.cast(y, tf.float32)
        y_embed = self.embed_0(y, **kwargs)

        x = self.dense_0(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)

        x = self.dense_1(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)

        x = self.dense_2(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)

        x = self.dense_3(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

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
        self.embed_0 = tfkl.Dense(hidden)
        self.embed_0.build((None, 1))

        # define a layer of the neural net with two pathways
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((None, latent_size + hidden))
        self.ln_0 = tfkl.LayerNormalization()
        self.ln_0.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((None, hidden + hidden))
        self.ln_1 = tfkl.LayerNormalization()
        self.ln_1.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_2 = tfkl.Dense(hidden)
        self.dense_2.build((None, hidden + hidden))
        self.ln_2 = tfkl.LayerNormalization()
        self.ln_2.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(np.prod(design_shape))
        self.dense_3.build((None, hidden + hidden))

    def sample(self,
               y,
               **kwargs):
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
        y_embed = self.embed_0(y, **kwargs)

        x = self.dense_0(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)
        x = self.dense_1(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)

        x = self.dense_2(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)
        x = self.dense_3(tf.concat([x, y_embed], 1), **kwargs)
        return tf.reshape(x, [tf.shape(y)[0], *self.design_shape])


class ContinuousConvGenerator(tf.keras.Model):
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

        super(ContinuousConvGenerator, self).__init__()
        self.design_shape = design_shape
        self.latent_size = latent_size
        self.embed_0 = tfkl.Dense(hidden)
        self.embed_0.build((None, 1))

        # define a layer of the neural net with two pathways
        self.dense_0 = tfkl.Conv1D(hidden, 3, strides=1, padding="same")
        self.dense_0.build((None, None, latent_size + hidden))
        self.ln_0 = tfkl.LayerNormalization()
        self.ln_0.build((None, None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Conv1D(hidden, 3, strides=1, padding="same")
        self.dense_1.build((None, None, hidden + hidden))
        self.ln_1 = tfkl.LayerNormalization()
        self.ln_1.build((None, None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_2 = tfkl.Conv1D(hidden, 3, strides=1, padding="same")
        self.dense_2.build((None, None, hidden + hidden))
        self.ln_2 = tfkl.LayerNormalization()
        self.ln_2.build((None, None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Conv1D(design_shape[1], 3, padding="same")
        self.dense_3.build((None, None, hidden + hidden))

    def sample(self,
               y,
               **kwargs):
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
        z = tf.random.normal([tf.shape(y)[0], self.design_shape[0], self.latent_size])
        x = tf.cast(z, tf.float32)
        y = tf.cast(y, tf.float32)

        y_embed = self.embed_0(y, **kwargs)

        x = self.dense_0(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)

        x = self.dense_1(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)

        x = self.dense_2(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)

        x = self.dense_3(tf.concat([
            x,
            tf.broadcast_to(y_embed[:, tf.newaxis, :], [
                tf.shape(y_embed)[0],
                tf.shape(x)[1],
                tf.shape(y_embed)[1]])
        ], 2), **kwargs)

        return x
