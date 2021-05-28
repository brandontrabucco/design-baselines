from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np


class ForwardModel(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    distribution = tfpd.Normal

    def __init__(self, task, embedding_size=50, hidden_size=50,
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
        if task.is_discrete:
            layers.append(tfkl.Embedding(task.num_classes, embedding_size,
                                         input_shape=task.input_shape))
        layers.append(tfkl.Flatten(input_shape=task.input_shape)
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


class Encoder(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    distribution = tfpd.MultivariateNormalDiag

    def __init__(self, task, latent_size, embedding_size=50, hidden_size=50,
                 num_layers=1, initial_max_std=1.5, initial_min_std=0.5):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        task: StaticGraphTask
            a model-based optimization task
        latent_size: int
            the cardinality of the latent variable
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
        if task.is_discrete:
            layers.append(tfkl.Embedding(task.num_classes, embedding_size,
                                         input_shape=task.input_shape))
        layers.append(tfkl.Flatten(input_shape=task.input_shape)
                      if len(layers) == 0 else tfkl.Flatten())
        for i in range(num_layers):
            layers.extend([tfkl.Dense(hidden_size), tfkl.LeakyReLU()])

        layers.append(tfkl.Dense(latent_size * 2))
        super(Encoder, self).__init__(layers)

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

        prediction = super(Encoder, self).__call__(inputs, **kwargs)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale_diag": tf.math.softplus(logstd)}

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


class DiscreteDecoder(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    distribution = tfpd.Categorical

    def __init__(self, task, latent_size, hidden_size=50,
                 num_layers=1, **kwargs):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        task: StaticGraphTask
            a model-based optimization task
        latent_size: int
            the cardinality of the latent variable
        hidden_size: int
            the global hidden size of the neural network
        """

        layers = []
        for i in range(num_layers):
            kwargs = dict()
            if i == 0:
                kwargs["input_shape"] = (latent_size,)
            layers.extend([tfkl.Dense(hidden_size, **kwargs),
                           tfkl.LeakyReLU()])

        layers.append(tfkl.Dense(np.prod(task.input_shape) * task.num_classes))
        layers.append(tfkl.Reshape(list(task.input_shape) + [task.num_classes]))
        super(DiscreteDecoder, self).__init__(layers)

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

        x = super(DiscreteDecoder, self).__call__(inputs, **kwargs)
        logits = tf.math.log_softmax(x, axis=-1)
        return {"logits": logits}

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

        return self.distribution(
            **self.get_params(inputs, **kwargs), dtype=tf.int32)


class ContinuousDecoder(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    distribution = tfpd.MultivariateNormalDiag

    def __init__(self, task, latent_size, hidden_size=50,
                 num_layers=1, initial_max_std=1.5, initial_min_std=0.5):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        task: StaticGraphTask
            a model-based optimization task
        latent_size: int
            the cardinality of the latent variable
        hidden_size: int
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

        layers = []
        for i in range(num_layers):
            kwargs = dict()
            if i == 0:
                kwargs["input_shape"] = (latent_size,)
            layers.extend([tfkl.Dense(hidden_size, **kwargs),
                           tfkl.LeakyReLU()])

        layers.append(tfkl.Dense(np.prod(task.input_shape) * 2))
        layers.append(tfkl.Reshape(list(task.input_shape) + [2]))
        super(ContinuousDecoder, self).__init__(layers)

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

        x = super(ContinuousDecoder, self).__call__(inputs, **kwargs)
        mean, logstd = x[..., 0], x[..., 1]
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale_diag": tf.math.exp(logstd)}

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
