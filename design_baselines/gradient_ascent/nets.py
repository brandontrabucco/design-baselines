from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np

def reverse_concat(x):
    rev_x = tf.reverse(x, axis=-1)

class ConvnetModel(tf.keras.Sequential):
    distribution = tfpd.Normal

    def __init__(self,
                 input_shape,
                 activations=('relu', 'relu'),
                 hidden=2048,
                 initial_max_std=0.2,
                 initial_min_std=0.1):
        self.max_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_max_std).astype(np.float32)), trainable=True)
        self.min_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_min_std).astype(np.float32)), trainable=True)

        activations = [tfkl.LeakyReLU if act == 'leaky_relu' else
                       tfkl.Activation(tf.math.cos) if act == 'cos' else
                       act for act in activations]

        # Conv net
        layers = []
        layers.append(tfkl.Lambda(
            lambda x: tf.concat([x, tf.reverse(x, axis=[-1,])], axis=-1), output_shape=(None, 6, 6)
        ))
        layers.append(tfkl.Lambda(
            lambda x: tf.expand_dims(x, -1)))
        layers.append(tf.keras.layers.Conv2D(
            8, 1, activation=None, input_shape=(6, 6, 1)
        ))
        layers.append(tfkl.LeakyReLU())
        layers.append(tf.keras.layers.Conv2D(
            32, 3, activation=None, padding="valid", input_shape=(6, 6, 8)
        ))
        layers.append(tfkl.LeakyReLU())
        layers.append(tf.keras.layers.Conv2D(
            32, 1, activation=None, input_shape=(4, 4, 32)
        ))
        layers.append(tfkl.LeakyReLU())
        layers.append(tfkl.Flatten(input_shape=(4, 4, 32)))
        for act in activations:
            layers.extend([tfkl.Dense(hidden),
                           tfkl.Activation(act)
                           if isinstance(act, str) else act()])
        layers.append(tfkl.Dense(2))
        super(ConvnetModel, self).__init__(layers)
    
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

        print ('inputs: ', inputs)
        prediction = super(ConvnetModel, self).__call__(inputs, **kwargs)
        print ('prediction: ', prediction)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        print ('Mean/logstd: ', mean.shape, logstd.shape)
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

class ForwardModel(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    distribution = tfpd.Normal

    def __init__(self,
                 input_shape,
                 activations=('relu', 'relu'),
                 hidden=2048,
                 initial_max_std=0.2,
                 initial_min_std=0.1):
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

        activations = [tfkl.LeakyReLU if act == 'leaky_relu' else
                       tfkl.Activation(tf.math.cos) if act == 'cos' else
                       act for act in activations]

        layers = [tfkl.Flatten(input_shape=input_shape)]
        for act in activations:
            layers.extend([tfkl.Dense(hidden),
                           tfkl.Activation(act)
                           if isinstance(act, str) else act()])
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


