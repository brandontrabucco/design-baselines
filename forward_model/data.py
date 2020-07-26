from forward_model.utils import get_weights
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as tfkl
import gym
import os


class PolicyWeightsDataset(object):

    def __init__(self,
                 obs_dim=11,
                 action_dim=3,
                 hidden_dim=64,
                 val_size=200,
                 batch_size=32,
                 env_name='Hopper-v2',
                 seed=0,
                 x_file='hopper_controller_X.txt',
                 y_file='hopper_controller_y.txt',
                 include_weights=False):
        """Load static datasets of weights and their corresponding
        expected returns from the disk

        Args:

        obs_dim: int
            the number of channels in the environment observations
        action_dim: int
            the number of channels in the environment actions
        hidden_dim: int
            the number of channels in policy hidden layers
        val_size: int
            the number of samples in the validation set
        batch_size: int
            the batch size when training
        env_name: str
            the name of the gym.Env to use when collecting rollouts
        seed: int
            the random seed that controls the dataset shuffle
        x_file: str
            the name of the dataset file to be loaded for x
        y_file: str
            the name of the dataset file to be loaded for y
        include_weights: bool
            whether to build a dataset that includes sample weights for MIN
        """

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.val_size = val_size
        self.batch_size = batch_size
        self.env_name = env_name

        self.load_resources(seed=seed, x_file=x_file, y_file=y_file)
        self.build(include_weights=include_weights)

    def load_resources(self,
                       seed=0,
                       x_file='hopper_controller_X.txt',
                       y_file='hopper_controller_y.txt'):
        """Load static datasets of weights and their corresponding
        expected returns from the disk

        Args:

        seed: int
            the random seed that controls the dataset shuffle
        x_file: str
            the name of the dataset file to be loaded for x
        y_file: str
            the name of the dataset file to be loaded for y
        """

        np.random.seed(seed)

        basedir = os.path.dirname(os.path.abspath(__file__))
        x = np.loadtxt(os.path.join(basedir, x_file))
        y = np.loadtxt(os.path.join(basedir, y_file))
        x = x.astype(np.float32)
        y = y.astype(np.float32).reshape([-1, 1])

        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        self.x = x[indices]
        self.y = y[indices]

    def build(self,
              include_weights=False):
        """Load static datasets of weights and their corresponding
        expected returns from the disk

        Args:

        include_weights: bool
            whether to build a dataset that includes sample weights for MIN
        """

        if include_weights:
            train = tf.data.Dataset.from_tensor_slices((
                self.x[self.val_size:],
                self.y[self.val_size:],
                get_weights(self.y[self.val_size:])))
            validate = tf.data.Dataset.from_tensor_slices((
                self.x[:self.val_size],
                self.y[:self.val_size],
                get_weights(self.y[:self.val_size])))

        else:
            train = tf.data.Dataset.from_tensor_slices((
                self.x[self.val_size:],
                self.y[self.val_size:]))
            validate = tf.data.Dataset.from_tensor_slices((
                self.x[:self.val_size],
                self.y[:self.val_size]))

        train = train.shuffle(self.x.shape[0] - self.val_size)
        validate = validate.shuffle(self.val_size)
        train = train.batch(self.batch_size)
        validate = validate.batch(self.batch_size)

        self.train = train.prefetch(
            tf.data.experimental.AUTOTUNE)
        self.validate = validate.prefetch(
            tf.data.experimental.AUTOTUNE)

    @property
    def stream_shapes(self):
        """Return the number of weights and biases in the design
        space of the policy

        Returns:

        shape: list
            the shape of a single data point in the dataset
        """

        return [(self.obs_dim, self.hidden_dim),
                (self.hidden_dim,),
                (self.hidden_dim, self.hidden_dim),
                (self.hidden_dim,),
                (self.hidden_dim, self.action_dim),
                (self.action_dim,),
                (1, self.action_dim)]

    @property
    def stream_sizes(self):
        """Return the number of weights and biases in the design
        space of the policy

        Returns:

        shape: list
            the shape of a single data point in the dataset
        """

        return [self.obs_dim * self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim * self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim * self.action_dim,
                self.action_dim,
                self.action_dim]

    @property
    def input_shape(self):
        """Return the number of weights and biases in the design
        space of the policy

        Returns:

        shape: list
            the shape of a single data point in the dataset
        """

        return self.x.shape[1],

    @property
    def input_size(self):
        """Return the number of weights and biases in the design
        space of the policy

        Returns:

        n: int
            the number of weights in a single data point
        """

        return np.prod(self.input_shape)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def score(self, x):
        """Assign a score to a large set of wrights provided by
        performing a rollout in an environment

        Args:

        x: tf.Tensor
            a batch of designs that will be evaluated using an oracle

        Returns:

        score: tf.Tensor
            a vector of returns for policies whose weights are x[i]
        """

        y = tf.map_fn(self.score_tf, x, parallel_iterations=16)
        y.set_shape(x.get_shape()[:1])
        return y

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def score_tf(self, x):
        """Assign a score to a single set of weights provided by
        performing a rollout in an environment

        Args:

        x: np.ndarray
            a single design that will be evaluated using an oracle

        Returns:

        score: np.ndarray
            a return for a policy whose weights are x
        """

        return tf.numpy_function(self.score_np, [x], tf.float32)

    def score_np(self, x) -> np.ndarray:
        """Assign a score to a single set of weights provided by
        performing a rollout in an environment

        Args:

        x: np.ndarray
            a single design that will be evaluated using an oracle

        Returns:

        score: np.ndarray
            a return for a policy whose weights are x
        """

        # extract weights from the vector design
        weights = []
        for s in self.stream_shapes:
            weights.append(x[0:np.prod(s)].reshape(s))
            x = x[np.prod(s):]

        # the final weight is logstd and is not used
        weights.pop(-1)

        # create a policy forward pass in numpy
        def mlp_policy(h):
            h = np.tanh(h @ weights[0] + weights[1])
            h = np.tanh(h @ weights[2] + weights[3])
            return h @ weights[4] + weights[5]

        # make a copy of the policy and the environment
        env = gym.make(self.env_name)

        # perform a single rollout for quick evaluation
        obs, done = env.reset(), False
        path_returns = np.zeros([], dtype=np.float32)
        while not done:
            obs, rew, done, info = env.step(mlp_policy(obs))
            path_returns += rew.astype(np.float32)
        return path_returns
