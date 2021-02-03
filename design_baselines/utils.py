from tensorflow_probability import distributions as tfpd
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function(experimental_relax_shapes=True)
def get_rank(x):
    return tf.cast(tf.argsort(tf.argsort(
        x, axis=-1, direction="ASCENDING"), axis=-1) + 1, x.dtype)


@tf.function(experimental_relax_shapes=True)
def spearman(a, b):
    """Computes the Spearman Rank-Correlation Coefficient for two
    continuous-valued tensors with the same shape

    Args:

    a: tf.Tensor
        a tensor of any shape whose last axis represents the axis of which
        to rank elements of the tensor from
    b: tf.Tensor
        a tensor of any shape whose last axis represents the axis of which
        to rank elements of the tensor from

    Returns:

    rho: tf.Tensor
        a tensor with the same shape as the first N - 1 axes of a and b, and
        represents the spearman p between a and b
    """

    x = get_rank(a)
    y = get_rank(b)
    cov = tfp.stats.covariance(
        x, y, sample_axis=-1, keepdims=False, event_axis=None)
    sd_x = tfp.stats.stddev(
        x, sample_axis=-1, keepdims=True, name=None)
    sd_y = tfp.stats.stddev(
        y, sample_axis=-1, keepdims=True, name=None)
    return cov / (sd_x * sd_y)


@tf.function(experimental_relax_shapes=True)
def disc_noise(x, keep=0.9, temp=5.0):
    """Add noise to a input that is either a continuous value or a probability
    distribution over discrete categorical values

    Args:

    x: tf.Tensor
        a tensor that will have noise added to it, such that the resulting
        tensor is sound given its definition
    keep: float
        the amount of probability mass to keep on the element that is activated
        teh rest is redistributed evenly to all elements
    temp: float
        the temperature of teh gumbel distribution that is used to corrupt
        the input probabilities x

    Returns:

    noisy_x: tf.Tensor
        a tensor that has noise added to it, which has the interpretation of
        the original tensor (such as a probability distribution)
    """

    p = tf.ones_like(x)
    p = p / tf.reduce_sum(p, axis=-1, keepdims=True)
    p = keep * x + (1.0 - keep) * p
    return tfpd.RelaxedOneHotCategorical(temp, probs=p).sample()


@tf.function(experimental_relax_shapes=True)
def soft_noise(x, keep=0.9, temp=5.0):
    """Softens a discrete one-hot distribution so that the maximum entry is
    keep and the minimum entries are (1 - keep)

    Args:

    x: tf.Tensor
        a tensor that will have noise added to it, such that the resulting
        tensor is sound given its definition
    keep: float
        the amount of probability mass to keep on the element that is
        activated; the rest is redistributed to all elements

    Returns:

    smooth_x: tf.Tensor
        a smoothed version of x that represents a discrete probability
        distribution of categorical variables
    """

    p = tf.ones_like(x)
    p = p / tf.reduce_sum(p, axis=-1, keepdims=True)
    return keep * x + (1.0 - keep) * p


@tf.function(experimental_relax_shapes=True)
def cont_noise(x, noise_std=1.0):
    """Add noise to a input that is either a continuous value or a probability
    distribution over discrete categorical values

    Args:

    x: tf.Tensor
        a tensor that will have noise added to it, such that the resulting
        tensor is sound given its definition
    noise_std: float
        the standard deviation of the gaussian noise that will be added
        to the continuous design parameters x

    Returns:

    noisy_x: tf.Tensor
        a tensor that has noise added to it, which has the interpretation of
        the original tensor (such as a probability distribution)
    """

    return x + noise_std * tf.random.normal(tf.shape(x))


def generate_ensemble(num_layers, *activations):
    """Given a set of string names and a number of target layers, generate
    a list of ensemble architectures with those activations

    Args:

    num_layers: int
        the number of hidden layers in the neural network, and also
        the number of activation functions
    activations: list of str
        a list of strings that indicates the candidates for activation
        functions at for every layer

    Returns:

    ensemble: list of list of str
        a list of architectures, where an architecture is given by a
        list of activation function names
    """

    if num_layers == 0:
        return []
    if num_layers == 1:
        return [[act] for act in activations]
    return [[act, *o] for act in activations
            for o in generate_ensemble(num_layers - 1, *activations)]


def render_video(config, task, solution):

    if config["task"] == "HopperController-v0":

        import gym
        import os
        import numpy as np
        from skvideo.io import FFmpegWriter
        out = FFmpegWriter(os.path.join(
            config['logging_dir'], f'vid.mp4'))

        weights = []
        for s in task.wrapped_task.stream_shapes:
            weights.append(np.reshape(solution[0:np.prod(s)], s))
            solution = solution[np.prod(s):]

        weights.pop(-1)

        def mlp_policy(h):
            h = np.tanh(h @ weights[0] + weights[1])
            h = np.tanh(h @ weights[2] + weights[3])
            return h @ weights[4] + weights[5]

        env = gym.make(task.wrapped_task.env_name)

        for i in range(5):
            obs, done = env.reset(), False
            path_returns = np.zeros([1], dtype=np.float32)
            while not done:
                obs, rew, done, info = env.step(mlp_policy(obs))
                path_returns += rew.astype(np.float32)

                out.writeFrame(env.render(
                    mode='rgb_array', height=500, width=500))

        out.close()
