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
def add_discrete_noise(x, keep=1.0, temp=1.0):
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

    noise = tf.ones_like(x)
    noise = noise / tf.reduce_sum(noise, axis=-1, keepdims=True)
    return tfpd.RelaxedOneHotCategorical(
        temp, probs=keep * x + (1.0 - keep) * noise).sample()


@tf.function(experimental_relax_shapes=True)
def add_continuous_noise(x, noise_std=1.0):
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
