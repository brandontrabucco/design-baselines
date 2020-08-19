import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def get_rank(x):
    return tf.cast(tf.argsort(tf.argsort(
        x, axis=-1, direction="ASCENDING"), axis=-1) + 1, x.dtype)


@tf.function
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


@tf.function
def add_noise(x, extent, is_discrete=False):
    """Add noise to a input that is either a continuous value or a probability
    distribution over discrete categorical values

    Args:

    x: tf.Tensor
        a tensor that will have noise added to it, such that the resulting
        tensor is sound given its definition
    extent: float
        the extent to which noise will be added, which is positive-valued when
        continuous and in (0, 1) when continuous
    is_discrete: bool
        determines the type of noise to add to the input, such as a mixture of
        distributions, when discrete x is provided

    Returns:

    noisy_x: tf.Tensor
        a tensor that has noise added to it, which has the interpretation of
        the original tensor (such as a probability distribution)
    """

    if is_discrete:
        noise = tf.random.uniform(tf.shape(x))
        noise = noise / tf.reduce_sum(noise, axis=-1, keepdims=True)
        return (1.0 - extent) * x + extent * noise
    else:
        noise = tf.random.normal(tf.shape(x))
        return x + extent * noise
