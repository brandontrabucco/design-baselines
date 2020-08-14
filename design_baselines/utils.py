import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def get_rank(x):
    return tf.cast(tf.argsort(tf.argsort(
        x, axis=-1, direction="ASCENDING"), axis=-1) + 1, x.dtype)


@tf.function
def sp_rank(x, y):
    cov = tfp.stats.covariance(
        x, y, sample_axis=-1, keepdims=False, event_axis=None)
    sd_x = tfp.stats.stddev(
        x, sample_axis=-1, keepdims=True, name=None)
    sd_y = tfp.stats.stddev(
        y, sample_axis=-1, keepdims=True, name=None)
    return cov / (sd_x * sd_y)


@tf.function
def spearman(a, b):
    return sp_rank(get_rank(a), get_rank(b))
