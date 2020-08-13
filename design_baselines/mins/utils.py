from tensorflow_probability import distributions as tfpd
import numpy as np
import tensorflow as tf


def adaptive_temp_v2(scores_np):
    """Calculate an adaptive temperature value based on the
    statistics of the scores array

    Args:

    scores_np: np.ndarray
        an array that represents the vectorized scores per data point

    Returns:

    temp: np.ndarray
        the scalar 90th percentile of scores in the dataset
    """

    inverse_arr = scores_np
    max_score = inverse_arr.max()
    scores_new = inverse_arr - max_score
    quantile_ninety = np.quantile(scores_new, q=0.9)
    return np.abs(quantile_ninety)


def softmax(arr,
            temp=1.0):
    """Calculate the softmax using numpy by normalizing a vector
    to have entries that sum to one

    Args:

    arr: np.ndarray
        the array which will be normalized using a tempered softmax
    temp: float
        a temperature parameter for the softmax

    Returns:

    normalized: np.ndarray
        the normalized input array which sums to one
    """

    max_arr = arr.max()
    arr_new = arr - max_arr
    exp_arr = np.exp(arr_new / temp)
    return exp_arr / np.sum(exp_arr)


def get_weights(scores):
    """Calculate weights used for training a model inversion
    network with a per-sample reweighted objective

    Args:

    scores: np.ndarray
        scores which correspond to the value of data points in the dataset

    Returns:

    weights: np.ndarray
        an array with the same shape as scores that reweights samples
    """

    scores_np = scores[:, 0]
    hist, bin_edges = np.histogram(scores_np, bins=20)
    hist = hist / np.sum(hist)

    base_temp = adaptive_temp_v2(scores_np)
    softmin_prob = softmax(bin_edges[1:], temp=base_temp)

    provable_dist = softmin_prob * (hist / (hist + 1e-3))
    provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)

    bin_indices = np.digitize(scores_np, bin_edges[1:])
    hist_prob = hist[np.minimum(bin_indices, 19)]

    weights = provable_dist[
        np.minimum(bin_indices, 19)] / (hist_prob + 1e-7)
    weights = np.clip(weights, a_min=0.0, a_max=5.0)
    return weights.astype(np.float32)


def get_p_y(scores):
    """Calculate weights used for training a model inversion
    network with a per-sample reweighted objective

    Args:

    scores: np.ndarray
        scores which correspond to the value of data points in the dataset

    Returns:

    weights: np.ndarray
        an array with the same shape as scores that reweights samples
    """

    scores_np = scores[:, 0]
    hist, bin_edges = np.histogram(scores_np, bins=20)
    hist = hist / np.sum(hist)

    base_temp = adaptive_temp_v2(scores_np)
    softmin_prob = softmax(bin_edges[1:], temp=base_temp)

    provable_dist = softmin_prob * (hist / (hist + 1e-3))
    provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)

    bin_edges = bin_edges.astype(np.float32)
    provable_dist = provable_dist.astype(np.float32)

    return provable_dist[:, np.newaxis], \
           bin_edges[:, np.newaxis]


@tf.function(experimental_relax_shapes=True)
def get_synthetic_data(x,
                       y,
                       is_discrete=False,
                       exploration_samples=32,
                       exploration_rate=10.0,
                       exploration_noise_std=0.1):
    """Generate a synthetic dataset of designs x and scores y using the
    Randomized Labelling algorithm

    Args:

    x: tf.Tensor
        a tensor containing an existing data set of realistic designs
    y: tf.Tensor
        a tensor containing an existing data set of realistic scores
    exploration_samples: int
        the number of samples to draw for randomized labelling
    exploration_rate: float
        the rate of the exponential noise added to y
    exploration_noise_std: float
        the standard deviation of the gaussian noise added to x

    Returns:

    tilde_x: tf.Tensor
        a tensor containing a data set of synthetic designs
    tilde_y: tf.Tensor
        a tensor containing a data set of synthetic scores
    """

    # sample ys based on their importance weight
    p_y, bin_edges = tf.numpy_function(get_p_y, [y], [tf.float32, tf.float32])
    p_y.set_shape([20, 1])
    bin_edges.set_shape([20, 1])

    # sample ys according to the optimal bins
    d = tfpd.Categorical(probs=p_y[:, 0])
    ys_ids = d.sample(sample_shape=(exploration_samples,))
    ys = tf.nn.embedding_lookup(bin_edges, ys_ids)

    # add positive noise to the sampled ys
    d = tfpd.Exponential(rate=exploration_rate)
    ys = ys + d.sample(sample_shape=tf.shape(ys))

    if is_discrete:

        # randomly sample data points from teh space of valid samples
        log_prob = tf.zeros(tf.shape(x)[1:])
        d = tfpd.OneHotCategorical(logits=log_prob, dtype=tf.float32)
        xs = d.sample(sample_shape=(exploration_samples,))

    else:

        # select data points randomly from the data set
        d = tfpd.Categorical(logits=tf.zeros([tf.shape(x)[0]]))
        xs_ids = d.sample(sample_shape=(exploration_samples,))
        xs = tf.nn.embedding_lookup(x, xs_ids)

        # add perturbation noise to the sampled xs
        d = tfpd.Normal(0, exploration_noise_std)
        xs = xs + d.sample(sample_shape=tf.shape(xs))

    # concatenate newly paired samples with the existing data set
    return tf.concat([x, xs], 0), \
           tf.concat([y, ys], 0)
