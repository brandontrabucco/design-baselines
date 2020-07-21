import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np


def adaptive_temp_v2(scores_np):
    """Calculate an adaptive temperature value based on the
    statistics of the scores array
    """

    inverse_arr = scores_np
    max_score = inverse_arr.max()
    scores_new = inverse_arr - max_score
    quantile_ninety = np.quantile(scores_new, q=0.9)
    return np.abs(quantile_ninety)


def softmax(arr, temp=1.0):
    """Calculate the softmax using numpy by normalizing a vector
    to have entries that sum to one
    """

    max_arr = arr.max()
    arr_new = arr - max_arr
    exp_arr = np.exp(arr_new / temp)
    return exp_arr / np.sum(exp_arr)


def get_weights(scores):
    """Calculate weights used for training a model inversion
    network with a per-sample reweighted objective
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

    return weights[:, np.newaxis]


def fgsm(model, x):
    """Implements the fast gradient sign method for identifying
    adversarial examples  for a forward model

    Args:

    model: tf.keras.Model
        a keras model that accepts x and returns a scalar score
    x: tf.Tensor
        an initial point for the fast gradient sign method

    Returns:

    g: tf.tensor
        a tensor with the same shape as x that is a fast gradient sign
    """

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    gradient = tape.batch_jacobian(y, x)[:, 0, :]
    return tf.math.sign(gradient)


def step_function(optim,
                  model,
                  X,
                  y,
                  w=None,
                  sc_noise_std=0.3,
                  sc_lambda=10.0,
                  sc_weight=1.0,
                  cs_noise_std=0.3,
                  cs_weight=1.0):
    """Perform a step of gradient descent on a forward model using a few
    regularization methods like the self-correcting property
    """

    with tf.GradientTape() as tape:
        loss_sc = loss_cs = 0
        loss = tf.reduce_mean(
            tf.keras.losses.mean_squared_error(y, model(X)) * w)

        if sc_noise_std > 0.0:

            # add a self-correcting term during optimization on the training set
            X_sc = X + tf.random.normal(X.shape) * sc_noise_std
            d_sc = tf.linalg.norm(X - X_sc, axis=1)
            loss_sc = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(
                    y - sc_lambda * d_sc, model(X_sc)) * w)

        if cs_noise_std > 0.0:

            # add a conservative term during optimization on the training set
            X_cs = X + tf.random.normal(X.shape) * cs_noise_std
            loss_cs = tf.reduce_mean(model(X_cs) * w)

        grads = tape.gradient(loss +
                              sc_weight * loss_sc +
                              cs_weight * loss_cs,
                              model.trainable_variables)

        optim.apply_gradients(
            zip(grads, model.trainable_variables))

    return loss


class ControllerDataset(object):

    def load_resources(self,
                       seed=0):
        """Load static datasets of weights and their corresponding
        expected returns from the disk
        """

        import os

        basedir = os.path.dirname(os.path.abspath(__file__))
        robots = np.loadtxt(os.path.join(
            basedir, "hopper_controller_X.txt"))
        scores = np.loadtxt(os.path.join(
            basedir, "hopper_controller_y.txt"))

        robots = robots.astype(np.float32)
        scores = scores.astype(np.float32).reshape([-1, 1])
        weights = get_weights(scores).astype(np.float32)

        indices = np.arange(robots.shape[0])
        np.random.seed(seed)
        np.random.shuffle(indices)

        self.robots = robots[indices]
        self.scores = scores[indices]
        self.weights = weights[indices]

    def build(self):

        train = tf.data.Dataset.from_tensor_slices((
            self.robots[self.val_size:],
            self.scores[self.val_size:],
            self.weights[self.val_size:]))
        train = train.shuffle(self.robots.shape[0] - self.val_size).batch(32)
        self.train = train.prefetch(tf.data.experimental.AUTOTUNE)

        val = tf.data.Dataset.from_tensor_slices((
            self.robots[:self.val_size],
            self.scores[:self.val_size],
            self.weights[:self.val_size]))
        val = val.shuffle(self.val_size).batch(32)
        self.val = val.prefetch(tf.data.experimental.AUTOTUNE)

    def __init__(self,
                 obs_dim=11,
                 action_dim=3,
                 hidden_dim=64,
                 seed=0,
                 val_size=200,
                 env_name='Hopper-v2'):
        """Load static datasets of weights and their corresponding
        expected returns from the disk
        """

        import tensorflow.keras.layers as tfkl
        import gym

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.val_size = val_size
        self.env_name = env_name

        self.robots = None
        self.scores = None
        self.weights = None
        self.load_resources(seed=seed)

        self.train = None
        self.val = None
        self.build()

        self.env = gym.make(env_name)
        self.policy = tf.keras.Sequential([
            tfkl.Dense(hidden_dim, use_bias=True, input_shape=(obs_dim,)),
            tfkl.Activation('tanh'),
            tfkl.Dense(hidden_dim, use_bias=True),
            tfkl.Activation('tanh'),
            tfkl.Dense(action_dim, use_bias=True)])

    def score(self, x) -> np.ndarray:
        """Assign a score for the specified designs using an internal
        simulator of the design problem mechanics.
        """

        scores = []
        for i in range(x.shape[0]):

            # extract weights from the vector design
            xi = x[i].numpy()
            weights = []
            for s in [(self.obs_dim, self.hidden_dim),
                      (self.hidden_dim,),
                      (self.hidden_dim, self.hidden_dim),
                      (self.hidden_dim,),
                      (self.hidden_dim, self.action_dim),
                      (self.action_dim,),
                      (1, self.action_dim)]:
                weights.append(xi[0:np.prod(s)].reshape(s))
                xi = xi[np.prod(s):]

            # the final weight is logstd and is not used
            weights.pop(-1)

            # set the policy weights to those provided
            self.policy.set_weights(weights)

            # perform a single rollout for quick evaluation
            obs, done = self.env.reset(), False
            scores.append(0.0)
            while not done:
                act = self.policy(obs[np.newaxis])[0]
                obs, rew, done, info = self.env.step(act)
                scores[-1] += rew

        return np.array(scores)[:, np.newaxis]
