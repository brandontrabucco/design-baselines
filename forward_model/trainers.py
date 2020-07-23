import tensorflow as tf


class Conservative(tf.Module):

    def __init__(self,
                 forward_model,
                 perturbation_distribution,
                 conservative_weight=1.0,
                 optim=tf.keras.optimizers.Adam,
                 **optimizer_kwargs):
        """Build a trainer for a conservative forward model with negatives
        sampled from a perturbation distribution

        Args:

        forward_model: tf.keras.Model
            a keras model that accepts vectorized inputs and returns scores
        perturbation_distribution: Perturbation
            a distribution that returns new samples conditioned on an X
        conservative_weight: float
            the weight of the conservative loss terms
        optim: __class__
            the optimizer class to use such as tf.keras.optimizers.SGD
        **optimizer_kwargs: dict
            additional keyword arguments passed to optim
        """

        super().__init__()
        self.forward_model = forward_model
        self.perturbation_distribution = perturbation_distribution
        self.conservative_weight = conservative_weight
        self.optim = optim(**optimizer_kwargs)
        self.optimizer_kwargs = optimizer_kwargs

    def train(self,
              dataset):
        """Train a conservative forward model and collect negative
        samples using a perturbation distribution

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss: np.ndarray
            the average loss on the training set this epoch
        """

        total_loss = tf.zeros([0])
        for X, y in dataset:
            with tf.GradientTape() as tape:
                loss = tf.keras.losses.mse(y, self.forward_model(X))
                total_loss = tf.concat([total_loss, loss], 0)
                perturb = tf.stop_gradient(
                    self.perturbation_distribution(X))
                loss = tf.reduce_mean(
                    loss + self.conservative_weight * (
                        self.forward_model(perturb)[:, 0] -
                        self.forward_model(X)[:, 0]))
            grads = tape.gradient(
                loss, self.forward_model.trainable_variables)
            self.optim.apply_gradients(
                zip(grads, self.forward_model.trainable_variables))
        return total_loss

    def validate(self,
                 dataset):
        """Validate a conservative forward model using a validation dataset
        and return the average validation loss

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss: np.ndarray
            the average loss on the validation set this epoch
        """

        total_loss = tf.zeros([0])
        for X, y in dataset:
            loss = tf.keras.losses.mse(y, self.forward_model(X))
            total_loss = tf.concat([total_loss, loss], 0)
        return total_loss
