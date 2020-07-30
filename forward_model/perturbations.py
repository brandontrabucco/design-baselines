import tensorflow as tf
import abc


class Perturbation(abc.ABC, tf.Module):
    """Base class for perturbation distributions that sample
    new x conditioned on real x
    """

    @abc.abstractmethod
    def __call__(self,
                 original_x,
                 epoch,
                 **kwargs):
        """Samples perturbed values for x using gradient ascent
        to find adversarial examples

        Args:

        original_x: tf.Tensor
            the original and central value of the tensor being optimized
        epoch: tf.Tensor
            the number of the current epoch of training

        Returns:

        optimized_x: tf.Tensor
            the perturbed value of x that maximizes the score function
        """

        return NotImplemented


class GradientAscent(Perturbation):

    def __init__(self,
                 forward_model,
                 learning_rate=tf.constant(0.001),
                 max_steps=tf.constant(100),
                 epochs=tf.constant(100)):
        """Create a gradient ascent module that finds adversarial
        negative samples from a forward model


        Args:

        forward_model: tf.keras.Model
            a keras model that accepts vectorized inputs and returns scores
        learning_rate: float
            the learning rate used when optimizing for the input x
        """

        super().__init__()
        self.forward_model = forward_model
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.epochs = epochs

    @tf.function
    def schedule(self, epoch):
        """Calculate a schedule for the number of gradient steps to
        optimize x for to find adversarial examples

        Args:

        epoch: tf.Tensor
            the number of the current epoch of training

        Returns:

        num_gradient_steps: tf.Tensor
            the number of gradient steps to optimize x for this epoch
        """

        return tf.cast(tf.math.ceil(
            tf.cast(self.max_steps, tf.float32) *
            tf.cast(epoch + 1, tf.float32) /
            tf.cast(self.epochs, tf.float32)), tf.int32)

    @tf.function(experimental_relax_shapes=True)
    def __call__(self,
                 original_x,
                 epoch,
                 **kwargs):
        """Samples perturbed values for x using gradient ascent
        to find adversarial examples

        Args:

        original_x: tf.Tensor
            the original and central value of the tensor being optimized
        epoch: tf.Tensor
            the number of the current epoch of training

        Returns:

        optimized_x: tf.Tensor
            the perturbed value of x that maximizes the score function
        """

        for _ in tf.range(self.max_steps):
            with tf.GradientTape() as tape:
                tape.watch(original_x)
                prediction = self.forward_model(original_x, **kwargs)
            original_x = original_x + tape.gradient(
                prediction, original_x) * self.learning_rate
        return original_x
