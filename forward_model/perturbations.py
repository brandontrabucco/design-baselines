import tensorflow as tf
import abc


class Perturbation(abc.ABC, tf.Module):
    """Base class for perturbation distributions that sample
    new x conditioned on real x
    """

    @abc.abstractmethod
    def __call__(self,
                 x,
                 **kwargs):
        """Samples perturbed values for x using gradient ascent
        to find adversarial examples

        Args:

        x: tf.Tensor
            the original and central value of the tensor being optimized

        Returns:

        optimized_x: tf.Tensor
            the perturbed value of x that maximizes the score function
        """

        return NotImplemented


class GradientAscent(Perturbation):

    def __init__(self,
                 forward_model,
                 learning_rate=tf.constant(0.001),
                 optim=tf.keras.optimizers.Adam,
                 max_steps=tf.constant(100)):
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
        self.optim = optim(learning_rate=learning_rate)
        self.max_steps = max_steps
        self.perturbation = None

    def optimization_step(self, **kwargs):
        """Take a step of gradient descent to find a design that maximizes
        the forward model
        """

        self.optim.minimize(
            lambda: -self.forward_model(
                self.perturbation, **kwargs), [self.perturbation])

    @tf.function(experimental_relax_shapes=True)
    def __call__(self,
                 x,
                 **kwargs):
        """Samples perturbed values for x using gradient ascent
        to find adversarial examples

        Args:

        x: tf.Tensor
            the original and central value of the tensor being optimized

        Returns:

        optimized_x: tf.Tensor
            the perturbed value of x that maximizes the score function
        """

        if self.perturbation is None:
            self.perturbation = tf.Variable(tf.zeros_like(x))
            self.optimization_step(**kwargs)

        self.perturbation.assign(x)
        for state in self.optim.variables():
            state.assign(tf.zeros_like(state))

        for _ in tf.range(self.max_steps):
            self.optimization_step(**kwargs)
        return tf.convert_to_tensor(self.perturbation)
