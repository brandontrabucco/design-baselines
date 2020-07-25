import tensorflow as tf
import tensorflow_probability as tfp
import abc


class Perturbation(abc.ABC, tf.Module):
    """Base class for perturbation distributions that sample
    new x conditioned on real x
    """

    @abc.abstractmethod
    def __call__(self,
                 original_x):
        return NotImplemented


class Gaussian(Perturbation):

    def __init__(self,
                 std=0.1):
        """Create a normal distribution that samples perturbed values
        for a real variable x

        Args:

        std: float
            the standard deviation of the normal distribution
        """

        super().__init__()
        self.std = std

    def __call__(self,
                 original_x):
        """Samples perturbed values for x using a fast gradient sign method
        to find adversarial examples

        Args:

        original_x: tf.Tensor
            the original and central value of the tensor being optimized

        Returns:

        optimized_x: tf.Tensor
            the perturbed value of x that maximizes the score function
        """

        distribution = tfp.distributions.MultivariateNormalDiag(
            loc=original_x, scale_diag=tf.ones_like(original_x) * self.std)
        return distribution.sample()


class FGSM(Perturbation):

    def __init__(self,
                 forward_model,
                 epsilon=0.015):
        """Create a fast gradient sign method module that
        samples perturbed x values

        Args:

        forward_model: tf.keras.Model
            a keras model that accepts vectorized inputs and returns scores
        epsilon: float or None
            the magnitude of the gradient sign to add to the original x
        """

        super().__init__()
        self.forward_model = forward_model
        self.epsilon = epsilon

    def __call__(self,
                 original_x):
        """Samples perturbed values for x using a fast gradient sign method
        to find adversarial examples

        Args:

        original_x: tf.Tensor
            the original and central value of the tensor being optimized

        Returns:

        optimized_x: tf.Tensor
            the perturbed value of x that maximizes the score function
        """

        x_var = tf.Variable(original_x)
        with tf.GradientTape() as gradient_tape:
            return original_x + self.epsilon * tf.math.sign(
                gradient_tape.gradient( -self.forward_model(x_var), [x_var]))


class PGD(Perturbation):

    def __init__(self,
                 forward_model,
                 clip_two_norm=None,
                 clip_inf_norm=None,
                 optim=tf.keras.optimizers.SGD,
                 **optimizer_kwargs):
        """Create a projected gradient descent module that
        samples perturbed x values

        Args:

        forward_model: tf.keras.Model
            a keras model that accepts vectorized inputs and returns scores
        clip_two_norm: float or None
            if provided, specifies the maximum l2 norm of perturbations
        clip_inf_norm: float or None
            if provided, specifies the maximum infinity norm of perturbations
        optim: __class__
            the optimizer class to use such as tf.keras.optimizers.SGD
        **optimizer_kwargs: dict
            additional keyword arguments passed to optim
        """

        assert clip_two_norm is None or clip_inf_norm is None
        super().__init__()
        self.forward_model = forward_model
        self.clip_two_norm = clip_two_norm
        self.clip_inf_norm = clip_inf_norm
        self.optim = optim(**optimizer_kwargs)
        self.optimizer_kwargs = optimizer_kwargs

    @property
    def epsilon(self):
        """Returns the clipping epsilon parameter used if clip_two_norm
        or clip_inf_norm are specified

        Returns:

        epsilon: float
            the value of either clip_two_norm or clip_inf_norm
        """

        if self.clip_inf_norm is not None:
            return self.clip_inf_norm
        if self.clip_two_norm is not None:
            return self.clip_two_norm

    def project(self,
                original_x,
                x_var):
        """Returns a projected value to be assigned to the variable x if
        clip_two_norm or clip_inf_norm are specified

        Args:

        original_x: tf.Tensor
            the original and central value of the tensor being optimized
        x_var: tf.Variable
            the variable form of the tensor being optimized

        Returns:

        projected_x: tf.Tensor
            a new value for x_var that is possibly projected
        """

        delta = tf.convert_to_tensor(x_var) - original_x
        if self.clip_two_norm is not None:
            delta = tf.clip_by_norm(delta, self.epsilon, axis=1)
        if self.clip_inf_norm is not None:
            delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)
        return delta + original_x

    def __call__(self,
                 original_x,
                 num_steps=10):
        """Samples perturbed values for x using projected gradient descent
        to find adversarial examples

        Args:

        original_x: tf.Tensor
            the original and central value of the tensor being optimized
        num_steps: int
            the number of gradient descent steps to use

        Returns:

        optimized_x: tf.Tensor
            the perturbed value of x that maximizes the score function
        """

        x_var = tf.Variable(original_x)
        for variable in self.optim.variables():
            variable.assign(tf.zeros_like(variable))
        for step in range(num_steps):
            self.optim.minimize(lambda: -self.forward_model(x_var), [x_var])
            x_var.assign(self.project(original_x, x_var))
        return tf.convert_to_tensor(x_var)
