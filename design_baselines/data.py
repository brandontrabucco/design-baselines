from design_bench.task import Task
from design_bench import make
import tensorflow as tf
import numpy as np


class StaticGraphTask(Task):

    def __init__(self,
                 task_name,
                 **task_kwargs):
        """An interface to a static-graph task which includes a validation
        set and a non differentiable score function

        Args:

        task_name: str
            the name to a valid task using design_bench.make(task_name)
            such as 'HopperController-v0'
        **task_kwargs: dict
            additional keyword arguments that are passed to the design_bench task
            when it is created using design_bench.make
        """

        # use the design_bench registry to make a task
        self.wrapped_task = make(task_name, **task_kwargs)

    @property
    def x(self):
        return self.wrapped_task.x.astype(np.float32)

    @property
    def y(self):
        return self.wrapped_task.y.astype(np.float32)

    @x.setter
    def x(self, x):
        self.wrapped_task.x = x

    @y.setter
    def y(self, y):
        self.wrapped_task.y = y

    @property
    def input_shape(self):
        """Return a tuple that corresponds to the shape of a single
        element of x from the data set
        """

        return self.wrapped_task.input_shape

    @property
    def input_size(self):
        """Return an int that corresponds to the size of a single
        element of x from the data set
        """

        return self.wrapped_task.input_size

    def build(self,
              x=None,
              y=None,
              val_size=200,
              batch_size=128,
              bootstraps=0,
              bootstraps_noise=None,
              importance_weights=None):
        """Provide an interface for splitting a task into a training and
        validation set and including sample re-weighting

        Args:

        x: None or tf.Tensor
            if provided this is used in place of task.x
        y: None or tf.Tensor
            if provided this is used in place of task.y
        val_size: int
            the size of the validation split to use when building the tensorflow
            Dataset.from_tensor_slices dataset
        batch_size: int
            the number of samples to include in a single batch when building the
            Dataset.from_tensor_slices dataset
        bootstraps: int
            the number of bootstrap dataset resamples to include
        bootstraps: float
            the standard deviation of noise to add to the labels for each bootstrap
        importance_weights: None or tf.Tensor
            an additional importance weight tensor to include in the dataset

        Returns:

        training_dataset: tf.data.Dataset
            a tensorflow dataset that has been batched and prefetched
            with an optional sample weight included
        validation_dataset: tf.data.Dataset
            a tensorflow dataset that has been batched and prefetched
            with an optional sample weight included
        """

        # shuffle the dataset using a common set of indices
        x = self.x if x is None else x
        y = self.y if y is None else y
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        # create a training and validation split
        x = x[indices]
        y = y[indices]
        train_inputs = [x[val_size:], y[val_size:]]
        validate_inputs = [x[:val_size], y[:val_size]]
        size = x.shape[0] - val_size

        if bootstraps > 0:
            # sample the data set with replacement
            train_inputs.append(tf.stack([
                tf.math.bincount(tf.random.uniform(
                    [size], minval=0, maxval=size, dtype=tf.int32), minlength=size,
                    dtype=tf.float32) for b in range(bootstraps)], axis=1))

            # add noise to the labels to increase diversity
            if bootstraps_noise is not None:
                train_inputs.append(bootstraps_noise *
                                    tf.random.normal([size, bootstraps]))

        # possibly add importance weights to the data set
        if importance_weights is not None:
            train_inputs.append(importance_weights[indices[val_size:]])

        # build the parallel tensorflow data loading pipeline
        train = tf.data.Dataset.from_tensor_slices(tuple(train_inputs))
        validate = tf.data.Dataset.from_tensor_slices(tuple(validate_inputs))
        train = train.shuffle(size)
        validate = validate.shuffle(val_size)

        # batch and prefetch each data set
        train = train.batch(batch_size)
        validate = validate.batch(batch_size)
        return train.prefetch(tf.data.experimental.AUTOTUNE),\
            validate.prefetch(tf.data.experimental.AUTOTUNE)

    def score_np(self, x):
        """Calculates a score for the provided tensor x using a ground truth
        oracle function (the goal of the task is to maximize this)

        Args:

        x: np.ndarray
            a batch of sampled designs that will be evaluated by
            an oracle score function

        Returns:

        scores: np.ndarray
            a batch of scores that correspond to the x values provided
            in the function argument
        """

        return self.wrapped_task.score(
            x).astype(np.float32).reshape([-1, 1])

    @tf.function(experimental_relax_shapes=True)
    def score(self, x):
        """Calculates a score for the provided tensor x using a ground truth
        oracle function (the goal of the task is to maximize this)

        Args:

        x: np.ndarray
            a batch of sampled designs that will be evaluated by
            an oracle score function

        Returns:

        scores: np.ndarray
            a batch of scores that correspond to the x values provided
            in the function argument
        """

        return tf.numpy_function(self.score_np, [x], tf.float32)
