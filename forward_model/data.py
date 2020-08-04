from forward_model.utils import get_weights
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
            additional keyword arguments that are passed to the design_eanch task
            when it is created using design_bench.make
        """

        self.wrapped_task = make(task_name, **task_kwargs)

    @property
    def x(self):
        return self.wrapped_task.x.astype(np.float32)

    @property
    def y(self):
        return self.wrapped_task.y.astype(np.float32).reshape([-1, 1])

    @x.setter
    def x(self, x):
        self.wrapped_task.x = x

    @y.setter
    def y(self, y):
        self.wrapped_task.y = y

    @property
    def input_shape(self):
        return self.wrapped_task.input_shape

    @property
    def input_size(self):
        return self.wrapped_task.input_size

    def score_np(self, x):
        return self.wrapped_task.score(x)

    def build(self,
              x=None,
              y=None,
              val_size=200,
              batch_size=128,
              bootstraps=0,
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

        x = self.x if x is None else x
        y = self.y if y is None else y

        # shuffle the dataset using a common set of indices
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        # create a training and validation split
        train_inputs = [x[val_size:], y[val_size:]]
        validate_inputs = [x[:val_size], y[:val_size]]
        size = x.shape[0] - val_size

        # possible add a bootstrap mask to the data set
        if bootstraps > 0:
            train_inputs.append(tf.stack([
                tf.math.bincount(tf.random.uniform(
                    [size],
                    minval=0, maxval=size, dtype=tf.int32),
                    minlength=size, dtype=tf.float32)
                for b in range(bootstraps)], axis=1))

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
