from forward_model.utils import get_weights
from design_bench.task import Task
from design_bench import make
import tensorflow as tf
import numpy as np


class StaticGraphTask(Task):

    def __init__(self,
                 task_name,
                 val_size=200,
                 batch_size=128,
                 **task_kwargs):
        """An interface to a static-graph task which includes a validation
        set and a non differentiable score function

        Args:

        task_name: str
            the name to a valid task using design_bench.make(task_name)
            such as 'HopperController-v0'
        val_size: int
            the size of the validation split to use when building the tensorflow
            Dataset.from_tensor_slices dataset
        **task_kwargs: dict
            additional keyword arguments that are passed to the design_eanch task
            when it is created using design_bench.make
        """

        self.wrapped_task = make(task_name, **task_kwargs)
        self.val_size = val_size
        self.batch_size = batch_size

    @property
    def x(self):
        return self.wrapped_task.x.reshape(
            [-1, self.wrapped_task.input_size]).astype(np.float32)

    @property
    def y(self):
        return self.wrapped_task.y

    @x.setter
    def x(self, x):
        self.wrapped_task.x = x.reshape(
            [-1, *self.wrapped_task.input_shape])

    @y.setter
    def y(self, y):
        self.wrapped_task.y = y

    @property
    def input_shape(self):
        return self.wrapped_task.input_size,

    @property
    def input_size(self):
        return self.wrapped_task.input_size

    def score_np(self, x):
        return self.wrapped_task.score(
            x.reshape([-1, *self.wrapped_task.input_shape]))

    def build(self,
              include_weights=False):
        """Provide an interface for splitting a task into a training and
        validation set and including sample re-weighting

        Args:

        include_weights: bool
            whether to build a dataset that includes sample weights

        Returns:

        training_dataset: tf.data.Dataset
            a tensorflow dataset that has been batched and prefetched
            with an optional sample weight included
        validation_dataset: tf.data.Dataset
            a tensorflow dataset that has been batched and prefetched
            with an optional sample weight included
        """

        indices = np.arange(self.x.shape[0])
        np.random.shuffle(indices)
        x = self.x[indices]
        y = self.y[indices]

        train_inputs = [x[self.val_size:], y[self.val_size:]]
        validate_inputs = [x[:self.val_size], y[:self.val_size]]

        if include_weights:
            train_inputs.append(get_weights(y[self.val_size:]))
            validate_inputs.append(get_weights(y[:self.val_size]))

        train = tf.data.Dataset.from_tensor_slices(tuple(train_inputs))
        validate = tf.data.Dataset.from_tensor_slices(tuple(validate_inputs))

        train = train.shuffle(x.shape[0] - self.val_size)
        validate = validate.shuffle(self.val_size)

        train = train.batch(self.batch_size)
        validate = validate.batch(self.batch_size)
        return train.prefetch(tf.data.experimental.AUTOTUNE),\
            validate.prefetch(tf.data.experimental.AUTOTUNE)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
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
