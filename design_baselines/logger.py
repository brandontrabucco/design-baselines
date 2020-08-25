import tensorflow as tf
import tensorflow_probability as tfp


class Logger(object):

    def __init__(self,
                 logging_dir):
        """Creates a logging interface to a tensorboard file for
        visualizing in the tensorboard web interface; note that
        mean, max, min, and std are recorded

        Arguments:

        logging_dir: str
            the path on the disk to save records to
        """

        tf.io.gfile.makedirs(logging_dir)
        self.writer = tf.summary.create_file_writer(logging_dir)

    def record(self,
               key,
               value,
               step):
        """Log statistics about training data to tensorboard
        log files for visualization later

        Arguments:

        key: str
            the string name to use when logging data in tensorboard
            that determines groupings in the web interface
        value: tf.tensor
            the tensor of values to record statistics about
            typically is multi dimensional
        step: int
            the total number of environment steps collected so far
            typically on intervals of 10000
        """

        step = tf.cast(tf.convert_to_tensor(step), tf.int64)
        with self.writer.as_default():

            # log several statistics of the incoming tensors
            tf.summary.scalar(key + '/100th',
                              tfp.stats.percentile(value, 100),
                              step=step)
            tf.summary.scalar(key + '/90th',
                              tfp.stats.percentile(value, 90),
                              step=step)
            tf.summary.scalar(key + '/80th',
                              tfp.stats.percentile(value, 80),
                              step=step)
            tf.summary.scalar(key + '/50th',
                              tfp.stats.percentile(value, 50),
                              step=step)
