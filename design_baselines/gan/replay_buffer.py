import tensorflow as tf


class ReplayBuffer(tf.Module):

    def __init__(self, capacity, shape):
        """A static graph replay buffer that stores samples collected
        from a generator in a GAN

        Args:

        capacity: tf.dtypes.int32
            the number of samples that can be in the buffer, maximum
            smaller number = the algorithm is more online
        shape: tf.dtypes.int32
            the shape of the tensors stored in the buffer
            this will be a 3-tensor for images
        """

        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.shape = shape

        # prepare a storage memory for samples
        self.xs = tf.Variable(
            tf.zeros([capacity, *shape], tf.dtypes.float32))

        # save size statistics for the buffer
        self.head = tf.Variable(tf.constant(0))
        self.size = tf.Variable(tf.constant(0))
        self.step = tf.Variable(tf.constant(0))

        # a tensor for indexing into scatter_nd_update
        self.idx = tf.concat([tf.tile(tf.reshape(
            tf.range(shape_i),
            [1] * i + [shape_i] + [1] * (len(shape) - i)),
            list(shape[:i]) + [1] + list(shape[i + 1:]) + [1])
            for i, shape_i in enumerate(shape)], axis=len(shape))

    @tf.function
    def insert(self, x):
        """Insert a single sample collected from the environment into
        the replay buffer at the current head position

        Args:

        x: tf.dtypes.float32
            a tensor corresponding to a fake sample from the generator
            images may be shaped like [height, width, channels]
        """

        # create scatter indices for one sample
        loc_x = tf.reshape(self.head, [1] * (len(self.shape) + 1))
        loc_x = tf.broadcast_to(loc_x, tf.shape(self.idx))[..., 0:1]
        loc_x = tf.concat([loc_x, self.idx], axis=len(self.shape))

        # insert samples at the position of the current head
        self.xs.assign(tf.tensor_scatter_nd_update(
            self.xs, loc_x, tf.cast(x, tf.dtypes.float32)))

        # increment the size statistics of the buffer
        self.head.assign(
            tf.math.floormod(self.head + 1, self.capacity))
        self.size.assign(
            tf.minimum(self.size + 1, self.capacity))
        self.step.assign(
            self.step + 1)

    @tf.function
    def insert_many(self, xs):
        """Insert a single sample collected from the environment into
        the replay buffer at the current head position

        Args:

        xs: tf.dtypes.float32
            a tensor corresponding to many fake samples from the generator
            images may be shaped like [batch, height, width, channels]
        """

        for i in tf.range(tf.shape(xs)[0]):
            self.insert(xs[i])

    @tf.function
    def sample(self, batch_size):
        """Samples a batch of training data from the replay buffer
        and returns the batch of data

        Args:

        batch_size: tf.dtypes.int32
            a scalar tensor that specifies how many elements to sample
            typically smaller than the replay buffer capacity

        Returns:

        xs: tf.dtypes.float32
            a tensor corresponding to many fake samples from the generator
            images may be shaped like [batch, height, width, channels]
        """

        return tf.gather(self.xs, tf.random.uniform([
            batch_size], maxval=self.size, dtype=tf.int32), axis=0)
