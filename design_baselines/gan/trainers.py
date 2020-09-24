from design_baselines.utils import disc_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
import tensorflow as tf


class GAN(tf.Module):

    def __init__(self,
                 generator,
                 discriminator,
                 pool,
                 critic_frequency=5,
                 flip_frac=0.0,
                 pool_frac=0.0,
                 pool_save=0,
                 fake_pair_frac=0.0,
                 penalty_weight=0.0,
                 generator_lr=0.0002,
                 generator_beta_1=0.5,
                 generator_beta_2=0.999,
                 discriminator_lr=0.0002,
                 discriminator_beta_1=0.5,
                 discriminator_beta_2=0.999,
                 is_discrete=False,
                 noise_std=0.0,
                 keep=0.99,
                 start_temp=5.0,
                 final_temp=1.0):
        """Build a trainer for an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        generator: tf.keras.Model
            the generator model in a generative adversarial network
            conditioned on gaussian noise and target y values
        discriminator: tf.keras.Model
            the discriminator model in a generative adversarial network
            conditioned on designs x and target y values
        pool: ReplayBuffer
            a replay buffer that is able to store previously generated
            designs x from the gan up to a certain capacity
        critic_frequency: int
            the number of critic gradient descent steps on different batches
            to take before optimizing the generator
        flip_frac: float
            the probability of flipping the labels of real samples
            when training the discriminator
        pool_frac: float
            the fraction of the fake loss taken from samples of
            designs from the replay pool
        pool_save: int
            the number of designs samples from the current generator to
            store in the replay buffer at every step
        fake_pair_frac: float
            the fraction of the fake loss taken from samples of
            fake pairs of real samples
        penalty_weight: float
            the weight of the gradient penalty on the discriminator
            in the discriminator loss function
        generator_lr: float
            the learning rate in the ADAM optimizer for the
            generator model
        generator_beta_1: float
            the beta_1 in the ADAM optimizer for the
            generator model
        generator_beta_2: float
            the beta_2 in the ADAM optimizer for the
            generator model
        discriminator_lr: float
            the learning rate in the ADAM optimizer for the
            discriminator model
        discriminator_beta_1: float
            the beta_1 in the ADAM optimizer for the
            discriminator model
        discriminator_beta_2: float
            the beta_2 in the ADAM optimizer for the
            discriminator model
        is_discrete: bool
            a boolean that indicates whether the designs x are discrete
            samples or continuous samples
        noise_std: float
            if designs x are continuous this specifies the standard
            deviation of gaussian noise added to real samples
        keep: float
            if designs x are discrete this specifies the amount of
            probability mass of the on location
        start_temp: float
            if designs x are discrete this specifies the initial
            temperature of the discrete noise
        final_temp: float
            if designs x are discrete this specifies the final
            temperature of the discrete noise
        """

        super().__init__()
        self.is_discrete = is_discrete
        self.noise_std = noise_std
        self.keep = keep
        self.critic_frequency = critic_frequency
        self.penalty_weight = penalty_weight
        self.fake_pair_frac = fake_pair_frac
        self.flip_frac = flip_frac

        self.start_temp = start_temp
        self.final_temp = final_temp
        self.temp = tf.Variable(0.0, dtype=tf.float32)

        self.pool = pool
        self.pool_frac = pool_frac
        self.pool_save = pool_save

        # create optimizers for the generator
        self.generator = generator
        self.generator_optim = tf.keras.optimizers.Adam(
            learning_rate=generator_lr,
            beta_1=generator_beta_1,
            beta_2=generator_beta_2)

        # create optimizers for the discriminator
        self.discriminator = discriminator
        self.discriminator_optim = tf.keras.optimizers.Adam(
            learning_rate=discriminator_lr,
            beta_1=discriminator_beta_1,
            beta_2=discriminator_beta_2)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   i,
                   x_real):
        """Perform a training step for a generator and a discriminator
        using a least squares objective function

        Args:

        x_real: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y_real: tf.Tensor
            a batch of training labels shaped like [batch_size, 1]
        w: tf.Tensor
            importance sampling weights shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()
        batch_dim = tf.shape(x_real)[0]

        # corrupt the inputs with noise
        if self.is_discrete:
            x_real = disc_noise(x_real, keep=self.keep, temp=self.temp)
        else:
            x_real = cont_noise(x_real, self.noise_std)

        with tf.GradientTape() as tape:

            # evaluate the discriminator on generated samples
            x_fake = self.generator.sample(batch_dim,
                                           temp=self.temp, training=False)
            p_fake, d_fake, acc_fake = self.discriminator.loss(
                x_fake, tf.zeros([batch_dim, 1]), training=False)

            statistics[f'generator/train/x_fake'] = x_fake
            statistics[f'discriminator/train/p_fake'] = p_fake
            statistics[f'discriminator/train/d_fake'] = d_fake
            statistics[f'discriminator/train/acc_fake'] = acc_fake

            # normalize the fake evaluation metrics
            d_fake = d_fake * (1.0 - self.fake_pair_frac - self.pool_frac)

            x_pair = tf.zeros_like(x_fake)
            p_pair = tf.zeros_like(p_fake)
            d_pair = tf.zeros_like(d_fake)
            acc_pair = tf.zeros_like(acc_fake)

            if self.fake_pair_frac > 0:

                # evaluate the discriminator on fake pairs of real inputs
                x_pair = tf.random.shuffle(x_real)
                p_pair, d_pair, acc_pair = self.discriminator.loss(
                    x_pair, tf.zeros([batch_dim, 1]), training=False)

                # average the metrics between fake samples
                d_fake = d_pair * self.fake_pair_frac + d_fake

            statistics[f'generator/train/x_pair'] = x_pair
            statistics[f'discriminator/train/p_pair'] = p_pair
            statistics[f'discriminator/train/d_pair'] = d_pair
            statistics[f'discriminator/train/acc_pair'] = acc_pair

            x_pool = tf.zeros_like(x_fake)
            p_pool = tf.zeros_like(p_fake)
            d_pool = tf.zeros_like(d_fake)
            acc_pool = tf.zeros_like(acc_fake)

            if self.pool.size > batch_dim and self.pool_frac > 0:

                # evaluate discriminator on samples from a replay buffer
                x_pool, y_pool = self.pool.sample(batch_dim)
                p_pool, d_pool, acc_pool = self.discriminator.loss(
                    x_pool, y_pool, tf.zeros([batch_dim, 1]), training=False)

                # average the metrics between fake samples
                d_fake = d_pool * self.pool_frac + d_fake

            statistics[f'generator/train/x_pool'] = x_pool
            statistics[f'discriminator/train/p_pool'] = p_pool
            statistics[f'discriminator/train/d_pool'] = d_pool
            statistics[f'discriminator/train/acc_pool'] = acc_pool

            if self.pool_save > 0:

                # possibly add more generated samples to the replay pool
                self.pool.insert_many(x_fake[:self.pool_save])

            # evaluate the discriminator on real inputs
            labels = tf.cast(self.flip_frac <=
                             tf.random.uniform([batch_dim, 1]), tf.float32)
            p_real, d_real, acc_real = self.discriminator.loss(
                x_real, labels, training=True)

            statistics[f'generator/train/x_real'] = x_real
            statistics[f'discriminator/train/p_real'] = p_real
            statistics[f'discriminator/train/d_real'] = d_real
            statistics[f'discriminator/train/acc_real'] = acc_real

            # evaluate a gradient penalty on interpolations
            e = tf.random.uniform([batch_dim] + [1] * (len(x_fake.shape) - 1))
            x_interp = x_real * e + x_fake * (1 - e)
            penalty = self.discriminator.penalty(x_interp, training=False)

            statistics[f'discriminator/train/neg_critic_loss'] = -(d_real + d_fake)
            statistics[f'discriminator/train/penalty'] = penalty

            # build the total loss
            total_loss = tf.reduce_mean(
                d_real + d_fake + self.penalty_weight * penalty)

        var_list = self.discriminator.trainable_variables
        self.discriminator_optim.apply_gradients(zip(
            tape.gradient(total_loss, var_list), var_list))

        if tf.equal(tf.math.floormod(i, self.critic_frequency), 0):

            with tf.GradientTape() as tape:

                # evaluate the discriminator on generated samples
                x_fake = self.generator.sample(batch_dim,
                                               temp=self.temp, training=True)
                p_fake, d_fake, acc_fake = self.discriminator.loss(
                    x_fake, tf.ones([batch_dim, 1]), training=False)

                # build the total loss
                total_loss = tf.reduce_mean(d_fake)

            var_list = self.generator.trainable_variables
            self.generator_optim.apply_gradients(zip(
                tape.gradient(total_loss, var_list), var_list))

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      x_real):
        """Perform a validation step for a generator and a discriminator
        using a least squares objective function

        Args:

        x_real: tf.Tensor
            a batch of validation inputs shaped like [batch_size, channels]
        y_real: tf.Tensor
            a batch of validation labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()
        batch_dim = tf.shape(x_real)[0]

        # corrupt the inputs with noise
        if self.is_discrete:
            x_real = disc_noise(x_real, keep=self.keep, temp=self.temp)
        else:
            x_real = cont_noise(x_real, self.noise_std)

        # evaluate the discriminator on generated samples
        x_fake = self.generator.sample(batch_dim,
                                       temp=self.temp, training=False)
        p_fake, d_fake, acc_fake = self.discriminator.loss(
            x_fake, tf.zeros([batch_dim, 1]), training=False)

        statistics[f'generator/validate/x_fake'] = x_fake
        statistics[f'discriminator/validate/p_fake'] = p_fake
        statistics[f'discriminator/validate/d_fake'] = d_fake
        statistics[f'discriminator/validate/acc_fake'] = acc_fake

        x_pair = tf.zeros_like(x_fake)
        p_pair = tf.zeros_like(p_fake)
        d_pair = tf.zeros_like(d_fake)
        acc_pair = tf.zeros_like(acc_fake)

        if self.fake_pair_frac > 0:

            # evaluate the discriminator on fake pairs of real inputs
            x_pair = tf.random.shuffle(x_real)
            p_pair, d_pair, acc_pair = self.discriminator.loss(
                x_pair, tf.zeros([batch_dim, 1]), training=False)

        statistics[f'generator/validate/x_pair'] = x_pair
        statistics[f'discriminator/validate/p_pair'] = p_pair
        statistics[f'discriminator/validate/d_pair'] = d_pair
        statistics[f'discriminator/validate/acc_pair'] = acc_pair

        x_pool = tf.zeros_like(x_fake)
        p_pool = tf.zeros_like(p_fake)
        d_pool = tf.zeros_like(d_fake)
        acc_pool = tf.zeros_like(acc_fake)

        if self.pool.size > batch_dim and self.pool_frac > 0:

            # evaluate discriminator on samples from a replay buffer
            x_pool, y_pool = self.pool.sample(batch_dim)
            p_pool, d_pool, acc_pool = self.discriminator.loss(
                x_pool, y_pool, tf.zeros([batch_dim, 1]), training=False)

        statistics[f'generator/validate/x_pool'] = x_pool
        statistics[f'discriminator/validate/p_pool'] = p_pool
        statistics[f'discriminator/validate/d_pool'] = d_pool
        statistics[f'discriminator/validate/acc_pool'] = acc_pool

        # evaluate the discriminator on real inputs
        p_real, d_real, acc_real = self.discriminator.loss(
            x_real, tf.ones([batch_dim, 1]), training=False)

        statistics[f'generator/validate/x_real'] = x_real
        statistics[f'discriminator/validate/p_real'] = p_real
        statistics[f'discriminator/validate/d_real'] = d_real
        statistics[f'discriminator/validate/acc_real'] = acc_real

        # evaluate a gradient penalty on interpolations
        e = tf.random.uniform([batch_dim] + [1] * (len(x_fake.shape) - 1))
        x_interp = x_real * e + x_fake * (1 - e)
        penalty = self.discriminator.penalty(x_interp, training=False)

        statistics[f'discriminator/validate/neg_critic_loss'] = -(d_real + d_fake)
        statistics[f'discriminator/validate/penalty'] = penalty

        return statistics

    def train(self,
              dataset):
        """Perform training using gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for i, (x, y, w) in enumerate(dataset):
            i = tf.convert_to_tensor(i)
            for name, tensor in self.train_step(i, x, y, w).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def validate(self,
                 dataset):
        """Perform validation on an ensemble of models without
        using bootstrapping weights

        Args:

        dataset: tf.data.Dataset
            the validation dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def launch(self,
               train_data,
               validate_data,
               logger,
               epochs,
               start_epoch=0,
               header=""):
        """Launch training and validation for the model for the specified
        number of epochs, and log statistics

        Args:

        train_data: tf.data.Dataset
            the training dataset already batched and prefetched
        validate_data: tf.data.Dataset
            the validation dataset already batched and prefetched
        logger: Logger
            an instance of the logger used for writing to tensor board
        epochs: int
            the number of epochs through the data sets to take
        """

        for e in range(epochs):
            self.temp.assign(self.final_temp * e / (epochs - 1) +
                             self.start_temp * (1.0 - e / (epochs - 1)))
            for name, loss in self.train(train_data).items():
                logger.record(header + name, loss, start_epoch + e)
            for name, loss in self.validate(validate_data).items():
                logger.record(header + name, loss, start_epoch + e)

    def get_saveables(self):
        """Collects and returns stateful objects that are serializeable
        using the tensorflow checkpoint format

        Returns:

        saveables: dict
            a dict containing stateful objects compatible with checkpoints
        """

        saveables = dict()
        saveables['generator'] = self.generator
        saveables['discriminator'] = self.discriminator
        saveables['generator_optim'] = self.generator_optim
        saveables['discriminator_optim'] = self.discriminator_optim
        saveables['temp'] = self.temp
        saveables['pool'] = self.pool
        return saveables
