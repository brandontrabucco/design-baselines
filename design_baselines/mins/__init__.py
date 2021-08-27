from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.mins.replay_buffer import ReplayBuffer
from design_baselines.mins.trainers import Ensemble
from design_baselines.mins.trainers import WeightedGAN
from design_baselines.mins.nets import ForwardModel
from design_baselines.mins.nets import Discriminator
from design_baselines.mins.nets import DiscreteGenerator
from design_baselines.mins.nets import ContinuousGenerator
from design_baselines.mins.nets import ConvDiscriminator
from design_baselines.mins.nets import DiscreteConvGenerator
from design_baselines.mins.nets import ContinuousConvGenerator
from design_baselines.mins.utils import get_weights
from design_baselines.mins.utils import get_synthetic_data
import tensorflow as tf
import  numpy as np
import os


def mins(config):
    """Optimize a design problem score using the algorithm MINS
    otherwise known as Model Inversion Networks

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])

    if config['normalize_ys']:
        task.map_normalize_y()
    if config['normalize_xs']:
        task.map_normalize_x()

    x = task.x
    y = task.y

    def map_to_probs(x, *rest):
        x = task.to_logits(x)
        x = tf.pad(x, [[0, 0]] * (len(x.shape) - 1) + [[1, 0]])
        return (tf.math.softmax(x / 1e-5), *rest)

    input_shape = x.shape[1:]
    if task.is_discrete:
        input_shape = list(x.shape[1:]) + [task.num_classes]

    base_temp = config.get('base_temp', None)

    if config['offline']:

        # make several keras neural networks with two hidden layers
        forward_models = [ForwardModel(
            input_shape,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            initial_max_std=config['initial_max_std'],
            initial_min_std=config['initial_min_std'])
            for _ in range(config['bootstraps'])]

        # create a trainer for a forward model with a conservative objective
        oracle = Ensemble(forward_models,
                          forward_model_optim=tf.keras.optimizers.Adam,
                          forward_model_lr=config['oracle_lr'],
                          is_discrete=task.is_discrete,
                          noise_std=config.get('noise_std', 0.0),
                          keep=config.get('keep', 1.0),
                          temp=config.get('temp', 0.001))

        # build a bootstrapped data set
        train_data, val_data = build_pipeline(
            x=x, y=y, bootstraps=config['bootstraps'],
            batch_size=config['oracle_batch_size'],
            val_size=config['val_size'], buffer=1)

        if task.is_discrete:
            train_data = train_data.map(
                map_to_probs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_data = val_data.map(
                map_to_probs, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # train the model for an additional number of epochs
        oracle.launch(train_data,
                      val_data,
                      logger,
                      config['oracle_epochs'])

    disc_class = Discriminator
    dgen_class = DiscreteGenerator
    cgen_class = ContinuousGenerator

    if config['use_conv']:

        # use a convolutional architecture for the GAN
        disc_class = ConvDiscriminator
        dgen_class = DiscreteConvGenerator
        cgen_class = ContinuousConvGenerator

    if task.is_discrete:

        # build a Gumbel-Softmax GAN to sample discrete outputs
        explore_gen = dgen_class(
            input_shape, config['latent_size'],
            hidden=config['hidden_size'])
        exploit_gen = dgen_class(
            input_shape, config['latent_size'],
            hidden=config['hidden_size'])

    else:

        # build an LS-GAN to sample continuous outputs
        explore_gen = cgen_class(
            input_shape, config['latent_size'],
            hidden=config['hidden_size'])
        exploit_gen = cgen_class(
            input_shape, config['latent_size'],
            hidden=config['hidden_size'])

    # build the neural network GAN components
    explore_discriminator = disc_class(
        input_shape,
        hidden=config['hidden_size'],
        method=config['method'])
    explore_gan = WeightedGAN(
        explore_gen, explore_discriminator,
        critic_frequency=config['critic_frequency'],
        flip_frac=config['flip_frac'],
        fake_pair_frac=config['fake_pair_frac'],
        penalty_weight=config['penalty_weight'],
        generator_lr=config['generator_lr'],
        generator_beta_1=config['generator_beta_1'],
        generator_beta_2=config['generator_beta_2'],
        discriminator_lr=config['discriminator_lr'],
        discriminator_beta_1=config['discriminator_beta_1'],
        discriminator_beta_2=config['discriminator_beta_2'],
        is_discrete=task.is_discrete,
        noise_std=config.get('noise_std', 0.0),
        keep=config.get('keep', 1.0),
        start_temp=config.get('start_temp', 5.0),
        final_temp=config.get('final_temp', 1.0))

    # build the neural network GAN components
    exploit_discriminator = disc_class(
        input_shape,
        hidden=config['hidden_size'],
        method=config['method'])
    exploit_gan = WeightedGAN(
        exploit_gen, exploit_discriminator,
        critic_frequency=config['critic_frequency'],
        flip_frac=config['flip_frac'],
        fake_pair_frac=config['fake_pair_frac'],
        penalty_weight=config['penalty_weight'],
        generator_lr=config['generator_lr'],
        generator_beta_1=config['generator_beta_1'],
        generator_beta_2=config['generator_beta_2'],
        discriminator_lr=config['discriminator_lr'],
        discriminator_beta_1=config['discriminator_beta_1'],
        discriminator_beta_2=config['discriminator_beta_2'],
        is_discrete=task.is_discrete,
        noise_std=config.get('noise_std', 0.0),
        keep=config.get('keep', 1.0),
        start_temp=config.get('start_temp', 5.0),
        final_temp=config.get('final_temp', 1.0))

    # build a weighted data set using newly collected samples
    train_data, val_data = build_pipeline(
        x=x, y=y, w=get_weights(y, base_temp=base_temp),
        batch_size=config['gan_batch_size'],
        val_size=config['val_size'], buffer=1)

    if task.is_discrete:
        train_data = train_data.map(
            map_to_probs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_data = val_data.map(
            map_to_probs, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # train the gan for several epochs
    explore_gan.launch(
        train_data, val_data, logger, config['initial_epochs'],
        header="exploration/")

    # sample designs from the GAN and evaluate them
    condition_ys = tf.tile(tf.reduce_max(
        y, keepdims=True), [config['solver_samples'], 1])

    # record score percentiles
    logger.record("exploration/condition_ys",
                  task.denormalize_y(condition_ys)
                  if task.is_normalized_y else condition_ys,
                  0,
                  percentile=True)

    # train the gan for several epochs
    exploit_gan.launch(
        train_data, val_data, logger, config['initial_epochs'],
        header="exploitation/")

    # record score percentiles
    logger.record("exploitation/condition_ys",
                  task.denormalize_y(condition_ys)
                  if task.is_normalized_y else condition_ys,
                  0,
                  percentile=True)

    # prevent the temperature from being annealed further
    if task.is_discrete:
        explore_gan.start_temp = explore_gan.final_temp
        exploit_gan.start_temp = exploit_gan.final_temp

    # train the gan using an importance sampled data set
    for iteration in range(config['iterations']):

        # generate synthetic x paired with high performing scores
        tilde_x, tilde_y = get_synthetic_data(
            x, y,
            exploration_samples=config['exploration_samples'],
            exploration_rate=config['exploration_rate'],
            base_temp=base_temp)

        # build a weighted data set using newly collected samples
        train_data, val_data = build_pipeline(
            x=tilde_x.numpy(), y=tilde_y.numpy(),
            w=get_weights(tilde_y.numpy(), base_temp=base_temp),
            batch_size=config['gan_batch_size'],
            val_size=config['val_size'], buffer=1)

        if task.is_discrete:
            train_data = train_data.map(
                map_to_probs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_data = val_data.map(
                map_to_probs, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # train the gan for several epochs
        explore_gan.launch(
            train_data, val_data, logger, config['epochs_per_iteration'],
            start_epoch=config['epochs_per_iteration'] * iteration +
                        config['initial_epochs'],
            header="exploration/")

        # sample designs from the GAN and evaluate them
        condition_ys = tf.tile(tf.reduce_max(
            tilde_y, keepdims=True), [config['thompson_samples'], 1])

        # generate samples for exploration
        solver_xs = explore_gen.sample(condition_ys, temp=0.001)
        if task.is_discrete:
            solver_xs = tf.argmax(
                solver_xs, axis=-1, output_type=tf.int32)
        actual_ys = oracle.get_distribution(solver_xs).mean() \
            if config['offline'] else task.predict(solver_xs)

        # record score percentiles
        logger.record("exploration/condition_ys",
                      task.denormalize_y(condition_ys)
                      if task.is_normalized_y else condition_ys,
                      0,
                      percentile=True)
        logger.record("exploration/actual_ys",
                      task.denormalize_y(actual_ys)
                      if task.is_normalized_y else actual_ys,
                      0,
                      percentile=True)

        # concatenate newly paired samples with the existing data set
        x = tf.concat([x, solver_xs], 0)
        y = tf.concat([y, actual_ys], 0)

        # build a weighted data set using newly collected samples
        train_data, val_data = build_pipeline(
            x=x.numpy(), y=y.numpy(),
            w=get_weights(y.numpy(), base_temp=base_temp),
            batch_size=config['gan_batch_size'],
            val_size=config['val_size'], buffer=1)

        if task.is_discrete:
            train_data = train_data.map(
                map_to_probs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_data = val_data.map(
                map_to_probs, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # train the gan for several epochs
        exploit_gan.launch(
            train_data, val_data, logger, config['epochs_per_iteration'],
            start_epoch=config['epochs_per_iteration'] * iteration +
                        config['initial_epochs'],
            header="exploitation/")

        # sample designs from the GAN and evaluate them
        condition_ys = tf.tile(tf.reduce_max(
            y, keepdims=True), [config['solver_samples'], 1])

        # record score percentiles
        logger.record("exploitation/condition_ys",
                      task.denormalize_y(condition_ys)
                      if task.is_normalized_y else condition_ys,
                      0,
                      percentile=True)

    # generate samples for exploration
    solver_xs = exploit_gen.sample(condition_ys, temp=0.001)
    solution = tf.argmax(solver_xs, axis=-1, output_type=tf.int32) \
               if task.is_discrete else solver_xs

    # save the current solution to the disk
    np.save(os.path.join(config["logging_dir"],
                         f"solution.npy"), solution.numpy())
    if config["do_evaluation"]:

        # evaluate the found solution and record a video
        score = task.predict(solution)
        if task.is_normalized_y:
            score = task.denormalize_y(score)
        logger.record("score", score, config['iterations'], percentile=True)
