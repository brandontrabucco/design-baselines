from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.gradient_ascent.trainers import MaximumLikelihood
from design_baselines.gradient_ascent.trainers import Ensemble, VAETrainer
from design_baselines.gradient_ascent.nets import ForwardModel, SequentialVAE
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os


def gradient_ascent(config):
    """Train a Score Function to solve a Model-Based Optimization
    using gradient ascent on the input design

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])

    if config['normalize_ys']:
        task.map_normalize_y()
    if task.is_discrete and not config["use_vae"]:
        task.map_to_logits()
    if config['normalize_xs']:
        task.map_normalize_x()

    x = task.x
    y = task.y

    if task.is_discrete and config["use_vae"]:

        vae_model = SequentialVAE(
            task,
            hidden_size=config['vae_hidden_size'],
            latent_size=config['vae_latent_size'],
            activation=config['vae_activation'],
            kernel_size=config['vae_kernel_size'],
            num_blocks=config['vae_num_blocks'])

        vae_trainer = VAETrainer(vae_model,
                                 vae_optim=tf.keras.optimizers.Adam,
                                 vae_lr=config['vae_lr'],
                                 beta=config['vae_beta'])

        # create the training task and logger
        train_data, val_data = build_pipeline(
            x=x, y=y,
            batch_size=config['vae_batch_size'],
            val_size=config['val_size'])

        # estimate the number of training steps per epoch
        vae_trainer.launch(train_data, val_data,
                           logger, config['vae_epochs'])

        # map the x values to latent space
        x = vae_model.encoder_cnn.predict(x)[0]

        mean = np.mean(x, axis=0, keepdims=True)
        standard_dev = np.std(x - mean, axis=0, keepdims=True)
        x = (x - mean) / standard_dev

    input_shape = x.shape[1:]
    input_size = np.prod(input_shape)

    # make several keras neural networks with different architectures
    forward_models = [ForwardModel(
        input_shape,
        activations=activations,
        hidden_size=config['hidden_size'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for activations in config['activations']]

    # scale the learning rate based on the number of channels in x
    config['solver_lr'] *= np.sqrt(np.prod(x.shape[1:]))

    trs = []
    for i, fm in enumerate(forward_models):

        # create a bootstrapped data set
        train_data, validate_data = build_pipeline(
            x=x, y=y, batch_size=config['batch_size'],
            val_size=config['val_size'], bootstraps=1)

        # create a trainer for a forward model with a conservative objective
        trainer = MaximumLikelihood(
            fm,
            forward_model_optim=tf.keras.optimizers.Adam,
            forward_model_lr=config['forward_model_lr'],
            noise_std=config.get('model_noise_std', 0.0))

        # train the model for an additional number of epochs
        trs.append(trainer)
        trainer.launch(train_data, validate_data, logger,
                       config['epochs'], header=f'oracle_{i}/')

    # select the top k initial designs from the dataset
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    indices = tf.math.top_k(y[:, 0], k=config['solver_samples'])[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = initial_x

    # evaluate the starting point
    solution = x
    if task.is_normalized_y:
        preds = [task.denormalize_y(
            fm.get_distribution(solution).mean())
            for fm in forward_models]
    else:
        preds = [fm.get_distribution(solution).mean()
                 for fm in forward_models]

    # record the prediction and score to the logger
    logger.record("distance/travelled", tf.linalg.norm(solution - initial_x), 0)
    logger.record("distance/from_mean", tf.linalg.norm(solution - mean_x), 0)
    for n, prediction_i in enumerate(preds):
        logger.record(f"oracle_{n}/prediction", prediction_i, 0)
        if n > 0:
            logger.record(f"rank_corr/0_to_{n}",
                          spearman(preds[0][:, 0], prediction_i[:, 0]), 0)

    # perform gradient ascent on the score through the forward model
    for i in range(1, config['solver_steps'] + 1):
        # back propagate through the forward model
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = []
            for fm in forward_models:
                solution = x
                predictions.append(fm.get_distribution(solution).mean())
            if config['aggregation_method'] == 'mean':
                score = tf.reduce_min(predictions, axis=0)
            if config['aggregation_method'] == 'min':
                score = tf.reduce_min(predictions, axis=0)
            if config['aggregation_method'] == 'random':
                score = predictions[np.random.randint(len(predictions))]
        grads = tape.gradient(score, x)

        # use the conservative optimizer to update the solution
        x = x + config['solver_lr'] * grads
        solution = x

        # evaluate the design using the oracle and the forward model
        if task.is_normalized_y:
            preds = [task.denormalize_y(
                fm.get_distribution(solution).mean())
                for fm in forward_models]
        else:
            preds = [fm.get_distribution(solution).mean()
                     for fm in forward_models]

        # record the prediction and score to the logger
        logger.record("distance/travelled", tf.linalg.norm(solution - initial_x), i)
        logger.record("distance/from_mean", tf.linalg.norm(solution - mean_x), i)
        for n, prediction_i in enumerate(preds):
            logger.record(f"oracle_{n}/prediction", prediction_i, i)
            logger.record(f"oracle_{n}/grad_norm", tf.linalg.norm(
                tf.reshape(grads[n], [-1, input_size]), axis=-1), i)
            if n > 0:
                logger.record(f"rank_corr/0_to_{n}",
                              spearman(preds[0][:, 0], prediction_i[:, 0]), i)
                logger.record(f"grad_corr/0_to_{n}", tfp.stats.correlation(
                    grads[0], grads[n], sample_axis=0, event_axis=None), i)

    if task.is_discrete and config["use_vae"]:
        solution = solution * standard_dev + mean
        logits = vae_model.decoder_cnn.predict(solution)
        solution = tf.argmax(logits, axis=2, output_type=tf.int32)

    # save the current solution to the disk
    np.save(os.path.join(config["logging_dir"],
                         f"solution.npy"), solution.numpy())
    if config["do_evaluation"]:

        # evaluate the found solution and record a video
        score = task.predict(solution)
        if task.is_normalized_y:
            score = task.denormalize_y(score)
        logger.record("score", score, config['solver_steps'], percentile=True)
