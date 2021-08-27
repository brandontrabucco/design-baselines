from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.cma_es.trainers import Ensemble, VAETrainer
from design_baselines.cma_es.nets import ForwardModel, SequentialVAE
from tensorflow_probability import distributions as tfpd
import tensorflow as tf
import tensorflow.keras as keras
import os
import math
import numpy as np


def cma_es(config):
    """Optimizes over designs x in an offline optimization problem
    using the CMA Evolution Strategy

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

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        input_shape,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for b in range(config['bootstraps'])]

    # create a trainer for a forward model with a conservative objective
    ensemble = Ensemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['ensemble_lr'])

    # create the training task and logger
    train_data, val_data = build_pipeline(
        x=x, y=y, bootstraps=config['bootstraps'],
        batch_size=config['ensemble_batch_size'],
        val_size=config['val_size'])

    # train the model for an additional number of epochs
    ensemble.launch(train_data,
                    val_data,
                    logger,
                    config['ensemble_epochs'])

    # select the top 1 initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=config['solver_samples'])[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = initial_x

    # create a fitness function for optimizing the expected task score
    def fitness(input_x):
        input_x = tf.reshape(input_x, input_shape)[tf.newaxis]
        if config["optimize_ground_truth"]:
            if task.is_discrete and config["use_vae"]:
                input_x = tf.argmax(vae_model.decoder_cnn.predict(
                    input_x * standard_dev + mean), axis=2, output_type=tf.int32)
            value = task.predict(input_x)
        else:
            value = ensemble.get_distribution(input_x).mean()
        return (-value[0].numpy()).tolist()[0]

    import cma
    result = []
    for i in range(config['solver_samples']):
        xi = x[i].numpy().flatten().tolist()
        es = cma.CMAEvolutionStrategy(xi, config['cma_sigma'])
        step = 0
        while not es.stop() and step < config['cma_max_iterations']:
            solutions = es.ask()
            es.tell(solutions, [fitness(x) for x in solutions])
            step += 1
        result.append(
            tf.reshape(es.result.xbest, input_shape))
        print(f"CMA: {i + 1} / {config['solver_samples']}")

    # convert the solution found by CMA-ES to a tensor
    x = tf.stack(result, axis=0)
    solution = x

    if task.is_discrete and config["use_vae"]:
        solution = solution * standard_dev + mean
        logits = vae_model.decoder_cnn.predict(solution)
        solution = tf.argmax(logits, axis=2, output_type=tf.int32)

    # save the current solution to the disk
    np.save(os.path.join(config["logging_dir"],
                         f"solution.npy"), solution.numpy())
    if config["do_evaluation"]:

        # evaluate the found solution
        score = task.predict(solution)
        if task.is_normalized_y:
            score = task.denormalize_y(score)
        logger.record("score", score, 0, percentile=True)
