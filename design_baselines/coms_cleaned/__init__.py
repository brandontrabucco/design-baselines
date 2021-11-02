from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.coms_cleaned.trainers import ConservativeObjectiveModel
from design_baselines.coms_cleaned.trainers import VAETrainer
from design_baselines.coms_cleaned.nets import ForwardModel
from design_baselines.coms_cleaned.nets import SequentialVAE
import tensorflow as tf
import numpy as np
import os
import click
import json


@click.command()
@click.option('--logging-dir',
              default='coms-cleaned', type=str,
              help='The directory in which tensorboard data is logged '
                   'during the experiment.')
@click.option('--task', type=str,
              default='HopperController-Exact-v0',
              help='The name of the design-bench task to use during '
                   'the experiment.')
@click.option('--task-relabel/--no-task-relabel',
              default=True, type=bool,
              help='Whether to relabel the real Offline MBO data with '
                   'predictions made by the oracle (this eliminates a '
                   'train-test discrepency if the oracle is not an '
                   'adequate model of the data).')
@click.option('--task-max-samples',
              default=None, type=int,
              help='The maximum number of samples to include in the task '
                   'visible training set, which can be left as None to not '
                   'further subsample the training set.')
@click.option('--task-distribution',
              default=None, type=str,
              help='The empirical distribution to be used when further '
                   'subsampling the training set to the specific '
                   'task_max_samples from the previous run argument.')
@click.option('--normalize-ys/--no-normalize-ys',
              default=True, type=bool,
              help='Whether to normalize the y values in the Offline MBO '
                   'dataset before performing model-based optimization.')
@click.option('--normalize-xs/--no-normalize-xs',
              default=True, type=bool,
              help='Whether to normalize the x values in the Offline MBO '
                   'dataset before performing model-based optimization. '
                   '(note that x must not be discrete)')
@click.option('--in-latent-space/--not-in-latent-space',
              default=False, type=bool,
              help='Whether to embed the designs into the latent space of '
                   'a VAE before performing model-based optimization '
                   '(based on Gomez-Bombarelli et al. 2018).')
@click.option('--vae-hidden-size',
              default=64, type=int,
              help='The hidden size of the neural network encoder '
                   'and decoder models used in the VAE.')
@click.option('--vae-latent-size',
              default=256, type=int,
              help='The size of the VAE latent vector space.')
@click.option('--vae-activation',
              default='relu', type=str,
              help='The activation function used in the VAE.')
@click.option('--vae-kernel-size',
              default=3, type=int,
              help='When the VAE is a CNN the kernel size of kernel '
                   'tensor in convolution layers.')
@click.option('--vae-num-blocks',
              default=4, type=int,
              help='The number of convolution blocks operating at '
                   'different spatial resolutions.')
@click.option('--vae-lr',
              default=0.0003, type=float,
              help='The learning rate of the VAE.')
@click.option('--vae-beta',
              default=1.0, type=float,
              help='The weight of the KL loss when training the VAE.')
@click.option('--vae-batch-size',
              default=32, type=int,
              help='The batch size used to train the VAE.')
@click.option('--vae-val-size',
              default=200, type=int,
              help='The number of samples in the VAE validation set.')
@click.option('--vae-epochs',
              default=10, type=int,
              help='The number of epochs to train the VAE.')
@click.option('--particle-lr',
              default=0.05, type=float,
              help='The learning rate used in the COMs inner loop.')
@click.option('--particle-train-gradient-steps',
              default=50, type=int,
              help='The number of gradient ascent steps used in the '
                   'COMs inner loop.')
@click.option('--particle-evaluate-gradient-steps',
              default=50, type=int,
              help='The number of gradient ascent steps used in the '
                   'COMs inner loop.')
@click.option('--particle-entropy-coefficient',
              default=0.0, type=float,
              help='The entropy bonus when solving the optimization problem.')
@click.option('--forward-model-activations',
              default=['relu', 'relu'], multiple=True, type=str,
              help='The series of activation functions for every layer '
                   'in the forward model.')
@click.option('--forward-model-hidden-size',
              default=2048, type=int,
              help='The hidden size of the forward model.')
@click.option('--forward-model-final-tanh/--no-forward-model-final-tanh',
              default=False, type=bool,
              help='Whether to use a final tanh activation as the final '
                   'layer of the forward model.')
@click.option('--forward-model-lr',
              default=0.0003, type=float,
              help='The learning rate of the forward model.')
@click.option('--forward-model-alpha',
              default=1.0, type=float,
              help='The initial lagrange multiplier of the forward model.')
@click.option('--forward-model-alpha-lr',
              default=0.01, type=float,
              help='The learning rate of the lagrange multiplier.')
@click.option('--forward-model-overestimation-limit',
              default=0.5, type=float,
              help='The target used when tuning the lagrange multiplier.')
@click.option('--forward-model-noise-std',
              default=0.0, type=float,
              help='Standard deviation of continuous noise added to '
                   'designs when training the forward model.')
@click.option('--forward-model-batch-size',
              default=32, type=int,
              help='The batch size used when training the forward model.')
@click.option('--forward-model-val-size',
              default=200, type=int,
              help='The number of samples in the forward model '
                   'validation set.')
@click.option('--forward-model-epochs',
              default=50, type=int,
              help='The number of epochs to train the forward model.')
@click.option('--evaluation-samples',
              default=128, type=int,
              help='The samples to generate when solving the model-based '
                   'optimization problem.')
@click.option('--fast/--not-fast',
              default=True, type=bool,
              help='Whether to run experiment quickly and only log once.')
def coms_cleaned(
        logging_dir,
        task,
        task_relabel,
        task_max_samples,
        task_distribution,
        normalize_ys,
        normalize_xs,
        in_latent_space,
        vae_hidden_size,
        vae_latent_size,
        vae_activation,
        vae_kernel_size,
        vae_num_blocks,
        vae_lr,
        vae_beta,
        vae_batch_size,
        vae_val_size,
        vae_epochs,
        particle_lr,
        particle_train_gradient_steps,
        particle_evaluate_gradient_steps,
        particle_entropy_coefficient,
        forward_model_activations,
        forward_model_hidden_size,
        forward_model_final_tanh,
        forward_model_lr,
        forward_model_alpha,
        forward_model_alpha_lr,
        forward_model_overestimation_limit,
        forward_model_noise_std,
        forward_model_batch_size,
        forward_model_val_size,
        forward_model_epochs,
        evaluation_samples,
        fast):
    """Solve a Model-Based Optimization problem using the method:
    Conservative Objective Models (COMs).

    """

    # store the command line params in a dictionary
    params = dict(
        logging_dir=logging_dir,
        task=task,
        task_relabel=task_relabel,
        task_max_samples=task_max_samples,
        task_distribution=task_distribution,
        normalize_ys=normalize_ys,
        normalize_xs=normalize_xs,
        in_latent_space=in_latent_space,
        vae_hidden_size=vae_hidden_size,
        vae_latent_size=vae_latent_size,
        vae_activation=vae_activation,
        vae_kernel_size=vae_kernel_size,
        vae_num_blocks=vae_num_blocks,
        vae_lr=vae_lr,
        vae_beta=vae_beta,
        vae_batch_size=vae_batch_size,
        vae_val_size=vae_val_size,
        vae_epochs=vae_epochs,
        particle_lr=particle_lr,
        particle_train_gradient_steps=
        particle_train_gradient_steps,
        particle_evaluate_gradient_steps=
        particle_evaluate_gradient_steps,
        particle_entropy_coefficient=
        particle_entropy_coefficient,
        forward_model_activations=forward_model_activations,
        forward_model_hidden_size=forward_model_hidden_size,
        forward_model_final_tanh=forward_model_final_tanh,
        forward_model_lr=forward_model_lr,
        forward_model_alpha=forward_model_alpha,
        forward_model_alpha_lr=forward_model_alpha_lr,
        forward_model_overestimation_limit=
        forward_model_overestimation_limit,
        forward_model_noise_std=forward_model_noise_std,
        forward_model_batch_size=forward_model_batch_size,
        forward_model_val_size=forward_model_val_size,
        forward_model_epochs=forward_model_epochs,
        evaluation_samples=evaluation_samples,
        fast=fast)

    # create the logger and export the experiment parameters
    logger = Logger(logging_dir)
    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    # create a model-based optimization task
    task = StaticGraphTask(task, relabel=task_relabel,
                           dataset_kwargs=dict(
                               max_samples=task_max_samples,
                               distribution=task_distribution))

    if normalize_ys:
        task.map_normalize_y()
    if task.is_discrete and not in_latent_space:
        task.map_to_logits()
    if normalize_xs:
        task.map_normalize_x()

    x = task.x
    y = task.y

    if task.is_discrete and in_latent_space:

        vae_model = SequentialVAE(
            task, hidden_size=vae_hidden_size,
            latent_size=vae_latent_size, activation=vae_activation,
            kernel_size=vae_kernel_size, num_blocks=vae_num_blocks)

        vae_trainer = VAETrainer(
            vae_model, optim=tf.keras.optimizers.Adam,
            lr=vae_lr, beta=vae_beta)

        # create the training task and logger
        train_data, val_data = build_pipeline(
            x=x, y=y, batch_size=vae_batch_size,
            val_size=vae_val_size)

        # estimate the number of training steps per epoch
        vae_trainer.launch(train_data, val_data,
                           logger, vae_epochs)

        # map the x values to latent space
        x = vae_model.encoder_cnn.predict(x)[0]

        mean = np.mean(x, axis=0, keepdims=True)
        standard_dev = np.std(x - mean, axis=0, keepdims=True)
        x = (x - mean) / standard_dev

    input_shape = x.shape[1:]

    # compute the normalized learning rate of the model
    particle_lr = particle_lr * np.sqrt(np.prod(input_shape))

    # make a neural network to predict scores
    forward_model = ForwardModel(
        input_shape, activations=forward_model_activations,
        hidden_size=forward_model_hidden_size,
        final_tanh=forward_model_final_tanh)

    # make a trainer for the forward model
    trainer = ConservativeObjectiveModel(
        forward_model, forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=forward_model_lr, alpha=forward_model_alpha,
        alpha_opt=tf.keras.optimizers.Adam, alpha_lr=forward_model_alpha_lr,
        overestimation_limit=forward_model_overestimation_limit,
        particle_lr=particle_lr, noise_std=forward_model_noise_std,
        particle_gradient_steps=particle_train_gradient_steps,
        entropy_coefficient=particle_entropy_coefficient)

    # create a data set
    train_data, validate_data = build_pipeline(
        x=x, y=y, batch_size=forward_model_batch_size,
        val_size=forward_model_val_size)

    # train the forward model
    trainer.launch(train_data, validate_data,
                   logger, forward_model_epochs)

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=evaluation_samples)[1]
    initial_x = tf.gather(x, indices, axis=0)
    initial_y = tf.gather(y, indices, axis=0)
    xt = initial_x

    if not fast:

        scores = []
        predictions = []

        solution = xt
        if task.is_discrete and in_latent_space:
            solution = solution * standard_dev + mean
            logits = vae_model.decoder_cnn.predict(solution)
            solution = tf.argmax(logits, axis=2, output_type=tf.int32)

        score = task.predict(solution)

        if normalize_ys:
            initial_y = task.denormalize_y(initial_y)
            score = task.denormalize_y(score)

        logger.record(f"dataset_score", initial_y, 0, percentile=True)
        logger.record(f"score", score, 0, percentile=True)

    for step in range(1, 1 + particle_evaluate_gradient_steps):

        # update the set of solution particles
        xt = trainer.optimize(xt, 1, training=False)
        final_xt = trainer.optimize(
            xt, particle_train_gradient_steps, training=False)

        if not fast or step == particle_evaluate_gradient_steps:

            solution = xt
            if task.is_discrete and in_latent_space:
                solution = solution * standard_dev + mean
                logits = vae_model.decoder_cnn.predict(solution)
                solution = tf.argmax(logits, axis=2, output_type=tf.int32)

            np.save(os.path.join(logging_dir, "solution.npy"), solution)

            # evaluate the solutions found by the model
            score = task.predict(solution)
            prediction = forward_model(xt, training=False).numpy()
            final_prediction = forward_model(final_xt, training=False).numpy()

            if normalize_ys:
                score = task.denormalize_y(score)
                prediction = task.denormalize_y(prediction)
                final_prediction = task.denormalize_y(final_prediction)

            # record the prediction and score to the logger
            logger.record(f"score", score, step, percentile=True)
            logger.record(f"solver/model_to_real",
                          spearman(prediction[:, 0], score[:, 0]), step)
            logger.record(f"solver/distance",
                          tf.linalg.norm(xt - initial_x), step)
            logger.record(f"solver/prediction",
                          prediction, step)
            logger.record(f"solver/model_overestimation",
                          final_prediction - prediction, step)
            logger.record(f"solver/overestimation",
                          prediction - score, step)

        if not fast:

            scores.append(score)
            predictions.append(prediction)

            # save the model predictions and scores to be aggregated later
            np.save(os.path.join(logging_dir, "scores.npy"),
                    np.concatenate(scores, axis=1))
            np.save(os.path.join(logging_dir, "predictions.npy"),
                    np.stack(predictions, axis=1))


# run COMs using the command line interface
if __name__ == '__main__':
    coms_cleaned()
