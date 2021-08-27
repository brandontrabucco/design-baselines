from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.bo_qei.trainers import Ensemble, VAETrainer
from design_baselines.bo_qei.nets import ForwardModel, SequentialVAE
from design_baselines.utils import render_video
import tensorflow as tf
import numpy as np
import os


def bo_qei(config):
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

    # create the training task and logger
    train_data, val_data = build_pipeline(
        x=x, y=y, bootstraps=config['bootstraps'],
        batch_size=config['ensemble_batch_size'],
        val_size=config['val_size'])

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

    # train the model for an additional number of epochs
    ensemble.launch(train_data,
                    val_data,
                    logger,
                    config['ensemble_epochs'])

    # select the top 1 initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=config['bo_gp_samples'])[1]
    initial_x = tf.gather(x, indices, axis=0)
    initial_y = tf.gather(y, indices, axis=0)

    from botorch.models import FixedNoiseGP, ModelListGP
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.optim import optimize_acqf
    from botorch import fit_gpytorch_model
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.sampling.samplers import SobolQMCNormalSampler
    from botorch.exceptions import BadInitialCandidatesWarning

    import torch
    import time
    import warnings

    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    def objective(input_x):
        original_x = input_x
        # convert the tensor into numpy before using a TF model
        if torch.cuda.is_available():
            input_x = input_x.detach().cpu().numpy()
        else:
            input_x = input_x.detach().numpy()
        batch_shape = input_x.shape[:-1]
        # pass the input into a TF model
        input_x = tf.reshape(input_x, [-1, *input_shape])

        # optimize teh ground truth or the learned model
        if config["optimize_ground_truth"]:
            if task.is_discrete and config["use_vae"]:
                input_x = tf.argmax(vae_model.decoder_cnn.predict(
                    input_x * standard_dev + mean),
                    axis=2, output_type=tf.int32)
            value = task.predict(input_x)
        else:
            value = ensemble.get_distribution(input_x).mean()

        ys = value.numpy()

        ys.reshape(list(batch_shape) + [1])
        # convert the scores back to pytorch tensors
        return torch.tensor(ys).type_as(
            original_x).to(device, dtype=dtype)

    NOISE_SE = config['bo_noise_se']
    train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)

    def initialize_model(train_x, train_obj, state_dict=None):
        # define models for objective
        model_obj = FixedNoiseGP(train_x, train_obj,
                                 train_yvar.expand_as(train_obj)).to(train_x)
        # combine into a multi-output GP model
        model = ModelListGP(model_obj)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model

    def obj_callable(Z):
        return Z[..., 0]

    # define a feasibility-weighted objective for optimization
    obj = GenericMCObjective(obj_callable)

    BATCH_SIZE = config['bo_batch_size']
    bounds = torch.tensor(
        [np.min(x, axis=0).reshape([input_size]).tolist(),
         np.max(x, axis=0).reshape([input_size]).tolist()],
        device=device, dtype=dtype)

    def optimize_acqf_and_get_observation(acq_func):
        """Optimizes the acquisition function, and returns
        a new candidate and a noisy observation."""
        # optimize
        try:
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=BATCH_SIZE,
                num_restarts=config['bo_num_restarts'],
                raw_samples=config['bo_raw_samples'],  # used for intialization heuristic
                options={"batch_limit": config['bo_batch_limit'],
                         "maxiter": config['bo_maxiter']})
        except RuntimeError:
            return
        # observe new values
        new_x = candidates.detach()
        exact_obj = objective(candidates)
        new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
        return new_x, new_obj

    N_BATCH = config['bo_iterations']
    MC_SAMPLES = config['bo_mc_samples']

    best_observed_ei = []

    # call helper functions to generate initial training data and initialize model
    train_x_ei = initial_x.numpy().reshape([initial_x.shape[0], input_size])
    train_x_ei = torch.tensor(train_x_ei).to(device, dtype=dtype)

    train_obj_ei = initial_y.numpy().reshape([initial_y.shape[0], 1])
    train_obj_ei = torch.tensor(train_obj_ei).to(device, dtype=dtype)

    best_observed_value_ei = train_obj_ei.max().item()
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)
    best_observed_ei.append(best_observed_value_ei)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):

        t0 = time.time()

        # fit the models
        fit_gpytorch_model(mll_ei)

        # define the qEI acquisition module using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

        # for best_f, we use the best observed noisy values as an approximation
        qEI = qExpectedImprovement(
            model=model_ei, best_f=train_obj_ei.max(),
            sampler=qmc_sampler, objective=obj)

        # optimize and get new observation
        result = optimize_acqf_and_get_observation(qEI)
        if result is None:
            print("RuntimeError was encountered, most likely a "
                  "'symeig_cpu: the algorithm failed to converge'")
            break
        new_x_ei, new_obj_ei = result

        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

        # update progress
        best_value_ei = obj(train_x_ei).max().item()
        best_observed_ei.append(best_value_ei)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei, train_obj_ei, model_ei.state_dict())

        t1 = time.time()
        print(f"Batch {iteration:>2}: best_value = "
              f"({best_value_ei:>4.2f}), "
              f"time = {t1 - t0:>4.2f}.", end="")

    if torch.cuda.is_available():
        x_sol = train_x_ei.detach().cpu().numpy()
        y_sol = train_obj_ei.detach().cpu().numpy()

    else:
        x_sol = train_x_ei.detach().numpy()
        y_sol = train_obj_ei.detach().numpy()

    # select the top 1 initial designs from the dataset
    indices = tf.math.top_k(y_sol[:, 0], k=config['solver_samples'])[1]
    solution = tf.gather(x_sol, indices, axis=0)
    solution = tf.reshape(solution, [-1, *input_shape])

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
        logger.record("score", score, N_BATCH, percentile=True)
