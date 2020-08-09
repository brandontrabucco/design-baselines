from ray import tune
import click
import ray


@click.group()
def cli():
    """A group of potential sub methods that are available for use through
    a command line interface
    """


@cli.command()
@click.option('--local-dir', type=str, default='c-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def conservative_policy(local_dir, cpus, gpus, num_parallel, num_samples):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    local_dir: str
        the path where model weights and tf events wil be saved
    cpus: int
        the number of cpu cores on the host machine to use
    gpus: int
        the number of gpu nodes on the host machine to use
    num_parallel: int
        the number of processes to run at once
    num_samples: int
        the number of samples to take per configuration
    """

    from forward_model.conservative import conservative

    ray.init(num_cpus=cpus, num_gpus=gpus)
    cpu = cpus // num_parallel
    gpu = gpus / num_parallel - 0.01
    tune.run(conservative, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "hidden_size": 2048,
        'forward_model_lr': 0.001,
        'target_conservative_gap': 100.0,
        'initial_alpha': 20.0,
        'alpha_lr': 0.02,
        "perturbation_lr": 0.0005,
        "perturbation_steps": 100,
        "solver_samples": 128,
        "solver_lr": 0.0005,
        "solver_steps": 100},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpu, 'gpu': gpu})


@cli.command()
@click.option('--local-dir', type=str, default='conservative_ensemble')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def conservative_ensemble_policy(local_dir, cpus, gpus, num_parallel, num_samples):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    local_dir: str
        the path where model weights and tf events wil be saved
    cpus: int
        the number of cpu cores on the host machine to use
    gpus: int
        the number of gpu nodes on the host machine to use
    num_parallel: int
        the number of processes to run at once
    num_samples: int
        the number of samples to take per configuration
    """

    from forward_model.conservative_ensemble import conservative_ensemble

    ray.init(num_cpus=cpus, num_gpus=gpus)
    cpu = cpus // num_parallel
    gpu = gpus / num_parallel - 0.01
    tune.run(conservative_ensemble, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "val_size": 200,
        "batch_size": 128,
        "bootstraps": tune.grid_search([1, 2, 4, 8, 16, 32]),
        "epochs": 200,
        "hidden_size": 2048,
        "initial_max_std": 1.5,
        "initial_min_std": 0.5,
        "forward_model_lr": 0.001,
        "target_conservative_gap": 100.0,
        "initial_alpha": 5.0,
        "alpha_lr": 0.02,
        "perturbation_lr": 0.0005,
        "perturbation_steps": 100,
        "solver_samples": 128,
        "solver_lr": 0.0005,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpu, 'gpu': gpu})


@cli.command()
@click.option('--local-dir', type=str, default='second_model_predictions')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def second_model_predictions_policy(local_dir, cpus, gpus, num_parallel, num_samples):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    local_dir: str
        the path where model weights and tf events wil be saved
    cpus: int
        the number of cpu cores on the host machine to use
    gpus: int
        the number of gpu nodes on the host machine to use
    num_parallel: int
        the number of processes to run at once
    num_samples: int
        the number of samples to take per configuration
    """

    from forward_model.conservative_ensemble import second_model_predictions

    ray.init(num_cpus=cpus, num_gpus=gpus)
    cpu = cpus // num_parallel
    gpu = gpus / num_parallel - 0.01
    tune.run(second_model_predictions, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "val_size": 200,
        "batch_size": 128,
        "epochs": 200,
        "hidden_size": 2048,
        "initial_max_std": 1.5,
        "initial_min_std": 0.5,
        "forward_model_lr": 0.001,
        "target_conservative_gap": 100.0,
        "initial_alpha": 5.0,
        "alpha_lr": 0.02,
        "perturbation_lr": 0.0005,
        "perturbation_steps": 100,
        "solver_samples": 128,
        "solver_lr": 0.0005,
        "solver_steps": 100},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpu, 'gpu': gpu})


#############


@cli.command()
@click.option('--local-dir', type=str, default='conservative_gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def conservative_gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    local_dir: str
        the path where model weights and tf events wil be saved
    cpus: int
        the number of cpu cores on the host machine to use
    gpus: int
        the number of gpu nodes on the host machine to use
    num_parallel: int
        the number of processes to run at once
    num_samples: int
        the number of samples to take per configuration
    """

    from forward_model.conservative import conservative

    ray.init(num_cpus=cpus, num_gpus=gpus)
    cpu = cpus // num_parallel
    gpu = gpus / num_parallel - 0.01
    tune.run(conservative, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {},
        "val_size": 200,
        "batch_size": 128,
        "epochs": 50,
        "hidden_size": 2048,
        'forward_model_lr': 0.001,
        'target_conservative_gap': 0.0,
        'initial_alpha': 0.00005,
        'alpha_lr': 0.0,
        "perturbation_lr": 1.0,
        "perturbation_steps": 100,
        "solver_samples": 128,
        "solver_lr": 1.0,
        "solver_steps": 100},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpu, 'gpu': gpu})


@cli.command()
@click.option('--local-dir', type=str, default='conservative_ensemble_gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def conservative_ensemble_gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    local_dir: str
        the path where model weights and tf events wil be saved
    cpus: int
        the number of cpu cores on the host machine to use
    gpus: int
        the number of gpu nodes on the host machine to use
    num_parallel: int
        the number of processes to run at once
    num_samples: int
        the number of samples to take per configuration
    """

    from forward_model.conservative_ensemble import conservative_ensemble

    ray.init(num_cpus=cpus, num_gpus=gpus)
    cpu = cpus // num_parallel
    gpu = gpus / num_parallel - 0.01
    tune.run(conservative_ensemble, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {},
        "val_size": 200,
        "batch_size": 128,
        "bootstraps": tune.grid_search([1, 2, 4, 8, 16, 32]),
        "epochs": 50,
        "hidden_size": 2048,
        "initial_max_std": 1.5,
        "initial_min_std": 0.5,
        "forward_model_lr": 0.001,
        "target_conservative_gap": 0.0,
        "initial_alpha": 0.00005,
        "alpha_lr": 0.0,
        "perturbation_lr": 1.0,
        "perturbation_steps": 100,
        "solver_samples": 128,
        "solver_lr": 1.0,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpu, 'gpu': gpu})


@cli.command()
@click.option('--local-dir', type=str, default='second_model_predictions_gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def second_model_predictions_gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    local_dir: str
        the path where model weights and tf events wil be saved
    cpus: int
        the number of cpu cores on the host machine to use
    gpus: int
        the number of gpu nodes on the host machine to use
    num_parallel: int
        the number of processes to run at once
    num_samples: int
        the number of samples to take per configuration
    """

    from forward_model.conservative_ensemble import second_model_predictions

    ray.init(num_cpus=cpus, num_gpus=gpus)
    cpu = cpus // num_parallel
    gpu = gpus / num_parallel - 0.01
    tune.run(second_model_predictions, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {},
        "val_size": 200,
        "batch_size": 128,
        "epochs": 50,
        "hidden_size": 2048,
        "initial_max_std": 1.5,
        "initial_min_std": 0.5,
        "forward_model_lr": 0.001,
        "target_conservative_gap": 0.0,
        "initial_alpha": 0.00005,
        "alpha_lr": 0.0,
        "perturbation_lr": 1.0,
        "perturbation_steps": 100,
        "solver_samples": 128,
        "solver_lr": 1.0,
        "solver_steps": 100},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpu, 'gpu': gpu})


#############


@cli.command()
@click.option('--local-dir', type=str, default='cbas')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def cbas_gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    local_dir: str
        the path where model weights and tf events wil be saved
    cpus: int
        the number of cpu cores on the host machine to use
    gpus: int
        the number of gpu nodes on the host machine to use
    num_parallel: int
        the number of processes to run at once
    num_samples: int
        the number of samples to take per configuration
    """

    from forward_model.cbas import condition_by_adaptive_sampling

    ray.init(num_cpus=cpus, num_gpus=gpus)
    cpu = cpus // num_parallel
    gpu = gpus / num_parallel - 0.01
    tune.run(condition_by_adaptive_sampling, config={
        "logging_dir": "gfp",
        "is_discrete": True,
        "task": "GFP-v0",
        "task_kwargs": {},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 128,
        "vae_batch_size": 10,
        "hidden_size": 50,
        "initial_max_std": 1.5,
        "initial_min_std": 0.5,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "latent_size": 20,
        "vae_lr": 0.001,
        "vae_beta": 1.0,
        "offline_epochs": 100,
        "online_batches": 70,
        "online_epochs": 10,
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpu, 'gpu': gpu})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def cbas_policy(local_dir, cpus, gpus, num_parallel, num_samples):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    local_dir: str
        the path where model weights and tf events wil be saved
    cpus: int
        the number of cpu cores on the host machine to use
    gpus: int
        the number of gpu nodes on the host machine to use
    num_parallel: int
        the number of processes to run at once
    num_samples: int
        the number of samples to take per configuration
    """

    from forward_model.cbas import condition_by_adaptive_sampling

    ray.init(num_cpus=cpus, num_gpus=gpus)
    cpu = cpus // num_parallel
    gpu = gpus / num_parallel - 0.01
    tune.run(condition_by_adaptive_sampling, config={
        "logging_dir": "hopper",
        "is_discrete": False,
        "task": "HopperController-v0",
        "task_kwargs": {},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 50,
        "hidden_size": 2048,
        "initial_max_std": 1.5,
        "initial_min_std": 0.5,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "latent_size": 256,
        "vae_lr": 0.001,
        "vae_beta": 1000.0,
        "offline_epochs": 100,
        "online_batches": 24,
        "online_epochs": 10,
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpu, 'gpu': gpu})


#############


@cli.command()
@click.option('--local-dir', type=str, default='mins-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def mins_policy(local_dir, cpus, gpus, num_parallel, num_samples):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    local_dir: str
        the path where model weights and tf events wil be saved
    cpus: int
        the number of cpu cores on the host machine to use
    gpus: int
        the number of gpu nodes on the host machine to use
    num_parallel: int
        the number of processes to run at once
    num_samples: int
        the number of samples to take per configuration
    """

    from forward_model.mins import model_inversion

    ray.init(num_cpus=cpus, num_gpus=gpus)
    cpu = cpus // num_parallel
    gpu = gpus / num_parallel - 0.01
    tune.run(model_inversion, config={
        "logging_dir": "hopper",
        "is_discrete": False,
        "task": "HopperController-v0",
        "task_kwargs": {},
        "bootstraps": 1,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "gan_batch_size": 100,
        "hidden_size": 2048,
        "initial_max_std": 1.5,
        "initial_min_std": 0.5,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 0,
        "latent_size": 256,
        "generator_lr": tune.grid_search([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]),
        "generator_beta_1": 0.5,
        "generator_beta_2": 0.999,
        "discriminator_lr": tune.grid_search([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]),
        "discriminator_beta_1": 0.5,
        "discriminator_beta_2": 0.999,
        "epochs_per_eval": 10,
        "iterations": 100,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpu, 'gpu': gpu})


@cli.command()
@click.option('--local-dir', type=str, default='mins-quadratic')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def mins_quadratic(local_dir, cpus, gpus, num_parallel, num_samples):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    local_dir: str
        the path where model weights and tf events wil be saved
    cpus: int
        the number of cpu cores on the host machine to use
    gpus: int
        the number of gpu nodes on the host machine to use
    num_parallel: int
        the number of processes to run at once
    num_samples: int
        the number of samples to take per configuration
    """

    from forward_model.mins import model_inversion

    ray.init(num_cpus=cpus, num_gpus=gpus)
    cpu = cpus // num_parallel
    gpu = gpus / num_parallel - 0.01
    tune.run(model_inversion, config={
        "logging_dir": "data",
        "is_discrete": False,
        "task": "Quadratic-v0",
        "task_kwargs": {'dataset_size': 5000},
        "bootstraps": 1,
        "val_size": 200,
        "ensemble_batch_size": 32,
        "gan_batch_size": 32,
        "hidden_size": 256,
        "initial_max_std": 1.5,
        "initial_min_std": 0.5,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 0,
        "latent_size": 16,
        "generator_lr": 1e-4,
        "generator_beta_1": 0.5,
        "generator_beta_2": 0.999,
        "discriminator_lr": 1e-3,
        "discriminator_beta_1": 0.5,
        "discriminator_beta_2": 0.999,
        "epochs_per_iteration": 10,
        "iterations": 100,
        "exploration_samples": 32,
        "exploration_rate": 10.0,
        "exploration_noise_std": 0.1,
        "thompson_samples": 32,
        "solver_samples": 32},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpu, 'gpu': gpu})


#############


@cli.command()
@click.option('--dir', type=str)
@click.option('--name', type=str, multiple=True)
@click.option('--tag', type=str)
@click.option('--xlabel', type=str)
@click.option('--ylabel', type=str)
@click.option('--title', type=str)
@click.option('--out', type=str)
def plot(dir, name, tag, xlabel, ylabel, title, out):

    import tensorflow as tf
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    sns.set(style='darkgrid')

    import os
    file = tf.io.gfile.glob(os.path.join(dir, '*/data/events*'))
    ids = [int(f.split('conservative_ensemble_')[
        1].split('_')[0]) for f in file]

    zipped_lists = zip(ids, file)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    ids, file = [list(tuple) for tuple in tuples]

    name = list(name) * int(math.ceil(len(file) / len(name)))
    df = pd.DataFrame(columns=[xlabel, ylabel, 'Type'])

    for f, n in zip(file, name):
        for e in tf.compat.v1.train.summary_iterator(f):
            for v in e.summary.value:
                if v.tag == tag:
                    df = df.append({xlabel: e.step,
                                    ylabel: tf.make_ndarray(v.tensor).tolist(),
                                    'Type': n}, ignore_index=True)

    plt.clf()
    g = sns.relplot(x=xlabel,
                    y=ylabel,
                    hue='Type',
                    data=df,
                    kind="line",
                    height=5,
                    aspect=2,
                    facet_kws={"legend_out": True})
    g.set(title=title)
    plt.savefig(out)
