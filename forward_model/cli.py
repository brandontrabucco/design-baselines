from ray import tune
import click
import ray


@click.group()
def cli():
    """A group of potential sub methods that are available for use through
    a command line interface
    """


@cli.command()
@click.option('--local-dir', type=str, default='data')
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

    from forward_model.algorithms import conservative_mbo

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(conservative_mbo, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {"val_size": 200, "batch_size": 128},
        "seed": tune.randint(10000),
        "epochs": tune.grid_search([50]),
        "hidden_size": tune.grid_search([2048]),
        'target_conservative_gap': tune.grid_search([0.0]),
        'initial_alpha': tune.grid_search([
            0.0, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]),
        'forward_model_lr': tune.grid_search([0.001]),
        'alpha_lr': tune.grid_search([0.0]),
        "perturbation_lr": tune.grid_search([1.0]),
        "perturbation_steps": tune.grid_search([100]),
        "solver_samples": tune.grid_search([128]),
        "solver_lr": tune.grid_search([1.0]),
        "solver_steps": tune.grid_search([100])},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='data')
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

    from forward_model.algorithms import conservative_mbo

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(conservative_mbo, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {"val_size": 200, "batch_size": 128},
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
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def cbas(local_dir, cpus, gpus, num_parallel, num_samples):
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

    from forward_model.algorithms import cbas

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(cbas, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {"val_size": 200, "batch_size": 32},
        "bootstraps": 8,
        "oracle_epochs": 200,
        "oracle_hidden_size": 2048,
        "oracle_lr": 0.001,
        "latent_size": 256,
        "vae_hidden_size": 2048,
        "vae_lr": 0.001,
        "vae_beta": tune.grid_search([10.0, 1.0, 0.1, 0.01]),
        "offline_vae_epochs": 200,
        "online_vae_epochs": tune.grid_search([1, 5, 10, 50]),
        "online_size": 328,  # 128 training set and 200 validation set
        "iterations": 50,
        "percentile": tune.grid_search([50, 80, 90, 100]),
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='data')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def model_inversion(local_dir, cpus, gpus, num_parallel, num_samples):
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

    from forward_model.algorithms import model_inversion

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(model_inversion, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {"val_size": 200, "batch_size": 128},
        "epochs": 100,
        "batch_size": 32,
        "hidden_size": 2048,
        "latent_size": 32,
        "model_lr": 0.0002,
        "beta_1": 0.5,
        "beta_2": 0.999,
        "solver_samples": 32},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel - 0.01})


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
    sns.set(style='darkgrid')

    import os
    file = tf.io.gfile.glob(os.path.join(dir, '*/data/events*'))
    ids = [int(f.split('conservative_mbo_')[
        1].split('_')[0]) for f in file]

    zipped_lists = zip(ids, file)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    ids, file = [list(tuple) for tuple in tuples]

    name = list(name) * (len(file) // len(name))
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
