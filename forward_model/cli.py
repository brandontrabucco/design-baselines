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
def conservative(local_dir, cpus, gpus, num_parallel, num_samples):
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
        "seed": tune.randint(10000),
        "epochs": tune.grid_search([100]),
        "hidden_size": tune.grid_search([256]),
        "batch_size": tune.grid_search([128]),
        "forward_model_lr": tune.grid_search([0.001]),
        "perturbation_lr": tune.grid_search([0.001]),
        "perturbation_steps": tune.grid_search([100]),
        "solver_samples": tune.grid_search([128]),
        "solver_lr": tune.grid_search([0.001]),
        "solver_steps": tune.grid_search([100]),
        "conservative_weight": tune.grid_search([
            0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])},
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
def model_inversion(local_dir, cpus, gpus, num_parallel):
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
    """

    from forward_model.algorithms import model_inversion

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(model_inversion, config={
        "logging_dir": local_dir,
        "epochs": tune.grid_search([100]),
        "batch_size": tune.grid_search([32]),
        "hidden_size": tune.grid_search([2048 * 8]),
        "latent_size": tune.grid_search([32]),
        "model_lr": tune.grid_search([0.0002]),
        "beta_1": tune.grid_search([0.5]),
        "beta_2": tune.grid_search([0.999]),
        "solver_samples": tune.grid_search([32])},
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
    ids = [int(f.split('_batch_size')[
        0].split('conservative_mbo_')[1]) for f in file]

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
