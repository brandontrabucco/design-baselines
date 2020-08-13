from ray import tune
import click
import ray
import os


@click.group()
def cli():
    """A group of potential sub methods that are available for use through
    a command line interface
    """


#############


@cli.command()
@click.option('--local-dir', type=str, default='forward-ensemble')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def forward_ensemble_policy(local_dir, cpus, gpus, num_parallel, num_samples):
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

    from forward_model.forward_ensemble import forward_ensemble
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(forward_ensemble, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "val_size": 200,
        "batch_size": 128,
        "bootstraps": 1,
        "epochs": 200,
        "hidden_size": 2048,
        "initial_max_std": 1.5,
        "initial_min_std": 0.5,
        "forward_model_lr": 0.001,
        "solver_samples": 128,
        "solver_lr": 0.0005,
        "solver_steps": 1000},
         num_samples=num_samples,
         local_dir=local_dir,
         resources_per_trial={'cpu': cpus // num_parallel,
                              'gpu': gpus / num_parallel - 0.01})


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
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
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
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


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
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
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
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


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
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(model_inversion, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "val_size": 200,
        "is_discrete": False,
        "fully_offline": False,
        "gan_batch_size": 32,
        "hidden_size": 2048,
        "latent_size": 32,
        "generator_lr": 1e-4,
        "generator_beta_1": 0.5,
        "generator_beta_2": 0.999,
        "discriminator_lr": 1e-7,
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
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def mins_gfp(local_dir, cpus, gpus, num_parallel, num_samples):
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
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(model_inversion, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {},
        "val_size": 200,
        "is_discrete": True,
        "fully_offline": False,
        "gan_batch_size": 32,
        "hidden_size": 256,
        "temperature": 0.75,
        "generator_lr": 1e-6,
        "generator_beta_1": 0.5,
        "generator_beta_2": 0.999,
        "discriminator_lr": 1e-3,
        "discriminator_beta_1": 0.5,
        "discriminator_beta_2": 0.999,
        "initial_epochs": 200,
        "epochs_per_iteration": 10,
        "iterations": 1000,
        "exploration_samples": 32,
        "exploration_rate": 50.0,
        "thompson_samples": 32,
        "solver_samples": 32},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


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
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(model_inversion, config={
        "logging_dir": "data",
        "task": "Quadratic-v0",
        "task_kwargs": {'dataset_size': 5000},
        "val_size": 200,
        "is_discrete": False,
        "fully_offline": False,
        "gan_batch_size": 32,
        "hidden_size": 256,
        "latent_size": 16,
        "generator_lr": 1e-4,
        "generator_beta_1": 0.5,
        "generator_beta_2": 0.999,
        "discriminator_lr": 1e-3,
        "discriminator_beta_1": 0.5,
        "discriminator_beta_2": 0.999,
        "initial_epochs": 20,
        "epochs_per_iteration": 10,
        "iterations": 100,
        "exploration_samples": 32,
        "exploration_rate": 10.0,
        "exploration_noise_std": 0.1,
        "thompson_samples": 32,
        "solver_samples": 32},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


#############


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--xlabel', type=str)
@click.option('--ylabel', type=str)
def plot(dir, tag, xlabel, ylabel):

    from collections import defaultdict
    import glob
    import os
    import re
    import pickle as pkl
    import pandas as pd
    import tensorflow as tf
    import seaborn as sns
    import matplotlib.pyplot as plt

    def pretty(s):
        return s.replace('_', ' ').title()

    # get the experiment ids
    pattern = re.compile(r'.*/(\w+)_(\d+)_(\w+=[\w.+-]+[,_])*(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\w{10})$')
    dirs = [d for d in glob.glob(os.path.join(dir, '*')) if pattern.search(d) is not None]
    matches = [pattern.search(d) for d in dirs]
    ids = [int(m.group(2)) for m in matches]

    # sort the files by the experiment ids
    zipped_lists = zip(ids, dirs)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    ids, dirs = [list(tuple) for tuple in tuples]

    # get the hyper parameters for each experiment
    params = []
    for d in dirs:
        with open(os.path.join(d, 'params.pkl'), 'rb') as f:
            params.append(pkl.load(f))

    # concatenate all params along axis 1
    all_params = defaultdict(list)
    for p in params:
        for key, val in p.items():
            if val not in all_params[key]:
                all_params[key].append(val)

    # locate the params of variation in this experiment
    params_of_variation = []
    for key, val in all_params.items():
        if len(val) > 1:
            params_of_variation.append(key)

    # get the task and algorithm name
    task_name = params[0]['task']
    algo_name = matches[0].group(1)
    if len(params_of_variation) == 0:
        params_of_variation.append('task')

    # read data from tensor board
    data = pd.DataFrame(columns=[xlabel, ylabel] + params_of_variation)
    for d, p in zip(dirs, params):
        for f in glob.glob(os.path.join(d, 'data/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag:
                        row = {ylabel: tf.make_ndarray(v.tensor).tolist(),
                               xlabel: e.step}
                        for key in params_of_variation:
                            row[key] = f'{pretty(key)} = {p[key]}'
                        data = data.append(row, ignore_index=True)

    # save a separate plot for every hyper parameter
    for key in params_of_variation:
        plt.clf()
        g = sns.relplot(x=xlabel, y=ylabel, hue=key, data=data,
                        kind="line", height=5, aspect=2,
                        facet_kws={"legend_out": True})
        g.set(title=f'Evaluating {pretty(algo_name)} On {task_name}')
        plt.savefig(f'{algo_name}_{task_name}_{key}_{tag.replace("/", "_")}.png',
                    bbox_inches='tight')
