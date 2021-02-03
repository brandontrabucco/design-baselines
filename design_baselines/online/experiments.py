from ray import tune
import click
import ray
import os
from random import randint


@click.group()
def cli():
    """A group of experiments for training Conservative Score Models
    and reproducing our ICLR 2021 results.
    """


#############


@cli.command()
@click.option('--local-dir', type=str, default='online-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on DKittyMorphology-v0
    """

    # Final Version

    from design_baselines.online import online
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(online, config={
        "logging_dir": "data",
        "task": "DKittyMorphology-v0",
        "task_kwargs": {"split_percentile": 40, 'num_parallel': 2},
        "is_discrete": False,
        "constraint_type": "dataset",
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 500,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.05,
        "target_conservatism": 0.05,
        "negatives_fraction": 1.0,
        "lookahead_steps": 1,
        "lookahead_backprop": True,
        "evaluate_steps": [450],
        "solver_lr": 0.01,
        "solver_interval": 1,
        "solver_warmup": 50,
        "solver_steps": 1,
        "solver_beta": tune.grid_search([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='online-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.online import online
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(online, config={
        "logging_dir": "data",
        "task": "AntMorphology-v0",
        "task_kwargs": {"split_percentile": 20, 'num_parallel': 2},
        "is_discrete": False,
        "constraint_type": "dataset",
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 500,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.05,
        "target_conservatism": 0.05,
        "negatives_fraction": 1.0,
        "lookahead_steps": 1,
        "lookahead_backprop": True,
        "evaluate_steps": [450],
        "solver_lr": 0.01,
        "solver_interval": 1,
        "solver_warmup": 50,
        "solver_steps": 1,
        "solver_beta": tune.grid_search([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='online-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.online import online
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(online, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "constraint_type": "mix",
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 500,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden_size": 512,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.05,
        "target_conservatism": 0.05,
        "negatives_fraction": 1.0,
        "lookahead_steps": 1,
        "lookahead_backprop": True,
        "evaluate_steps": [450],
        "solver_lr": 0.01,
        "solver_interval": 1,
        "solver_warmup": 50,
        "solver_steps": 1,
        "solver_beta": tune.grid_search([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='online-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.online import online
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(online, config={
        "logging_dir": "data",
        "task": "Superconductor-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "constraint_type": "mix",
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 500,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.05,
        "target_conservatism": 0.05,
        "negatives_fraction": 1.0,
        "lookahead_steps": 1,
        "lookahead_backprop": True,
        "evaluate_steps": [450],
        "solver_lr": 0.01,
        "solver_interval": 1,
        "solver_warmup": 50,
        "solver_steps": 1,
        "solver_beta": tune.grid_search([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='online-molecule')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on MoleculeActivity-v0
    """

    # Final Version

    from design_baselines.online import online
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(online, config={
        "logging_dir": "data",
        "task": "MoleculeActivity-v0",
        "task_kwargs": {'split_percentile': 80},
        "is_discrete": True,
        "constraint_type": "mix",
        "normalize_ys": True,
        "normalize_xs": False,
        "continuous_noise_std": 0.2,
        "discrete_clip": 0.6,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 500,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden_size": 512,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.05,
        "target_conservatism": 0.05,
        "negatives_fraction": 1.0,
        "lookahead_steps": 1,
        "lookahead_backprop": True,
        "evaluate_steps": [i for i in range(0, 500, 10)],
        "solver_lr": 0.01,
        "solver_interval": 1,
        "solver_warmup": 50,
        "solver_steps": 1,
        "solver_beta": tune.grid_search([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='online-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on GFP-v0
    """

    # Final Version

    from design_baselines.online import online
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(online, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {'seed': tune.randint(1000)},
        "is_discrete": True,
        "constraint_type": "mix",
        "normalize_ys": True,
        "normalize_xs": False,
        "continuous_noise_std": 0.2,
        "discrete_clip": 0.6,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 500,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden_size": 512,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.05,
        "target_conservatism": 0.05,
        "negatives_fraction": 1.0,
        "lookahead_steps": 1,
        "lookahead_backprop": True,
        "evaluate_steps": [i for i in range(0, 500, 10)],
        "solver_lr": 0.01,
        "solver_interval": 1,
        "solver_warmup": 50,
        "solver_steps": 1,
        "solver_beta": tune.grid_search([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})
