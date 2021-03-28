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
@click.option('--local-dir', type=str, default='coms-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on DKittyMorphology-v0
    """

    # Final Version

    from design_baselines.coms import coms
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(coms, config={
        "logging_dir": "data",
        "task": "DKittyMorphology-v0",
        "task_kwargs": {"split_percentile": 40, 'num_parallel': 2},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -0.5,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.coms import coms
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(coms, config={
        "logging_dir": "data",
        "task": "AntMorphology-v0",
        "task_kwargs": {"split_percentile": 20, 'num_parallel': 2},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.0,
        "eval_beta": tune.grid_search([0.0, 0.4]),
        "initial_alpha": tune.sample_from(
            lambda c: 0.0 if c['config']["eval_beta"] == 0.0 else 1.0),
        "alpha_lr": tune.sample_from(
            lambda c: 0.0 if c['config']["eval_beta"] == 0.0 else 0.01),
        "target_conservatism": tune.sample_from(
            lambda c: 0.0 if c['config']["eval_beta"] == 0.0 else -2.0),
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.coms import coms
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(coms, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.0,
        "eval_beta": tune.grid_search([0.0, 0.4]),
        "initial_alpha": tune.sample_from(
            lambda c: 0.0 if c['config']["eval_beta"] == 0.0 else 1.0),
        "alpha_lr": tune.sample_from(
            lambda c: 0.0 if c['config']["eval_beta"] == 0.0 else 0.01),
        "target_conservatism": tune.sample_from(
            lambda c: 0.0 if c['config']["eval_beta"] == 0.0 else -2.0),
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.coms import coms
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(coms, config={
        "logging_dir": "data",
        "task": "Superconductor-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -0.5,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-molecule')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on MoleculeActivity-v0
    """

    # Final Version

    from design_baselines.coms import coms
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(coms, config={
        "logging_dir": "data",
        "task": "MoleculeActivity-v0",
        "task_kwargs": {'split_percentile': 80},
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "discrete_clip": 0.6,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -0.5,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on GFP-v0
    """

    # Final Version

    from design_baselines.coms import coms
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(coms, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {'seed': tune.randint(1000)},
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "discrete_clip": 0.6,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -0.5,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-tfbind8')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def tfbind8(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on TfBind8-v0
    """

    # Final Version

    from design_baselines.coms import coms
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(coms, config={
        "logging_dir": "data",
        "task": "TfBind8-v0",
        "task_kwargs": {'split_percentile': 20},
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "discrete_clip": 0.6,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -0.5,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-utr')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def utr(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on UTRExpression-v0
    """

    # Final Version

    from design_baselines.coms import coms
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser(f'~/tmp_{randint(0, 1000000)}'))
    tune.run(coms, config={
        "logging_dir": "data",
        "task": "UTRExpression-v0",
        "task_kwargs": {'split_percentile': 20},
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "discrete_clip": 0.6,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 0.001,
        "alpha_lr": 0.01,
        "target_conservatism": -1000.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})
