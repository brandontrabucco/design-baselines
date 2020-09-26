from ray import tune
import click
import ray
import os


@click.group()
def cli():
    """A group of experiments for training Conservative Score Models
    and reproducing our ICLR 2021 results.
    """


#############


@cli.command()
@click.option('--local-dir', type=str, default='csm-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.csm import csm
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(csm, config={
        "logging_dir": "data",
        "task": "DKittyMorphology-v0",
        "task_kwargs": {"split_percentile": 40, 'num_parallel': 2},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "target_conservative_gap": 0.0,
        "initial_alpha": 0.01,
        "alpha_lr": 0.0,
        "perturbation_lr": 0.01,
        "perturbation_steps": 50,
        "perturbation_backprop": True,
        "solver_samples": 128,
        "solver_lr": tune.sample_from(lambda c: c['config']['perturbation_lr']),
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='csm-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.csm import csm
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(csm, config={
        "logging_dir": "data",
        "task": "AntMorphology-v0",
        "task_kwargs": {"split_percentile": 20, 'num_parallel': 2},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "target_conservative_gap": 0.0,
        "initial_alpha": 0.01,
        "alpha_lr": 0.0,
        "perturbation_lr": 0.01,
        "perturbation_steps": 50,
        "perturbation_backprop": True,
        "solver_samples": 128,
        "solver_lr": tune.sample_from(lambda c: c['config']['perturbation_lr']),
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='csm-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.csm import csm
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(csm, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "target_conservative_gap": 0.0,
        "initial_alpha": 0.01,
        "alpha_lr": 0.0,
        "perturbation_lr": 0.01,
        "perturbation_steps": 50,
        "perturbation_backprop": True,
        "solver_samples": 128,
        "solver_lr": tune.sample_from(lambda c: c['config']['perturbation_lr']),
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='csm-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.csm import csm
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(csm, config={
        "logging_dir": "data",
        "task": "Superconductor-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "noise_std": 0.2,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "target_conservative_gap": 0.0,
        "initial_alpha": 0.01,
        "alpha_lr": 0.0,
        "perturbation_lr": 0.01,
        "perturbation_steps": 50,
        "perturbation_backprop": True,
        "solver_samples": 128,
        "solver_lr": tune.sample_from(lambda c: c['config']['perturbation_lr']),
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='csm-ablate-steps')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ablate_steps(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.csm import csm
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(csm, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "target_conservative_gap": 0.0,
        "initial_alpha": 5.0,
        "alpha_lr": 0.0,
        "perturbation_lr": 0.01,
        "perturbation_steps": tune.grid_search([
            0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]),
        "perturbation_backprop": True,
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})

