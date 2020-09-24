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
@click.option('--local-dir', type=str, default='mins-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.mins import mins
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins, config={
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
        "perturbation_backprop": tune.grid_search([True, False]),
        "solver_samples": 128,
        "solver_lr": tune.sample_from(lambda c: c['config']['perturbation_lr']),
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.mins import mins
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins, config={
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
        "perturbation_backprop": tune.grid_search([True, False]),
        "solver_samples": 128,
        "solver_lr": tune.sample_from(lambda c: c['config']['perturbation_lr']),
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.mins import mins
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins, config={
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
        "perturbation_backprop": tune.grid_search([True, False]),
        "solver_samples": 128,
        "solver_lr": tune.sample_from(lambda c: c['config']['perturbation_lr']),
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.mins import mins
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins, config={
        "logging_dir": "data",
        "task": "Superconductor-v0",
        "task_kwargs": {'split_percentile': 80},
        "val_size": 200,
        "offline": False,
        "is_discrete": False,
        "base_temp": 0.1,
        "noise_std": 0.0,
        "method": "least_squares",
        "gan_batch_size": 128,
        "hidden_size": 1024,
        "latent_size": 32,
        "critic_frequency": 1,
        "flip_frac": .2,
        "pool_size": 0,
        "pool_frac": 0.,
        "pool_save": 0,
        "fake_pair_frac": 0.0,
        "penalty_weight": 0.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 500,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins-molecule')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate MINs on MoleculeActivity-v0
    """

    # Final Version

    from design_baselines.mins import mins
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins, config={
        "logging_dir": "data",
        "task": "MoleculeActivity-v0",
        "task_kwargs": {'split_percentile': 80},
        "val_size": 200,
        "offline": False,
        "is_discrete": True,
        "base_temp": 0.1,
        "keep": 0.99,
        "start_temp": 1.0,
        "final_temp": 1.0,
        "method": "least_squares",
        "gan_batch_size": 128,
        "hidden_size": 256,
        "latent_size": 32,
        "critic_frequency": 1,
        "flip_frac": 0.,
        "pool_size": 0,
        "pool_frac": 0.,
        "pool_save": 0,
        "fake_pair_frac": 0.0,
        "penalty_weight": 0.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.5,
        "generator_beta_2": 0.999,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.5,
        "discriminator_beta_2": 0.999,
        "initial_epochs": 500,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128},
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
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate MINs on GFP-v0
    """

    # Final Version

    from design_baselines.mins import mins
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {'seed': tune.randint(1000)},
        "val_size": 200,
        "offline": False,
        "is_discrete": True,
        "base_temp": 0.1,
        "keep": 0.99,
        "start_temp": 5.0,
        "final_temp": 5.0,
        "method": "least_squares",
        "gan_batch_size": 128,
        "hidden_size": 256,
        "latent_size": 32,
        "critic_frequency": 1,
        "flip_frac": 0.,
        "pool_size": 0,
        "pool_frac": 0.,
        "pool_save": 0,
        "fake_pair_frac": 0.0,
        "penalty_weight": 0.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.5,
        "generator_beta_2": 0.999,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.5,
        "discriminator_beta_2": 0.999,
        "initial_epochs": 200,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})
