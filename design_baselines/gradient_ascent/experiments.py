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
@click.option('--local-dir', type=str, default='gradient-ascent-molecule')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "MoleculeActivity-v0",
        "task_kwargs": {'split_percentile': 80, },
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {'seed': tune.randint(1000),
                        'split_percentile': 100, },
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-gfp-v1')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_v1(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on GFP-v1
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "GFP-v1",
        "task_kwargs": {'split_percentile': 20, },
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on DKittyMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "DKittyMorphology-v0",
        "task_kwargs": {"split_percentile": 40, 'num_parallel': 1, },
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "AntMorphology-v0",
        "task_kwargs": {"split_percentile": 20, 'num_parallel': 1, },
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {'split_percentile': 100, },
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 20},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "Superconductor-v0",
        "task_kwargs": {'split_percentile': 80, },
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


#############


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def normalized_hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": tune.grid_search([0.1,
                                       0.05,
                                       0.02,
                                       0.01,
                                       0.005,
                                       0.002,
                                       0.001,
                                       0.0005]),
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='unnormalized-hopper-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def unnormalized_hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": False,
        "normalize_xs": False,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": tune.grid_search([0.00005,
                                       0.00002,
                                       0.00001,
                                       0.000005,
                                       0.000002,
                                       0.000001,
                                       0.0000005,
                                       0.0000002]),
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


#############


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-molecule')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def molecule_mean_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "MoleculeActivity-v0",
        "task_kwargs": {'split_percentile': 80, },
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_mean_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on GFP-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {'seed': tune.randint(1000), },
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-gfp-v1')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_v1_mean_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on GFP-v1
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "GFP-v1",
        "task_kwargs": {'split_percentile': 20, },
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def dkitty_mean_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on DKittyMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "DKittyMorphology-v0",
        "task_kwargs": {"split_percentile": 40, 'num_parallel': 1, },
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant_mean_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "AntMorphology-v0",
        "task_kwargs": {"split_percentile": 20, 'num_parallel': 1, },
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper_mean_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def superconductor_mean_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "Superconductor-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'mean',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


#############


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-molecule')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def molecule_min_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "MoleculeActivity-v0",
        "task_kwargs": {'split_percentile': 80, },
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_min_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on GFP-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {'seed': tune.randint(1000), },
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-gfp-v1')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_v1_min_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on GFP-v1
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "GFP-v1",
        "task_kwargs": {'split_percentile': 20, },
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def dkitty_min_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on DKittyMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "DKittyMorphology-v0",
        "task_kwargs": {"split_percentile": 40, 'num_parallel': 1, },
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant_min_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "AntMorphology-v0",
        "task_kwargs": {"split_percentile": 20, 'num_parallel': 1, },
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper_min_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def superconductor_min_ensemble(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "Superconductor-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 8,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


#############


@cli.command()
@click.option('--local-dir', type=str, default='ablate-architecture-molecule')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def molecule_ablate_architecture(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import ablate_architecture
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(ablate_architecture, config={
        "logging_dir": tune.grid_search([
            "/home/btrabucco/final-results/online/gradient-ascent-molecule/gradient_ascent/gradient_ascent_0_2020-09-30_21-42-049y77namm/data",
            "/home/btrabucco/final-results/online/gradient-ascent-molecule/gradient_ascent/gradient_ascent_0_2020-09-30_21-42-04cwdcm319/data",
            "/home/btrabucco/final-results/online/gradient-ascent-molecule/gradient_ascent/gradient_ascent_0_2020-09-30_21-55-50djj2garj/data",
            "/home/btrabucco/final-results/online/gradient-ascent-molecule/gradient_ascent/gradient_ascent_0_2020-09-30_22-28-52wmok817v/data",
            "/home/btrabucco/final-results/online/gradient-ascent-molecule/gradient_ascent/gradient_ascent_0_2020-09-30_22-30-522yva_tus/data",
            "/home/btrabucco/final-results/online/gradient-ascent-molecule/gradient_ascent/gradient_ascent_0_2020-10-01_00-29-45ltjrwrq1/data",
            "/home/btrabucco/final-results/online/gradient-ascent-molecule/gradient_ascent/gradient_ascent_0_2020-10-01_00-32-52xj512n47/data",
            "/home/btrabucco/final-results/online/gradient-ascent-molecule/gradient_ascent/gradient_ascent_0_2020-10-01_00-34-00cff0kpyl/data",
            "/home/btrabucco/final-results/online/gradient-ascent-molecule/gradient_ascent/gradient_ascent_0_2020-10-01_00-36-43psjojj9r/data",
        ]),
        "task": "MoleculeActivity-v0",
        "task_kwargs": {'split_percentile': 80},
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='ablate-architecture-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_ablate_architecture(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import ablate_architecture
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(ablate_architecture, config={
        "logging_dir": tune.grid_search([
            "/home/btrabucco/final-results/online/gradient-ascent-gfp/gradient_ascent/gradient_ascent_0_seed=251_2020-10-01_00-14-23_8hsa8w8/data",
            "/home/btrabucco/final-results/online/gradient-ascent-gfp/gradient_ascent/gradient_ascent_0_seed=254_2020-09-30_21-55-40wz6uuod2/data",
            "/home/btrabucco/final-results/online/gradient-ascent-gfp/gradient_ascent/gradient_ascent_0_seed=553_2020-09-30_21-51-01w70xp2py/data",
            "/home/btrabucco/final-results/online/gradient-ascent-gfp/gradient_ascent/gradient_ascent_0_seed=584_2020-09-30_21-55-44h24hnizr/data",
            "/home/btrabucco/final-results/online/gradient-ascent-gfp/gradient_ascent/gradient_ascent_0_seed=737_2020-10-01_00-29-46z8mvln2i/data",
            "/home/btrabucco/final-results/online/gradient-ascent-gfp/gradient_ascent/gradient_ascent_0_seed=87_2020-09-30_21-49-02plk6etwo/data",
            "/home/btrabucco/final-results/online/gradient-ascent-gfp/gradient_ascent/gradient_ascent_0_seed=986_2020-10-01_00-08-34cgy0r788/data",
        ]),
        "task": "GFP-v0",
        "task_kwargs": {'seed': tune.randint(1000)},
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "discrete_smoothing": 0.6,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='ablate-architecture-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def dkitty_ablate_architecture(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on DKittyMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import ablate_architecture
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(ablate_architecture, config={
        "logging_dir": "data",
        "task": "DKittyMorphology-v0",
        "task_kwargs": {"split_percentile": 40, 'num_parallel': 2},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='ablate-architecture-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant_ablate_architecture(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import ablate_architecture
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(ablate_architecture, config={
        "logging_dir": "data",
        "task": "AntMorphology-v0",
        "task_kwargs": {"split_percentile": 20, 'num_parallel': 2},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='ablate-architecture-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper_ablate_architecture(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import ablate_architecture
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(ablate_architecture, config={
        "logging_dir": tune.grid_search([
            "/home/btrabucco/final-results/online/gradient-ascent-hopper/gradient_ascent/gradient_ascent_0_2020-09-30_21-48-57ps4moq6h/data",
            "/home/btrabucco/final-results/online/gradient-ascent-hopper/gradient_ascent/gradient_ascent_0_2020-09-30_21-49-05whwnc2un/data",
            "/home/btrabucco/final-results/online/gradient-ascent-hopper/gradient_ascent/gradient_ascent_0_2020-09-30_21-49-05_wl4kzj2/data",
            "/home/btrabucco/final-results/online/gradient-ascent-hopper/gradient_ascent/gradient_ascent_0_2020-09-30_21-55-53ck99l1gq/data",
            "/home/btrabucco/final-results/online/gradient-ascent-hopper/gradient_ascent/gradient_ascent_0_2020-09-30_23-04-4084sewsd6/data",
            "/home/btrabucco/final-results/online/gradient-ascent-hopper/gradient_ascent/gradient_ascent_0_2020-09-30_23-19-23e4omgube/data",
            "/home/btrabucco/final-results/online/gradient-ascent-hopper/gradient_ascent/gradient_ascent_0_2020-09-30_23-20-50oifowxld/data",
        ]),
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='ablate_architecture-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def superconductor_ablate_architecture(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import ablate_architecture
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(ablate_architecture, config={
        "logging_dir": tune.grid_search([
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-09-30_21-42-19jstptgp5/data",
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-09-30_21-47-18h7a568jo/data",
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-09-30_21-47-18imu9z_te/data",
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-09-30_21-47-19x47j7ld2/data",
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-09-30_22-36-3174a153vu/data",
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-09-30_22-44-45ne4j3vy5/data",
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-09-30_22-46-51nj5pt3ix/data",
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-09-30_22-52-168ixlggtw/data",
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-10-01_00-46-23fdy5ch4k/data",
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-10-01_00-46-43jh8et8ss/data",
            "/home/btrabucco/final-results/online/gradient-ascent-superconductor/gradient_ascent/gradient_ascent_0_2020-10-01_00-51-55mqjq4j5e/data",
        ]),
        "task": "Superconductor-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.001,
        "aggregation_method": 'min',
        "solver_samples": 128,
        "solver_lr": 0.01,
        "solver_steps": 1000},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})
