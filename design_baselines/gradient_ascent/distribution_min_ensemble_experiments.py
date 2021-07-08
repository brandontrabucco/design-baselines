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
@click.option('--local-dir', type=str, default='gradient-ascent-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Naive Gradient Ascent on HopperController-Exact-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "HopperController-Exact-v0",
        "task_kwargs": {"relabel": False, "dataset_kwargs": {
            "max_samples": 1000,
            "distribution": tune.grid_search([
                "uniform",
                "linear",
                "quadratic",
                "exponential",
                "circular"
            ]),
            "max_percentile": 100,
            "min_percentile": 0
        }},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 5,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'min',
        "solver_samples": 512,
        "solver_lr": 0.01,
        "solver_steps": 200},
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
@click.option('--oracle', type=str, default="RandomForest")
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples, oracle):
    """Evaluate Naive Gradient Ascent on Superconductor-RandomForest-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": f"Superconductor-{oracle}-v0",
        "task_kwargs": {"relabel": False, "dataset_kwargs": {
            "max_samples": 5000,
            "distribution": tune.grid_search([
                "uniform",
                "linear",
                "quadratic",
                "exponential",
                "circular"
            ]),
            "max_percentile": 100,
            "min_percentile": 0
        }},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 5,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'min',
        "solver_samples": 512,
        "solver_lr": 0.01,
        "solver_steps": 200},
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
@click.option('--oracle', type=str, default="Transformer")
def gfp(local_dir, cpus, gpus, num_parallel, num_samples, oracle):
    """Evaluate Naive Gradient Ascent on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": f"GFP-{oracle}-v0",
        "task_kwargs": {"relabel": False, "dataset_kwargs": {
            "max_samples": 5000,
            "distribution": tune.grid_search([
                "uniform",
                "linear",
                "quadratic",
                "exponential",
                "circular"
            ]),
            "max_percentile": 100,
            "min_percentile": 0
        }},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 50,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 5,
        "vae_lr": 0.0003,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 5,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'min',
        "solver_samples": 512,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-utr')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def utr(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Naive Gradient Ascent on UTR-ResNet-v0
    """

    # Final Version

    from design_baselines.gradient_ascent import gradient_ascent
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent, config={
        "logging_dir": "data",
        "task": "UTR-ResNet-v0",
        "task_kwargs": {"relabel": False, "dataset_kwargs": {
            "max_samples": 20000,
            "distribution": tune.grid_search([
                "uniform",
                "linear",
                "quadratic",
                "exponential",
                "circular"
            ]),
            "max_percentile": 100,
            "min_percentile": 0
        }},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 50,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 5,
        "vae_lr": 0.0003,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']] * 5,
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'min',
        "solver_samples": 512,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})
