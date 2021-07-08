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
@click.option('--local-dir', type=str, default='cbas-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on HopperController-Exact-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
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
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.0003,
        "ensemble_epochs": 100,
        "latent_size": 32,
        "vae_lr": 0.0003,
        "vae_beta": 1.0,
        "offline_epochs": 200,
        "online_batches": 10,
        "online_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 512},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--oracle', type=str, default="RandomForest")
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples, oracle):
    """Evaluate AutoFocusing on Superconductor-RandomForest-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
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
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.0003,
        "ensemble_epochs": 100,
        "latent_size": 32,
        "vae_lr": 0.0003,
        "vae_beta": 1.0,
        "offline_epochs": 200,
        "online_batches": 10,
        "online_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 512},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--oracle', type=str, default="Transformer")
def gfp(local_dir, cpus, gpus, num_parallel, num_samples, oracle):
    """Evaluate AutoFocusing on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
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
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.0003,
        "ensemble_epochs": 100,
        "latent_size": 32,
        "vae_lr": 0.0003,
        "vae_beta": 1.0,
        "offline_epochs": 200,
        "online_batches": 10,
        "online_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 512},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-utr')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def utr(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on UTR-ResNet-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
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
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.0003,
        "ensemble_epochs": 100,
        "latent_size": 32,
        "vae_lr": 0.0003,
        "vae_beta": 1.0,
        "offline_epochs": 200,
        "online_batches": 10,
        "online_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 512},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})

