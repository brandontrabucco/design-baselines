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
@click.option('--local-dir', type=str, default='cbas-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on DKittyMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "DKittyMorphology-Exact-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on AntMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "AntMorphology-Exact-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on HopperController-Exact-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "HopperController-Exact-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
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
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on Superconductor-FullyConnected-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "Superconductor-FullyConnected-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 200,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-chembl')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def chembl(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on ChEMBL-ResNet-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "ChEMBL-ResNet-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-gfp-gp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_gp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "GFP-GP-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-gfp-rf')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_rf(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "GFP-RandomForest-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-gfp-fc')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_fc(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "GFP-FullyConnected-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-gfp-rn')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_rn(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "GFP-ResNet-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-gfp-tr')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def gfp_tr(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "GFP-Transformer-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cbas-tf-bind-8')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def tf_bind_8(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on TFBind8-Exact-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "TFBind8-Exact-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
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
    """Evaluate AutoFocusing on UTR-Transformer-v0
    """

    # Final Version

    from design_baselines.cbas import cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "UTR-Transformer-v0",
        "task_kwargs": {'relabel': True, 'oracle_kwargs': {'noise_std': 0.0}},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "hidden_size": 256,
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
        "iterations": 50,
        "percentile": 80.0,
        "solver_samples": 128},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})

