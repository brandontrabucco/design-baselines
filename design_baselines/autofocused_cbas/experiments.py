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
@click.option('--local-dir', type=str, default='autofocused-cbas-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on DKittyMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.autofocused_cbas import autofocused_cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(autofocused_cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "DKittyMorphology-Exact-v0",
        "task_kwargs": {"relabel": False},
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
        "autofocus_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='autofocused-cbas-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on AntMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.autofocused_cbas import autofocused_cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(autofocused_cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "AntMorphology-Exact-v0",
        "task_kwargs": {"relabel": False},
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
        "autofocus_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='autofocused-cbas-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on HopperController-Exact-v0
    """

    # Final Version

    from design_baselines.autofocused_cbas import autofocused_cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(autofocused_cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "HopperController-Exact-v0",
        "task_kwargs": {"relabel": False},
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
        "autofocus_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='autofocused-cbas-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--oracle', type=str, default="RandomForest")
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples, oracle):
    """Evaluate AutoFocusing on Superconductor-RandomForest-v0
    """

    # Final Version

    from design_baselines.autofocused_cbas import autofocused_cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(autofocused_cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": f"Superconductor-{oracle}-v0",
        "task_kwargs": {"relabel": False},
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
        "autofocus_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='autofocused-cbas-chembl')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--assay-chembl-id', type=str, default='CHEMBL3885882')
@click.option('--standard-type', type=str, default='MCHC')
def chembl(local_dir, cpus, gpus, num_parallel, num_samples,
           assay_chembl_id, standard_type):
    """Evaluate AutoFocusing on ChEMBL-ResNet-v0
    """

    # Final Version

    from design_baselines.autofocused_cbas import autofocused_cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(autofocused_cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": f"ChEMBL_{standard_type}_{assay_chembl_id}"
                f"_MorganFingerprint-RandomForest-v0",
        "task_kwargs": {"relabel": False,
                        "dataset_kwargs": dict(
                            assay_chembl_id=assay_chembl_id,
                            standard_type=standard_type)},
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "vae_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 2048,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.0003,
        "ensemble_epochs": 50,
        "latent_size": 32,
        "vae_lr": 0.0003,
        "vae_beta": 10.0,
        "offline_epochs": 50,
        "online_batches": 10,
        "online_epochs": 10,
        "autofocus_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='autofocused-cbas-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--oracle', type=str, default="Transformer")
def gfp(local_dir, cpus, gpus, num_parallel, num_samples, oracle):
    """Evaluate AutoFocusing on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.autofocused_cbas import autofocused_cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(autofocused_cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": f"GFP-{oracle}-v0",
        "task_kwargs": {"relabel": False},
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
        "autofocus_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='autofocused-cbas-tf-bind-8')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def tf_bind_8(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on TFBind8-Exact-v0
    """

    # Final Version

    from design_baselines.autofocused_cbas import autofocused_cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(autofocused_cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "TFBind8-Exact-v0",
        "task_kwargs": {"relabel": False},
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
        "autofocus_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='autofocused-cbas-utr')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def utr(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on UTR-ResNet-v0
    """

    # Final Version

    from design_baselines.autofocused_cbas import autofocused_cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(autofocused_cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "UTR-ResNet-v0",
        "task_kwargs": {"relabel": False},
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
        "autofocus_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='autofocused-cbas-tf_bind_10')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def tf_bind_10(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on TFBind10-Exact-v0
    """

    # Final Version

    from design_baselines.autofocused_cbas import autofocused_cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(autofocused_cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "TFBind10-Exact-v0",
        "task_kwargs": {"relabel": False, "dataset_kwargs": {"max_samples": 10000}},
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
        "offline_epochs": 100,
        "online_batches": 10,
        "online_epochs": 10,
        "autofocus_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='autofocused-cbas-nas')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def nas(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate AutoFocusing on CIFARNAS-Exact-v0
    """

    # Final Version

    from design_baselines.autofocused_cbas import autofocused_cbas
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(autofocused_cbas, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "CIFARNAS-Exact-v0",
        "task_kwargs": {"relabel": False},
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
        "autofocus_epochs": 10,
        "iterations": 20,
        "percentile": 80.0,
        "solver_samples": 128, "do_evaluation": False},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})

