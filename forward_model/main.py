from ray import tune
import click
import ray


@click.group()
def cli():
    """A group of potential sub methods that are available for use through
    a command line interface
    """


@cli.command()
@click.option('--local-dir', type=str, default='data')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
def conservative(local_dir, cpus, gpus, num_parallel):
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
    """

    from forward_model.algorithms import conservative_mbo

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(conservative_mbo, config={
        "logging_dir": local_dir,
        "epochs": tune.grid_search([500]),
        "hidden_size": tune.grid_search([2048]),
        "batch_size": tune.grid_search([128]),
        "forward_model_lr": tune.grid_search([0.0001]),
        "conservative_weight": tune.grid_search([0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
        "perturbation_lr": tune.grid_search([0.001]),
        "perturbation_steps": tune.grid_search([100]),
        "solver_samples": tune.grid_search([128]),
        "solver_lr": tune.grid_search([0.001]),
        "solver_steps": tune.grid_search([100])},
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='data')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
def model_inversion(local_dir, cpus, gpus, num_parallel):
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
    """

    from forward_model.algorithms import model_inversion

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(model_inversion, config={
        "logging_dir": local_dir,
        "epochs": tune.grid_search([100]),
        "batch_size": tune.grid_search([32]),
        "hidden_size": tune.grid_search([2048]),
        "latent_size": tune.grid_search([32]),
        "model_lr": tune.grid_search([0.0002]),
        "beta_1": tune.grid_search([0.5]),
        "beta_2": tune.grid_search([0.999]),
        "solver_samples": tune.grid_search([32])},
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel - 0.01})
