from forward_model.run_experiment import run_experiment
from ray import tune
import click
import ray


@click.group()
def cli():
    """A group of potential sub methods that are available for use through
    a command line interface
    """


@cli.command()
@click.option('--local-dir', type=str, default='./data')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--init-lr', type=click.FLOAT, default=[0.0001], multiple=True)
@click.option('--num-epochs', type=click.INT, default=[100], multiple=True)
@click.option('--hidden-size', type=click.INT, default=[2048], multiple=True)
@click.option('--solver-lr', type=click.FLOAT, default=[0.001], multiple=True)
@click.option('--solver-samples', type=click.INT, default=[10], multiple=True)
@click.option('--solver-steps', type=click.INT, default=[100], multiple=True)
@click.option('--evaluate-interval', type=click.INT, default=[1], multiple=True)
@click.option('--sc-noise-std', type=click.FLOAT, default=[0.3], multiple=True)
@click.option('--sc-lambda', type=click.FLOAT, default=[10.0], multiple=True)
@click.option('--sc-weight', type=click.FLOAT, default=[1.0], multiple=True)
@click.option('--cs-noise-std', type=click.FLOAT, default=[0.3], multiple=True)
@click.option('--cs-weight', type=click.FLOAT, default=[1.0], multiple=True)
@click.option('--fgsm-lambda', type=click.FLOAT, default=[0.01], multiple=True)
@click.option('--fgsm-interval', type=click.INT, default=[1], multiple=True)
@click.option('--online-noise-std', type=click.FLOAT, default=[0.3], multiple=True)
@click.option('--online-steps', type=click.INT, default=[8], multiple=True)
def train(local_dir, cpus, gpus, num_parallel,
          init_lr, num_epochs, hidden_size,
          solver_lr, solver_samples, solver_steps, evaluate_interval,
          sc_noise_std, sc_lambda, sc_weight,
          cs_noise_std, cs_weight,
          fgsm_lambda, fgsm_interval,
          online_noise_std, online_steps):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem
    """

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(run_experiment, config={
        "local_dir": local_dir,
        "init_lr": tune.grid_search(list(init_lr)),
        "num_epochs": tune.grid_search(list(num_epochs)),
        "hidden_size": tune.grid_search(list(hidden_size)),
        "solver_lr": tune.grid_search(list(solver_lr)),
        "solver_samples": tune.grid_search(list(solver_samples)),
        "solver_steps": tune.grid_search(list(solver_steps)),
        "evaluate_interval": tune.grid_search(list(evaluate_interval)),
        "sc_noise_std": tune.grid_search(list(sc_noise_std)),
        "sc_lambda": tune.grid_search(list(sc_lambda)),
        "sc_weight": tune.grid_search(list(sc_weight)),
        "cs_noise_std": tune.grid_search(list(cs_noise_std)),
        "cs_weight": tune.grid_search(list(cs_weight)),
        "fgsm_lambda": tune.grid_search(list(fgsm_lambda)),
        "fgsm_interval": tune.grid_search(list(fgsm_interval)),
        "online_noise_std": tune.grid_search(list(online_noise_std)),
        "online_steps": tune.grid_search(list(online_steps))},
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel})


@cli.command()
@click.option('--local-dir', type=str, default='./data')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
def sweep_sc(local_dir, cpus, gpus, num_parallel):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem
    """

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(run_experiment, config={
        "local_dir": local_dir,
        "init_lr": tune.grid_search([0.0001]),
        "num_epochs": tune.grid_search([100]),
        "hidden_size": tune.grid_search([2048]),
        "solver_lr": tune.grid_search([0.001]),
        "solver_samples": tune.grid_search([32]),
        "solver_steps": tune.grid_search([100]),
        "evaluate_interval": tune.grid_search([1]),
        "sc_noise_std": tune.grid_search([0.5, 0.1, 0.05, 0.01]),
        "sc_lambda": tune.grid_search([50.0, 10.0, 5.0, 1.0]),
        "sc_weight": tune.grid_search([5.0, 1.0, 0.5, 0.1]),
        "cs_noise_std": tune.grid_search([0.0]),
        "cs_weight": tune.grid_search([0.0]),
        "fgsm_lambda": tune.grid_search([0.0]),
        "fgsm_interval": tune.grid_search([1]),
        "online_noise_std": tune.grid_search([0.0]),
        "online_steps": tune.grid_search([0])},
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel})


@cli.command()
@click.option('--local-dir', type=str, default='./data')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
def sweep_cs(local_dir, cpus, gpus, num_parallel):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem
    """

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(run_experiment, config={
        "local_dir": local_dir,
        "init_lr": tune.grid_search([0.0001]),
        "num_epochs": tune.grid_search([100]),
        "hidden_size": tune.grid_search([2048]),
        "solver_lr": tune.grid_search([0.001]),
        "solver_samples": tune.grid_search([32]),
        "solver_steps": tune.grid_search([100]),
        "evaluate_interval": tune.grid_search([1]),
        "sc_noise_std": tune.grid_search([0.0]),
        "sc_lambda": tune.grid_search([0.0]),
        "sc_weight": tune.grid_search([0.0]),
        "cs_noise_std": tune.grid_search([5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]),
        "cs_weight": tune.grid_search([5.0, 1.0, 0.5, 0.1]),
        "fgsm_lambda": tune.grid_search([0.0]),
        "fgsm_interval": tune.grid_search([1]),
        "online_noise_std": tune.grid_search([0.0]),
        "online_steps": tune.grid_search([0])},
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel})


@cli.command()
@click.option('--local-dir', type=str, default='./data')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
def sweep_fgsm(local_dir, cpus, gpus, num_parallel):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem
    """

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(run_experiment, config={
        "local_dir": local_dir,
        "init_lr": tune.grid_search([0.0001]),
        "num_epochs": tune.grid_search([100]),
        "hidden_size": tune.grid_search([2048]),
        "solver_lr": tune.grid_search([0.001]),
        "solver_samples": tune.grid_search([32]),
        "solver_steps": tune.grid_search([100]),
        "evaluate_interval": tune.grid_search([1]),
        "sc_noise_std": tune.grid_search([0.0]),
        "sc_lambda": tune.grid_search([0.0]),
        "sc_weight": tune.grid_search([0.0]),
        "cs_noise_std": tune.grid_search([0.0]),
        "cs_weight": tune.grid_search([0.0]),
        "fgsm_lambda": tune.grid_search([0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]),
        "fgsm_interval": tune.grid_search([50, 10, 5, 1]),
        "online_noise_std": tune.grid_search([0.0]),
        "online_steps": tune.grid_search([0])},
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel})


@cli.command()
@click.option('--local-dir', type=str, default='./data')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
def sweep_online(local_dir, cpus, gpus, num_parallel):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem
    """

    ray.init(num_cpus=cpus, num_gpus=gpus)
    tune.run(run_experiment, config={
        "local_dir": local_dir,
        "init_lr": tune.grid_search([0.0001]),
        "num_epochs": tune.grid_search([100]),
        "hidden_size": tune.grid_search([2048]),
        "solver_lr": tune.grid_search([0.001]),
        "solver_samples": tune.grid_search([32]),
        "solver_steps": tune.grid_search([100]),
        "evaluate_interval": tune.grid_search([1]),
        "sc_noise_std": tune.grid_search([0.0]),
        "sc_lambda": tune.grid_search([0.0]),
        "sc_weight": tune.grid_search([0.0]),
        "cs_noise_std": tune.grid_search([0.0]),
        "cs_weight": tune.grid_search([0.0]),
        "fgsm_lambda": tune.grid_search([0.0]),
        "fgsm_interval": tune.grid_search([1]),
        "online_noise_std": tune.grid_search([0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]),
        "online_steps": tune.grid_search([50, 10, 5, 1])},
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel})
