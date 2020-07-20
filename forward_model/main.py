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
@click.option('--sc/--no-sc', default=[False], multiple=True)
@click.option('--sc-noise_std', type=click.FLOAT, default=[0.3], multiple=True)
@click.option('--sc-lambda', type=click.FLOAT, default=[10.0], multiple=True)
@click.option('--sc-weight', type=click.FLOAT, default=[1.0], multiple=True)
@click.option('--cs/--no-cs', default=[False], multiple=True)
@click.option('--cs-noise-std', type=click.FLOAT, default=[0.3], multiple=True)
@click.option('--cs-weight', type=click.FLOAT, default=[1.0], multiple=True)
@click.option('--fgsm/--no-fgsm', default=[False], multiple=True)
@click.option('--fgsm-lambda', type=click.FLOAT, default=[0.01], multiple=True)
@click.option('--fgsm-per-epoch', type=click.INT, default=[1], multiple=True)
@click.option('--online/--not-online', default=[False], multiple=True)
@click.option('--online-noise-std', type=click.FLOAT, default=[0.3], multiple=True)
@click.option('--online-steps', type=click.INT, default=[8], multiple=True)
def train(local_dir, cpus, gpus, num_parallel,
          init_lr, num_epochs, hidden_size,
          solver_lr, solver_samples, solver_steps, evaluate_interval,
          sc, sc_noise_std, sc_lambda, sc_weight,
          cs, cs_noise_std, cs_weight,
          fgsm, fgsm_lambda, fgsm_per_epoch,
          online, online_noise_std, online_steps):
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
        "use_sc": tune.grid_search(list(sc)),
        "sc_noise_std": tune.grid_search(list(sc_noise_std)),
        "sc_lambda": tune.grid_search(list(sc_lambda)),
        "sc_weight": tune.grid_search(list(sc_weight)),
        "use_cs": tune.grid_search(list(cs)),
        "cs_noise_std": tune.grid_search(list(cs_noise_std)),
        "cs_weight": tune.grid_search(list(cs_weight)),
        "use_fgsm": tune.grid_search(list(fgsm)),
        "fgsm_lambda": tune.grid_search(list(fgsm_lambda)),
        "fgsm_per_epoch": tune.grid_search(list(fgsm_per_epoch)),
        "use_online": tune.grid_search(list(online)),
        "online_noise_std": tune.grid_search(list(online_noise_std)),
        "online_steps": tune.grid_search(list(online_steps))},
        resources_per_trial={
            'cpu': cpus // num_parallel,
            'gpu': gpus / num_parallel})
