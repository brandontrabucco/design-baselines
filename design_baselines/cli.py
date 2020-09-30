import click


@click.group()
def cli():
    """A group of potential sub methods that are available for use through
    a command line interface
    """


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--xlabel', type=str)
@click.option('--ylabel', type=str)
@click.option('--separate-runs', is_flag=True)
def plot(dir, tag, xlabel, ylabel, separate_runs):

    from collections import defaultdict
    import glob
    import os
    import re
    import pickle as pkl
    import pandas as pd
    import tensorflow as tf
    import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt

    def pretty(s):
        return s.replace('_', ' ').title()

    # get the experiment ids
    pattern = re.compile(r'.*/(\w+)_(\d+)_(\w+=[\w.+-]+[,_])*(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\w{10})$')
    dirs = [d for d in glob.glob(os.path.join(dir, '*')) if pattern.search(d) is not None]
    matches = [pattern.search(d) for d in dirs]
    ids = [int(m.group(2)) for m in matches]

    # sort the files by the experiment ids
    zipped_lists = zip(ids, dirs)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    ids, dirs = [list(tuple) for tuple in tuples]

    # get the hyper parameters for each experiment
    params = []
    for d in dirs:
        with open(os.path.join(d, 'params.pkl'), 'rb') as f:
            params.append(pkl.load(f))

    # concatenate all params along axis 1
    all_params = defaultdict(list)
    for p in params:
        for key, val in p.items():
            if val not in all_params[key]:
                all_params[key].append(val)

    # locate the params of variation in this experiment
    params_of_variation = []
    for key, val in all_params.items():
        if len(val) > 1 and (not isinstance(val[0], dict)
                             or 'seed' not in val[0]):
            params_of_variation.append(key)

    # get the task and algorithm name
    task_name = params[0]['task']
    algo_name = matches[0].group(1)
    if len(params_of_variation) == 0:
        params_of_variation.append('task')

    # read data from tensor board
    data = pd.DataFrame(columns=['id', xlabel, ylabel] + params_of_variation)
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag and e.step < 500:
                        row = {'id': i,
                               ylabel: tf.make_ndarray(v.tensor).tolist(),
                               xlabel: e.step}
                        for key in params_of_variation:
                            row[key] = f'{pretty(key)} = {p[key]}'
                        data = data.append(row, ignore_index=True)

    if separate_runs:
        params_of_variation.append('id')

    # save a separate plot for every hyper parameter
    for key in params_of_variation:
        plt.clf()
        g = sns.relplot(x=xlabel, y=ylabel, hue=key, data=data,
                        kind="line", height=5, aspect=2,
                        facet_kws={"legend_out": True})
        g.set(title=f'Evaluating {pretty(algo_name)} On {task_name}')
        plt.savefig(f'{algo_name}_{task_name}_{key}_{tag.replace("/", "_")}.png',
                    bbox_inches='tight')


#############


@cli.command()
@click.option('--task', type=str)
@click.option('--task-kwargs', type=str, default="{}")
@click.option('--name', type=str)
def plot_task(task, task_kwargs, name):

    import ast
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    sns.set_style("whitegrid")
    sns.set_context(
        "notebook",
        font_scale=3.5,
        rc={"lines.linewidth": 3.5,
            "grid.linewidth": 2.5,
            'axes.titlesize': 64})

    import design_bench
    t = design_bench.make(task, **ast.literal_eval(task_kwargs))
    df = pd.DataFrame({name: t.y[:, 0]})

    plt.clf()
    g = sns.displot(df, x=name, bins=50,
                    kind="hist", stat="count",
                    height=8, aspect=2)
    g.ax.spines['left'].set_color('black')
    g.ax.spines['left'].set_linewidth(3.5)
    g.ax.spines['bottom'].set_color('black')
    g.ax.spines['bottom'].set_linewidth(3.5)
    g.ax.set_title(f'{task}', pad=64)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig(f'{task}.png',
                bbox_inches='tight')


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--xlabel', type=str)
@click.option('--ylabel', type=str)
@click.option('--pkey', type=str, default='perturbation_backprop')
@click.option('--pval', type=str, default='True')
@click.option('--iteration', type=int, default=50)
@click.option('--legend', is_flag=True)
def plot_one(dir, tag, xlabel, ylabel, pkey, pval, iteration, legend):

    import glob
    import os
    import re
    import pickle as pkl
    import pandas as pd
    import tensorflow as tf
    import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_style("whitegrid")
    sns.set_context("notebook",
                    font_scale=3.5,
                    rc={"lines.linewidth": 3.5,
                        'grid.linewidth': 2.5})

    # get the experiment ids
    pattern = re.compile(r'.*/(\w+)_(\d+)_(\w+=[\w.+-]+[,_])*(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\w{10})$')
    dirs = [d for d in glob.glob(os.path.join(dir, '*')) if pattern.search(d) is not None]
    matches = [pattern.search(d) for d in dirs]
    ids = [int(m.group(2)) for m in matches]

    # sort the files by the experiment ids
    zipped_lists = zip(ids, dirs)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    ids, dirs = [list(tuple) for tuple in tuples]

    # get the hyper parameters for each experiment
    params = []
    for d in dirs:
        with open(os.path.join(d, 'params.pkl'), 'rb') as f:
            params.append(pkl.load(f))

    # get the task and algorithm name
    task_name = params[0]['task']
    algo_name = matches[0].group(1)

    # read data from tensor board
    data = pd.DataFrame(columns=['id', xlabel, ylabel])
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag and str(p[pkey]) == pval:
                        data = data.append({
                            'id': i,
                            ylabel: tf.make_ndarray(v.tensor).tolist(),
                            xlabel: e.step}, ignore_index=True)

    # get the best sample in the dataset
    import design_bench
    import numpy as np
    if 'num_parallel' in params[0]['task_kwargs']:
        params[0]['task_kwargs']['num_parallel'] = 8
    task = design_bench.make(params[0]['task'], **params[0]['task_kwargs'])
    ind = np.argsort(task.y[:, 0])[::-1][:128]
    best_y = task.score(task.x[ind]).max()

    # save a separate plot for every hyper parameter
    plt.clf()
    g = sns.relplot(x=xlabel, y=ylabel, data=data,
                    kind="line", height=10, aspect=1.33)
    plt.plot([iteration, iteration],
             [data[ylabel].to_numpy().min(), data[ylabel].to_numpy().max()],
             '--', c='black', label='Evaluation Point')
    plt.plot([data[xlabel].to_numpy().min(), data[xlabel].to_numpy().max()],
             [best_y, best_y],
             '-.', c='orange', label='Best Observed')
    if legend:
        plt.legend(loc='lower right')
    g.set(title=f'{task_name}')
    plt.savefig(f'{algo_name}_{task_name}_{tag.replace("/", "_")}.png',
                bbox_inches='tight')


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--label', type=str)
@click.option('--pone', type=str, default='perturbation_steps')
@click.option('--ptwo', type=str, default='initial_alpha')
@click.option('--iterations', multiple=True, default=list(range(0, 220, 20)))
def ablation_heatmap(dir, tag, label, pone, ptwo, iterations):

    import glob
    import os
    import re
    import pickle as pkl
    import pandas as pd
    import tensorflow as tf
    import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    def pretty(s):
        return s.replace('_', ' ').title()

    sns.set_style("whitegrid")
    sns.set_context("notebook",
                    font_scale=2.5,
                    rc={"lines.linewidth": 3.5,
                        'grid.linewidth': 2.5})

    # get the experiment ids
    pattern = re.compile(r'.*/(\w+)_(\d+)_(\w+=[\w.+-]+[,_])*(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\w{10})$')
    dirs = [d for d in glob.glob(os.path.join(dir, '*')) if pattern.search(d) is not None]
    matches = [pattern.search(d) for d in dirs]
    ids = [int(m.group(2)) for m in matches]

    # sort the files by the experiment ids
    zipped_lists = zip(ids, dirs)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    ids, dirs = [list(tuple) for tuple in tuples]

    # get the hyper parameters for each experiment
    params = []
    for d in dirs:
        with open(os.path.join(d, 'params.pkl'), 'rb') as f:
            params.append(pkl.load(f))

    # get the task and algorithm name
    task_name = params[0]['task']
    algo_name = matches[0].group(1)

    # read data from tensor board
    data = pd.DataFrame(columns=[label,
                                 pretty(pone),
                                 pretty(ptwo),
                                 'Solver Steps'])
    for d, p in tqdm.tqdm(zip(dirs, params)):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag and e.step in iterations:
                        data = data.append({
                            label: tf.make_ndarray(v.tensor).tolist(),
                            pretty(pone): p[pone],
                            pretty(ptwo): p[ptwo],
                            'Solver Steps': e.step
                            }, ignore_index=True)

    # get the best sample in the dataset
    plt.clf()
    pivot = pd.pivot_table(data,
                           index=pretty(pone),
                           columns=pretty(ptwo),
                           values=label,
                           aggfunc=np.mean)
    sns.heatmap(pivot)
    plt.title(f'{task_name}')
    plt.savefig(f'{algo_name}_{task_name}_{tag.replace("/", "_")}_{pone}_{ptwo}.png',
                bbox_inches='tight')

    # get the best sample in the dataset
    plt.clf()
    pivot = pd.pivot_table(data,
                           index='Solver Steps',
                           columns=pretty(ptwo),
                           values=label,
                           aggfunc=np.mean)
    sns.heatmap(pivot)
    plt.title(f'{task_name}')
    plt.savefig(f'{algo_name}_{task_name}_{tag.replace("/", "_")}_solver_steps_{ptwo}.png',
                bbox_inches='tight')

    # get the best sample in the dataset
    plt.clf()
    pivot = pd.pivot_table(data,
                           index=pretty(pone),
                           columns='Solver Steps',
                           values=label,
                           aggfunc=np.mean)
    sns.heatmap(pivot)
    plt.title(f'{task_name}')
    plt.savefig(f'{algo_name}_{task_name}_{tag.replace("/", "_")}_{pone}_solver_steps.png',
                bbox_inches='tight')


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--iteration', type=int)
def evaluate(dir, tag, iteration):

    from collections import defaultdict
    import pickle as pkl
    import glob
    import os
    import re
    import numpy as np
    import tensorflow as tf
    import tqdm

    def pretty(s):
        return s.replace('_', ' ').title()

    # get the experiment ids
    pattern = re.compile(r'.*/(\w+)_(\d+)_(\w+=[\w.+-]+[,_])*(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\w{10})$')
    dirs = [d for d in glob.glob(os.path.join(dir, '*')) if pattern.search(d) is not None]
    matches = [pattern.search(d) for d in dirs]
    ids = [int(m.group(2)) for m in matches]

    # sort the files by the experiment ids
    zipped_lists = zip(ids, dirs)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    ids, dirs = [list(tuple) for tuple in tuples]

    # get the hyper parameters for each experiment
    params = []
    for d in dirs:
        with open(os.path.join(d, 'params.pkl'), 'rb') as f:
            params.append(pkl.load(f))

    # concatenate all params along axis 1
    all_params = defaultdict(list)
    for p in params:
        for key, val in p.items():
            if val not in all_params[key]:
                all_params[key].append(val)

    # locate the params of variation in this experiment
    params_of_variation = []
    for key, val in all_params.items():
        if len(val) > 1 and (not isinstance(val[0], dict)
                             or 'seed' not in val[0]):
            params_of_variation.append(key)

    # get the task and algorithm name
    if len(params_of_variation) == 0:
        params_of_variation.append('task')

    # read data from tensor board
    param_to_scores = defaultdict(list)
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag and e.step == iteration:
                        for key in params_of_variation:
                            key = f'{pretty(key)} = {p[key]}'
                            param_to_scores[key].append(tf.make_ndarray(v.tensor).tolist())

    # return the mean score and standard deviation
    for key in param_to_scores:
        if len(param_to_scores[key]) > 0:
            scores = np.array(param_to_scores[key])
            mean = np.mean(scores)
            std = np.std(scores - mean)
            print(f"key: {key}\n\tmean: {mean}\n\tstd: {std}")
