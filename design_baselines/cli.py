import click


@click.group()
def cli():
    """A group of potential sub methods that are available for use through
    a command line interface
    """


"""

design-baselines compare-runs \
--hopper ~/final-results/online/online-hopper/online/ \
--hopper ~/final-results/online/gradient-ascent-hopper/gradient_ascent/ \
--superconductor ~/final-results/online/online-superconductor/online/ \
--superconductor ~/final-results/online/gradient-ascent-superconductor/gradient_ascent/ \
--gfp ~/final-results/online/online-gfp/online/ \
--gfp ~/final-results/online/gradient-ascent-gfp/gradient_ascent/ \
--molecule ~/final-results/online/online-molecule/online/ \
--molecule ~/final-results/online/gradient-ascent-molecule/gradient_ascent/ \
--names 'Conservative Objective Models' \
--names 'Gradient Ascent' \
--tag 'score/100th' \
--max-iterations 200

"""


@cli.command()
@click.option('--hopper', multiple=True)
@click.option('--superconductor', multiple=True)
@click.option('--gfp', multiple=True)
@click.option('--molecule', multiple=True)
@click.option('--names', multiple=True)
@click.option('--tag', type=str)
@click.option('--max-iterations', type=int)
def compare_runs(hopper,
                 superconductor,
                 gfp,
                 molecule,
                 names,
                 tag,
                 max_iterations):

    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    import os
    import re
    import pandas as pd
    import tensorflow as tf
    import tqdm

    plt.rcParams['text.usetex'] = True
    matplotlib.rc('font', family='serif', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    color_palette = ['#EE7733',
                     '#0077BB',
                     '#33BBEE',
                     '#009988',
                     '#CC3311',
                     '#EE3377',
                     '#BBBBBB',
                     '#000000']
    palette = sns.color_palette(color_palette)
    sns.palplot(palette)
    sns.set_palette(palette)

    pattern = re.compile(
        r'.*/(\w+)_(\d+)_(\w+=[\w.+-]+[,_])*'
        r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\w{10})$')

    name_to_dir = {}

    for (hopper_i,
         superconductor_i,
         gfp_i,
         molecule_i,
         names_i) in zip(
            hopper,
            superconductor,
            gfp,
            molecule,
            names):

        hopper_dir = [d for d in glob.glob(
            os.path.join(hopper_i, '*'))
            if pattern.search(d) is not None]
        superconductor_dir = [d for d in glob.glob(
            os.path.join(superconductor_i, '*'))
            if pattern.search(d) is not None]
        gfp_dir = [d for d in glob.glob(
            os.path.join(gfp_i, '*'))
            if pattern.search(d) is not None]
        molecule_dir = [d for d in glob.glob(
            os.path.join(molecule_i, '*'))
            if pattern.search(d) is not None]

        name_to_dir[names_i] = {
            'HopperController-v0': hopper_dir,
            'Superconductor-v0': superconductor_dir,
            'GFP-v0': gfp_dir,
            'MoleculeActivity-v0': molecule_dir}

    task_to_ylabel = {
        'HopperController-v0': "Average Return",
        'Superconductor-v0': "Critical Temperature",
        'GFP-v0': "Protein Fluorescence",
        'MoleculeActivity-v0': "Drug Activity"}

    fig, axes = plt.subplots(
        nrows=1, ncols=4, figsize=(25.0, 5.0))

    task_to_axis = {
        'HopperController-v0': axes[0],
        'Superconductor-v0': axes[1],
        'GFP-v0': axes[2],
        'MoleculeActivity-v0': axes[3]}

    for task in [
            'HopperController-v0',
            'Superconductor-v0',
            'GFP-v0',
            'MoleculeActivity-v0']:

        # read data from tensor board
        ylabel = task_to_ylabel[task]
        data = pd.DataFrame(columns=[
            'Algorithm',
            'Gradient Ascent Steps',
            ylabel])

        for name, task_to_dir_i in name_to_dir.items():

            for d in tqdm.tqdm(task_to_dir_i[task]):
                for f in glob.glob(os.path.join(d, '*/events.out*')):
                    for e in tf.compat.v1.train.summary_iterator(f):
                        for v in e.summary.value:
                            if v.tag == tag and e.step < max_iterations:

                                data = data.append({
                                    'Algorithm': name,
                                    'Gradient Ascent Steps': e.step,
                                    ylabel: tf.make_ndarray(v.tensor).tolist(),
                                    }, ignore_index=True)

        axis = task_to_axis[task]

        sns.lineplot(
            x='Gradient Ascent Steps',
            y=ylabel,
            hue='Algorithm',
            data=data,
            ax=axis,
            legend=False)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel('Gradient Ascent Steps', fontsize=24)
        axis.set_ylabel(ylabel, fontsize=24)
        axis.set_title(task, fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=1)

    plt.legend(name_to_dir.keys(),
               ncol=len(name_to_dir.keys()),
               loc='lower center',
               bbox_to_anchor=(-1.3, -0.5),
               fontsize=16,
               fancybox=True)
    fig.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    fig.savefig('compare_runs.png')


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
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rcParams['text.usetex'] = True

    matplotlib.rc('font', family='serif', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 13}

    matplotlib.rc('font', **font)

    color_palette = ['#EE7733', '#0077BB', '#33BBEE', '#009988', '#CC3311', '#EE3377', '#BBBBBB', '#000000']

    palette = sns.color_palette(color_palette)
    sns.palplot(palette)
    sns.set_palette(palette)

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
                    if v.tag == tag and e.step:
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
@click.option('--param', type=str)
@click.option('--xlabel', type=str)
@click.option('--ylabel', type=str)
@click.option('--eval-tag', type=str)
def plot_comparison(dir, tag, param, xlabel, ylabel, eval_tag):

    import glob
    import os
    import re
    import numpy as np
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
        it_to_tag = dict()
        it_to_eval_tag = dict()
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag:
                        it_to_tag[e.step] = tf.make_ndarray(v.tensor).tolist()
                    if v.tag == eval_tag:
                        it_to_eval_tag[e.step] = tf.make_ndarray(v.tensor).tolist()

        if len(it_to_eval_tag) > 0:
            eval_position = int(np.argmax(list(it_to_eval_tag.values())))
            iteration = list(it_to_eval_tag.keys())[eval_position]
            score = it_to_tag[iteration]
            data = data.append({
                'id': i,
                ylabel: score,
                xlabel: p[param]}, ignore_index=True)

    # save a separate plot for every hyper parameter
    plt.clf()
    g = sns.relplot(x=xlabel, y=ylabel, data=data,
                    kind="line", height=10, aspect=1.33)
    g.set(title=f'Ablate {task_name}')
    plt.savefig(f'{algo_name}_{task_name}_ablate_{param}_{tag.replace("/", "_")}.png',
                bbox_inches='tight')


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--eval-tag', type=str)
def evaluate_offline(dir, tag, eval_tag):

    import glob
    import os
    import re
    import numpy as np
    import pickle as pkl
    import tensorflow as tf
    import tqdm
    import seaborn as sns
    from collections import defaultdict

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
    it_to_tag = defaultdict(list)
    it_to_eval_tag = defaultdict(list)
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag and e.step < 500:
                        it_to_tag[e.step].append(tf.make_ndarray(v.tensor).tolist())
                    if v.tag == eval_tag and e.step < 500:
                        it_to_eval_tag[e.step].append(tf.make_ndarray(v.tensor).tolist())

    keys, values = zip(*it_to_eval_tag.items())
    values = [np.mean(vs) for vs in values]
    iteration = keys[int(np.argmax(values))]
    mean = np.mean(it_to_tag[iteration])
    std = np.std(it_to_tag[iteration])

    # save a separate plot for every hyper parameter
    print(f'Evaluate {task_name} At {iteration}\n\t{mean} +- {std}')


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--iteration', type=int)
def evaluate_fixed(dir, tag, iteration):

    import glob
    import os
    import re
    import numpy as np
    import pickle as pkl
    import tensorflow as tf
    import tqdm
    import seaborn as sns
    from collections import defaultdict

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
    it_to_tag = defaultdict(list)
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag and e.step < 500:
                        it_to_tag[e.step].append(tf.make_ndarray(v.tensor).tolist())

    if iteration in it_to_tag:
        mean = np.mean(it_to_tag[iteration])
        std = np.std(it_to_tag[iteration])

        # save a separate plot for every hyper parameter
        print(f'Evaluate {task_name} At {iteration}\n\t{mean} +- {std}')


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
