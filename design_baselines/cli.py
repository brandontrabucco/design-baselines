import click


@click.group()
def cli():
    """A group of potential sub methods that are available for use through
    a command line interface
    """


@cli.command()
@click.option('--dir', type=str)
@click.option('--percentile', type=float, default=100.)
@click.option('--modifier', type=str, default="-fidelity")
@click.option('--load', is_flag=True, default=False)
def agreement_heatmap(dir, percentile, modifier, load):

    import glob
    import os
    import tensorflow as tf
    import tqdm
    import numpy as np
    import pandas as pd
    import itertools
    import scipy.stats as stats
    from collections import defaultdict

    from collections import defaultdict
    import glob
    import os
    import re
    import pickle as pkl
    import tensorflow as tf
    import tqdm
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import design_bench as db
    from copy import deepcopy

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

    tasks = [
        "gfp",
        "superconductor"
    ]

    baselines = [
        "autofocused-cbas",
        "cbas",
        "bo-qei",
        # "cma-es", didn't finish by the deadline
        "gradient-ascent",
        "gradient-ascent-min-ensemble",
        "gradient-ascent-mean-ensemble",
        "mins",
        "reinforce"
    ]

    metrics = [
        "rank-correlation",
        "max-shift",
        "avg-shift"
    ]

    baseline_to_logits = {
        "autofocused-cbas": False,
        "cbas": False,
        "bo-qei": True,
        "cma-es": True,
        "gradient-ascent": True,
        "gradient-ascent-min-ensemble": True,
        "gradient-ascent-mean-ensemble": True,
        "mins": False,
        "reinforce": False
    }

    task_to_oracles = {
        "gfp": [
            "GP",
            "RandomForest",
            "FullyConnected",
            "ResNet",
            "Transformer"
        ],
        "superconductor": [
            # "GP",
            "RandomForest",
            "FullyConnected"
        ]
    }

    p = defaultdict(list)
    task_pattern = re.compile(r'(\w+)-(\w+)-v(\d+)$')

    if not load:
        for baseline, task in tqdm.tqdm(
                list(itertools.product(baselines, tasks))):

            is_logits = baseline_to_logits[baseline]
            files = glob.glob(os.path.join(dir, f"{baseline}{modifier}-"
                              f"{task}/*/*/*/solution.npy"))

            for f in files:

                solution_tensor = np.load(f)

                params = os.path.join(os.path.dirname(
                    os.path.dirname(f)), "params.json")

                with open(params, "r") as params_file:
                    params = json.load(params_file)

                for oracle in task_to_oracles[task]:
                    matches = task_pattern.search(params["task"])
                    db_task = db.make(params["task"].replace(
                        matches.group(2), oracle), **params["task_kwargs"])

                    if is_logits and db_task.is_discrete:
                        db_task.map_to_logits()

                    elif db_task.is_discrete:
                        db_task.map_to_integers()

                    if params["normalize_xs"]:
                        db_task.map_normalize_x()

                    scores = db_task.predict(solution_tensor)
                    p[f"{baseline}-{task}-"
                      f"{oracle}"].append(np.percentile(scores, percentile))

    print("aggregating performance")

    p2 = dict()
    if not load:
        for task in tasks:
            for oracle in task_to_oracles[task]:
                p2[f"{task}-{oracle}"] = [
                    p[f"{baseline}-{task}-{oracle}"]
                    for baseline in baselines]

    print("rendering heatmaps")

    for metric, task in tqdm.tqdm(
            list(itertools.product(metrics, tasks))):

        task_oracles = task_to_oracles[task]

        metric_data = np.zeros([len(task_oracles), len(task_oracles)])

        if not load:
            for i, oracle0 in enumerate(task_oracles):
                for j, oracle1 in enumerate(task_oracles):
                    oracle0_data = deepcopy(p2[f"{task}-{oracle0}"])
                    oracle1_data = deepcopy(p2[f"{task}-{oracle1}"])

                    oracle0_data = [0.0 if len(value) == 0 else
                                    np.mean(value) for value in oracle0_data]
                    oracle1_data = [0.0 if len(value) == 0 else
                                    np.mean(value) for value in oracle1_data]

                    oracle0_data = np.array(oracle0_data)
                    oracle1_data = np.array(oracle1_data)

                    if metric == "rank-correlation":
                        rho = stats.spearmanr(oracle0_data, oracle1_data)[0]
                        metric_data[j][i] = rho

                    elif metric == "max-shift":
                        table0_index = oracle0_data.argsort().argsort()
                        table1_index = oracle1_data.argsort().argsort()
                        max_shift = np.abs(table0_index - table1_index).max()
                        metric_data[j][i] = max_shift

                    elif metric == "avg-shift":
                        table0_index = oracle0_data.argsort().argsort()
                        table1_index = oracle1_data.argsort().argsort()
                        avg_shift = np.abs(table0_index - table1_index).mean()
                        metric_data[j][i] = avg_shift
        else:
            metric_data = np.load(f'{task}{modifier}-{metric}-heatmap.npy')

        # save a separate plot for every hyper parameter
        plt.clf()
        sns.heatmap(metric_data,
                    xticklabels=task_oracles,
                    yticklabels=task_oracles,
                    cbar_kws={'label': metric},
                    square=True, vmin=0,
                    vmax=1 if metric == "rank-correlation" else None)
        plt.title(f"Oracle Agreement: {task}")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig(f'{task}{modifier}-{metric}-heatmap.png', bbox_inches='tight')
        np.save(f'{task}{modifier}-{metric}-heatmap.npy', metric_data)



@cli.command()
@click.option('--table0', type=str)
@click.option('--table1', type=str)
def rank_tables(table0, table1):

    import glob
    import os
    import tensorflow as tf
    import tqdm
    import numpy as np
    import pandas as pd
    import scipy.stats as stats

    tasks = [
        "gfp",
        "tf-bind-8",
        "utr",
        "hopper",
        "superconductor",
        "chembl",
        "ant",
        "dkitty"
    ]

    metrics = [
        "rank-correlation",
        "max-shift",
        "avg-shift"
    ]

    table0_df = pd.read_csv(table0)
    table1_df = pd.read_csv(table1)

    final_data_numeric = [[None for t in tasks] for m in metrics]
    for i, task in enumerate(tasks):
        table0_rank = table0_df[task].to_numpy()
        table1_rank = table1_df[task].to_numpy()
        for j, metric in enumerate(metrics):
            if metric == "rank-correlation":
                rho = stats.spearmanr(table0_rank, table1_rank)[0]
                final_data_numeric[j][i] = rho
            elif metric == "max-shift":
                table0_index = table0_rank.argsort().argsort()
                table1_index = table1_rank.argsort().argsort()
                final_data_numeric[j][i] = np.abs(table0_index - table1_index).max()
            elif metric == "avg-shift":
                table0_index = table0_rank.argsort().argsort()
                table1_index = table1_rank.argsort().argsort()
                final_data_numeric[j][i] = np.abs(table0_index - table1_index).mean()

    final_df_numeric = pd.DataFrame(data=final_data_numeric, columns=tasks, index=metrics)
    print(final_df_numeric.to_latex())
    final_df_numeric.to_csv(f"{os.path.basename(table0)[:-4]}-to-"
                            f"{os.path.basename(table1)[:-4]}-rank-metrics.csv")


@cli.command()
@click.option('--dir', type=str)
@click.option('--samples', type=int, default=128)
@click.option('--percentile', type=int, default=100)
@click.option('--main-table', type=str, default="performance.csv")
@click.option('--load/--no-load', is_flag=True, default=False)
def make_diversity_table(dir, samples, percentile, main_table, load):

    import glob
    import os
    import tqdm
    import numpy as np
    import itertools
    import design_bench as db
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import json
    import pandas as pd

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
    sns.set_palette(palette)

    tasks = [
        "gfp",
        "utr",
    ]

    baselines = [
        "autofocused-cbas",
        "cbas",
        "bo-qei",
        #"cma-es",
        "gradient-ascent",
        "gradient-ascent-min-ensemble",
        "gradient-ascent-mean-ensemble",
        "mins",
        "reinforce"
    ]

    dist_options = [
        "uniform",
        "linear",
        "quadratic",
        "circular",
        "exponential",
    ]

    task_to_name = {
        "gfp": "GFP",
        "utr": "UTR",
    }

    if not load:

        task_to_task = {
            "gfp": db.make("GFP-Transformer-v0"),
            "utr": db.make("UTR-ResNet-v0"),
        }

        baseline_to_logits = {
            "autofocused-cbas": False,
            "cbas": False,
            "bo-qei": True,
            "cma-es": True,
            "gradient-ascent": True,
            "gradient-ascent-min-ensemble": True,
            "gradient-ascent-mean-ensemble": True,
            "mins": False,
            "reinforce": False
        }

        dist_to_performance = dict()

        for dist in dist_options:
            dist_to_performance[dist] = dict()

            for task in tasks:
                dist_to_performance[dist][task] = dict()
                for baseline in baselines:
                    dist_to_performance[dist][task][baseline] = list()

        for task, baseline in tqdm.tqdm(list(itertools.product(tasks, baselines))):
            dirs = glob.glob(os.path.join(dir, f"{baseline}-{task}/*/*"))
            for d in [d for d in dirs if os.path.isdir(d)]:

                solution_files = glob.glob(os.path.join(d, '*/solution.npy'))[:samples]
                for current_solution in solution_files:

                    params = os.path.join(os.path.dirname(
                        os.path.dirname(current_solution)), "params.json")
                    with open(params, "r") as p_file:
                        params = json.load(p_file)

                    dist = params["task_kwargs"]["dataset_kwargs"]["distribution"]

                    db_task = task_to_task[task]
                    is_logits = baseline_to_logits[baseline] and not task == "chembl"

                    if is_logits and db_task.is_discrete:
                        db_task.map_to_logits()

                    elif db_task.is_discrete:
                        db_task.map_to_integers()

                    if params["normalize_xs"]:
                        db_task.map_normalize_x()

                    scores = task_to_task[task].predict(np.load(current_solution))
                    dist_to_performance[dist][task][baseline]\
                        .append(np.percentile(scores, percentile))

        for dist in dist_options:
            for task, baseline in tqdm.tqdm(list(itertools.product(tasks, baselines))):
                mean_perf = np.mean(dist_to_performance[dist][task][baseline])
                dist_to_performance[dist][task][baseline] = mean_perf

        diversity = np.zeros([len(tasks), len(dist_options)])

        for task_idx, task in enumerate(tasks):
            for dist_idx, dist in enumerate(dist_options):
                diversity[task_idx, dist_idx] = np.std([
                    dist_to_performance[dist][task][b] for b in baselines])

        np.save(f"dist-diversity-{percentile}.npy", diversity)

    else:

        diversity = np.load(f"dist-diversity-{percentile}.npy")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    main_table = pd.read_csv(main_table)

    for task_idx, task in enumerate(tasks):

        main_data = main_table[task].to_numpy().std()

        axes[task_idx].bar(np.arange(len(dist_options) + 1),
                           [main_data, *diversity[task_idx]],
                           tick_label=["original"] + dist_options,
                           color=color_palette[:len(dist_options) + 1])

        axes[task_idx].spines['right'].set_visible(False)
        axes[task_idx].spines['top'].set_visible(False)
        axes[task_idx].yaxis.set_ticks_position('left')
        axes[task_idx].xaxis.set_ticks_position('bottom')
        axes[task_idx].yaxis.set_tick_params(labelsize=18, labelrotation=0)
        axes[task_idx].xaxis.set_tick_params(labelsize=18, labelrotation=90)

        axes[task_idx].set_xlabel(r'\textbf{Subsampling Distribution}',
                                  fontsize=18)
        axes[task_idx].set_ylabel(r'\textbf{Standard Deviation Of All Baselines}',
                                  fontsize=18)
        axes[task_idx].set_title(r'\textbf{' + task_to_name[task] + '}',
                                 fontsize=18)
        axes[task_idx].grid(color='grey',
                            linestyle='dotted',
                            linewidth=2)

    plt.tight_layout()
    plt.savefig(f"dist-diversity-{percentile}.png")


@cli.command()
@click.option('--dir', type=str)
@click.option('--samples', type=int, default=128)
@click.option('--percentile', type=int, default=100)
@click.option('--load/--no-load', is_flag=True, default=False)
def make_table_from_distributions(dir, samples, percentile, load):

    import glob
    import os
    import tqdm
    import numpy as np
    import itertools
    import design_bench as db
    import scipy.stats as stats
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import json

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
    sns.set_palette(palette)

    tasks = [
        "gfp",
        "utr",
        "superconductor",
        "hopper",
    ]

    baselines = [
        "autofocused-cbas",
        "cbas",
        "bo-qei",
        #"cma-es",
        "gradient-ascent",
        "gradient-ascent-min-ensemble",
        "gradient-ascent-mean-ensemble",
        "mins",
        "reinforce"
    ]

    dist_options = [
        "uniform",
        "linear",
        "quadratic",
        "circular",
        "exponential",
    ]

    if not load:

        task_to_task = {
            "gfp": db.make("GFP-Transformer-v0"),
            "tf-bind-8": db.make("TFBind8-Exact-v0"),
            "utr": db.make("UTR-ResNet-v0"),
            "chembl": db.make("ChEMBL-ResNet-v0"),
            "superconductor": db.make("Superconductor-RandomForest-v0"),
            "ant": db.make("AntMorphology-Exact-v0"),
            "dkitty": db.make("DKittyMorphology-Exact-v0"),
            "hopper": db.make("HopperController-Exact-v0")
        }

        baseline_to_logits = {
            "autofocused-cbas": False,
            "cbas": False,
            "bo-qei": True,
            "cma-es": True,
            "gradient-ascent": True,
            "gradient-ascent-min-ensemble": True,
            "gradient-ascent-mean-ensemble": True,
            "mins": False,
            "reinforce": False
        }

        dist_to_performance = dict()

        for dist in dist_options:
            dist_to_performance[dist] = dict()

            for task in tasks:
                dist_to_performance[dist][task] = dict()
                for baseline in baselines:
                    dist_to_performance[dist][task][baseline] = list()

        for task, baseline in tqdm.tqdm(list(itertools.product(tasks, baselines))):
            dirs = glob.glob(os.path.join(dir, f"{baseline}-{task}/*/*"))
            for d in [d for d in dirs if os.path.isdir(d)]:

                solution_files = glob.glob(os.path.join(d, '*/solution.npy'))[:samples]
                for current_solution in solution_files:

                    params = os.path.join(os.path.dirname(
                        os.path.dirname(current_solution)), "params.json")
                    with open(params, "r") as p_file:
                        params = json.load(p_file)

                    dist = params["task_kwargs"]["dataset_kwargs"]["distribution"]

                    db_task = task_to_task[task]
                    is_logits = baseline_to_logits[baseline] and not task == "chembl"

                    if is_logits and db_task.is_discrete:
                        db_task.map_to_logits()

                    elif db_task.is_discrete:
                        db_task.map_to_integers()

                    if params["normalize_xs"]:
                        db_task.map_normalize_x()

                    scores = task_to_task[task].predict(np.load(current_solution))
                    dist_to_performance[dist][task][baseline]\
                        .append(np.percentile(scores, percentile))

        for dist in dist_options:
            for task, baseline in tqdm.tqdm(list(itertools.product(tasks, baselines))):
                mean_perf = np.mean(dist_to_performance[dist][task][baseline])
                dist_to_performance[dist][task][baseline] = mean_perf

        correlation = np.zeros([len(dist_options), len(dist_options)])
        for a_idx, b_idx in itertools.product(
                range(len(dist_options)), range(len(dist_options))):

            perf_a = dist_to_performance[dist_options[a_idx]]
            perf_b = dist_to_performance[dist_options[b_idx]]

            correlation[a_idx, b_idx] = np.mean([
                stats.spearmanr(np.array([perf_a[task][b] for b in baselines]),
                                np.array([perf_b[task][b] for b in baselines]))[0]
                for task in tasks
            ])

    else:

        correlation = np.load(f"dist-heatmap-{percentile}.npy")

    mask = np.zeros_like(correlation)
    mask[np.triu_indices_from(mask)] = True
    ax = sns.heatmap(correlation,
                     xticklabels=dist_options,
                     yticklabels=dist_options,
                     mask=mask,
                     vmin=0.0,
                     vmax=1.0,
                     square=True)

    plt.title(r"Sensitivity To Distribution " + f"({percentile}th Percentile)", fontsize=20)
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.ylabel(r"Subsampling Distribution", fontsize=18)
    plt.xlabel(r"Subsampling Distribution", fontsize=18)

    ax.collections[0].colorbar.ax.tick_params(labelsize=18)
    ax.collections[0].colorbar.set_label(label=r"Spearman's $\rho$", size=20)

    plt.tight_layout()
    np.save(f"dist-heatmap-{percentile}.npy", correlation)
    plt.savefig(f"dist-heatmap-{percentile}.png")


@cli.command()
@click.option('--dir', type=str)
@click.option('--distribution', type=str, default="uniform")
@click.option('--percentile', type=int, default=100)
@click.option('--load/--no-load', is_flag=True, default=False)
def make_table_from_solutions(dir, distribution, percentile, load):

    import glob
    import os
    import tqdm
    import numpy as np
    import itertools
    import design_bench as db
    import scipy.stats as stats
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import json

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
    sns.set_palette(palette)

    tasks = [
        "gfp",
        "utr",
        "superconductor",
        "hopper",
    ]

    baselines = [
        "autofocused-cbas",
        "cbas",
        "bo-qei",
        #"cma-es",
        "gradient-ascent",
        "gradient-ascent-min-ensemble",
        "gradient-ascent-mean-ensemble",
        "mins",
        "reinforce"
    ]

    sample_options = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    if not load:

        task_to_task = {
            "gfp": db.make("GFP-Transformer-v0"),
            "tf-bind-8": db.make("TFBind8-Exact-v0"),
            "utr": db.make("UTR-ResNet-v0"),
            "chembl": db.make("ChEMBL-ResNet-v0"),
            "superconductor": db.make("Superconductor-RandomForest-v0"),
            "ant": db.make("AntMorphology-Exact-v0"),
            "dkitty": db.make("DKittyMorphology-Exact-v0"),
            "hopper": db.make("HopperController-Exact-v0")
        }

        baseline_to_logits = {
            "autofocused-cbas": False,
            "cbas": False,
            "bo-qei": True,
            "cma-es": True,
            "gradient-ascent": True,
            "gradient-ascent-min-ensemble": True,
            "gradient-ascent-mean-ensemble": True,
            "mins": False,
            "reinforce": False
        }

        max_samples_to_performance = dict()

        for max_samples in sample_options:
            max_samples_to_performance[max_samples] = dict()

            for task in tasks:
                max_samples_to_performance[max_samples][task] = dict()
                for baseline in baselines:
                    max_samples_to_performance[max_samples][task][baseline] = list()

        for task, baseline in tqdm.tqdm(list(itertools.product(tasks, baselines))):
            dirs = glob.glob(os.path.join(dir, f"{baseline}-{task}/*/*"))
            for d in [d for d in dirs if os.path.isdir(d)]:

                solution_files = glob.glob(os.path.join(d, '*/solution.npy'))
                for current_solution in solution_files:

                    params = os.path.join(os.path.dirname(
                        os.path.dirname(current_solution)), "params.json")
                    with open(params, "r") as p_file:
                        params = json.load(p_file)

                    dataset_kwargs = params["task_kwargs"]["dataset_kwargs"]
                    if dataset_kwargs["distribution"] != distribution:
                        continue

                    db_task = task_to_task[task]
                    is_logits = baseline_to_logits[baseline] and not task == "chembl"

                    if is_logits and db_task.is_discrete:
                        db_task.map_to_logits()

                    elif db_task.is_discrete:
                        db_task.map_to_integers()

                    if params["normalize_xs"]:
                        db_task.map_normalize_x()

                    scores = task_to_task[task].predict(np.load(current_solution))

                    for max_samples in sample_options:
                        max_samples_to_performance[max_samples][task][baseline]\
                            .append(np.percentile(scores[:max_samples], percentile))

        for max_samples in sample_options:
            for task, baseline in tqdm.tqdm(list(itertools.product(tasks, baselines))):

                mean_perf = np.mean(
                    max_samples_to_performance[max_samples][task][baseline])

                max_samples_to_performance[
                    max_samples][task][baseline] = mean_perf

        correlation = np.zeros([len(sample_options), len(sample_options)])
        for a_idx, b_idx in itertools.product(
                range(len(sample_options)), range(len(sample_options))):

            perf_a = max_samples_to_performance[sample_options[a_idx]]
            perf_b = max_samples_to_performance[sample_options[b_idx]]

            correlation[a_idx, b_idx] = np.mean([
                stats.spearmanr(np.array([perf_a[task][b] for b in baselines]),
                                np.array([perf_b[task][b] for b in baselines]))[0]
                for task in tasks
            ])

    else:

        correlation = np.load(f"k-heatmap-{percentile}.npy")

    mask = np.zeros_like(correlation)
    mask[np.triu_indices_from(mask)] = True
    ax = sns.heatmap(correlation,
                     xticklabels=sample_options,
                     yticklabels=sample_options,
                     mask=mask,
                     vmin=0.0,
                     vmax=1.0,
                     square=True)

    plt.title(r"Sensitivity To $K$ " + f"({percentile}th Percentile)", fontsize=20)
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.ylabel(r"Evaluation Budget $K$", fontsize=18)
    plt.xlabel(r"Evaluation Budget $K$", fontsize=18)

    ax.collections[0].colorbar.ax.tick_params(labelsize=18)
    ax.collections[0].colorbar.set_label(label=r"Spearman's $\rho$", size=20)

    plt.tight_layout()
    np.save(f"k-heatmap-{percentile}.npy", correlation)
    plt.savefig(f"k-heatmap-{percentile}.png")


@cli.command()
@click.option('--dir', type=str)
@click.option('--percentile', type=str, default="100th")
@click.option('--modifier', type=str, default="")
@click.option('--group', type=str, default="")
@click.option('--normalize/--no-normalize', is_flag=True, default=True)
def make_table(dir, percentile, modifier, group, normalize):

    import glob
    import os
    import tensorflow as tf
    import tqdm
    import numpy as np
    import pandas as pd

    from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
    from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
    from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset
    from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset
    from design_bench.datasets.discrete.utr_dataset import UTRDataset
    from design_bench.datasets.discrete.gfp_dataset import GFPDataset

    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.hopper_controller_dataset import HopperControllerDataset

    import design_bench as db

    tasks = [
        "tf-bind-8",
        "tf-bind-10",
        "chembl",
        "cifar-nas",
    ] if group == "A" else [
        "superconductor",
        "ant",
        "dkitty",
    ] if group == "B" else [
        "gfp",
        "utr",
        "hopper",
    ] if group == "C" else [
        "tf-bind-8",
        "tf-bind-10",
        "chembl",
        "cifar-nas",
        "superconductor",
        "ant",
        "dkitty",
    ]

    tf_bind_8_dataset = TFBind8Dataset()
    tf_bind_10_dataset = TFBind10Dataset()
    chembl_dataset = ChEMBLDataset(assay_chembl_id="CHEMBL3885882", standard_type="MCHC")
    cifar_nas_dataset = CIFARNASDataset()

    superconductor_dataset = SuperconductorDataset()
    ant_dataset = AntMorphologyDataset()
    dkitty_dataset = DKittyMorphologyDataset()

    utr_dataset = UTRDataset()
    gfp_dataset = GFPDataset()
    hopper_controller_dataset = HopperControllerDataset()

    task_to_min = {
        "tf-bind-8": tf_bind_8_dataset.y.min(),
        "tf-bind-10": tf_bind_10_dataset.y.min(),
        "chembl": chembl_dataset.y.min(),
        "cifar-nas": cifar_nas_dataset.y.min(),
        "superconductor": superconductor_dataset.y.min(),
        "ant": ant_dataset.y.min(),
        "dkitty": dkitty_dataset.y.min(),
        "utr": utr_dataset.y.min(),
        "gfp": gfp_dataset.y.min(),
        "hopper": hopper_controller_dataset.y.min(),
    }

    task_to_max = {
        "tf-bind-8": tf_bind_8_dataset.y.max(),
        "tf-bind-10": tf_bind_10_dataset.y.max(),
        "chembl": chembl_dataset.y.max(),
        "cifar-nas": cifar_nas_dataset.y.max(),
        "superconductor": superconductor_dataset.y.max(),
        "ant": ant_dataset.y.max(),
        "dkitty": dkitty_dataset.y.max(),
        "utr": utr_dataset.y.max(),
        "gfp": gfp_dataset.y.max(),
        "hopper": hopper_controller_dataset.y.max(),
    }

    task_to_best = {
        "tf-bind-8": db.make("TFBind8-Exact-v0").y.max(),
        "tf-bind-10": db.make("TFBind10-Exact-v0", dataset_kwargs=dict(max_samples=10000)).y.max(),
        "chembl": db.make("ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0").y.max(),
        "cifar-nas": db.make("CIFARNAS-Exact-v0").y.max(),
        "superconductor": db.make("Superconductor-RandomForest-v0").y.max(),
        "ant": db.make("AntMorphology-Exact-v0").y.max(),
        "dkitty": db.make("DKittyMorphology-Exact-v0").y.max(),
        "gfp": db.make("GFP-Transformer-v0").y.max(),
        "utr": db.make("UTR-ResNet-v0", relabel=True).y.max(),
        "hopper": db.make("HopperController-Exact-v0").y.max(),
    }

    for task_name, task_best in task_to_best.items():
        task_to_best[task_name] = (task_to_best[task_name] -
                                   task_to_min[task_name]) / (
                task_to_max[task_name] - task_to_min[task_name])
        
    print("D(Best) = ", task_to_best)

    baselines = [
        "autofocused-cbas",
        "cbas",
        "bo-qei",
        "cma-es",
        "gradient-ascent",
        "gradient-ascent-min-ensemble",
        "gradient-ascent-mean-ensemble",
        "mins",
        "reinforce",
        "coms"
    ]

    baseline_to_tag = {
        "autofocused-cbas": [f"score/{percentile}"],
        "cbas": [f"score/{percentile}"],
        "bo-qei": [f"score/{percentile}"],
        "cma-es": [f"score/{percentile}"],
        "gradient-ascent": [f"score/{percentile}"],
        "gradient-ascent-min-ensemble": [f"score/{percentile}"],
        "gradient-ascent-mean-ensemble": [f"score/{percentile}"],
        "mins": [f"exploitation/actual_ys/{percentile}", f"score/{percentile}"],
        "reinforce": [f"score/{percentile}"],
        "coms": [f"score/{percentile}"]
    }

    baseline_to_iteration = {
        "autofocused-cbas": 20,
        "cbas": 20,
        "bo-qei": 10,
        "cma-es": 0,
        "gradient-ascent": 200,
        "gradient-ascent-min-ensemble": 200,
        "gradient-ascent-mean-ensemble": 200,
        "mins": 0,
        "reinforce": 200,
        "coms": 50
    }

    performance = dict()
    for task in tqdm.tqdm(tasks):
        task_min = task_to_min[task]
        task_max = task_to_max[task]
        performance[task] = dict()
        for baseline in baselines:
            performance[task][baseline] = list()

            if baseline == "coms":

                dirs = [d for d in glob.glob(os.path.join(
                    dir, f"coms-{task}/coms-{task}-{modifier}*/*")) if os.path.isdir(d)]

            else:

                dirs = f"{baseline}{modifier}-{task}/*/*"
                if task == "utr":
                    dirs = f"{baseline}-relabelled-{task}/*/*"

                dirs = [d for d in glob.glob(
                    os.path.join(dir, dirs)) if os.path.isdir(d)]

            for d in dirs:
                event_files = (
                    list(glob.glob(os.path.join(d, '*/events.out*'))) +
                    list(glob.glob(os.path.join(d, 'events.out*')))
                )
                for f in event_files:
                    for e in tf.compat.v1.train.summary_iterator(f):
                        for v in e.summary.value:

                            if v.tag in baseline_to_tag[baseline]\
                                    and e.step == baseline_to_iteration[baseline]:
                                score = tf.make_ndarray(v.tensor)
                                performance[task][baseline].append(
                                    ((score - task_min) / (
                                        task_max - task_min)) if normalize else score
                                )

    final_data = [[None for t in tasks] for b in baselines]
    final_data_mean = [[None for t in tasks] for b in baselines]
    final_data_standard_dev = [[None for t in tasks] for b in baselines]
    for i, task in enumerate(tasks):
        for j, baseline in enumerate(baselines):
            data = np.array(performance[task][baseline])
            mean = 0.0
            standard_dev = 0.0
            if data.shape[0] > 0:
                mean = np.mean(data)
            if data.shape[0] > 1:
                standard_dev = np.std(data - mean)
            final_data[j][i] = f"{mean:0.3f} ± {standard_dev:0.3f}"
            final_data_mean[j][i] = mean
            final_data_standard_dev[j][i] = standard_dev

    final_data_mean = np.asarray(final_data_mean)
    final_data_standard_dev = np.asarray(final_data_standard_dev)

    final_df = pd.DataFrame(data=final_data, columns=tasks, index=baselines)
    final_df_numeric = pd.DataFrame(data=final_data_mean, columns=tasks, index=baselines)
    print(final_df.to_latex())
    final_df_numeric.to_csv(f"performance{modifier}.csv")

    #
    # average performance only makes sense when data is normalized
    #

    final_average_perf = final_data_mean.mean(axis=1)

    print()
    print("Average Performance: ")
    for baseline_idx, baseline in enumerate(baselines):
        print(f"{baseline} = ", final_average_perf[baseline_idx])

    #
    # how many tasks is a particular method optimal (or within 1 sd)
    #
    optimal_tasks = dict()
    for baseline in baselines:
        optimal_tasks[baseline] = list()

    final_data_optimality = np.zeros([len(baselines), len(tasks)])
    for task_idx, task in enumerate(tasks):
        top_idx = (final_data_mean[:, task_idx] - final_data_standard_dev[:, task_idx]).argmax()

        top_mean = final_data_mean[top_idx, task_idx]
        top_standard_dev = final_data_standard_dev[top_idx, task_idx]
        for baseline_idx, baseline in enumerate(baselines):
            current_mean = final_data_mean[baseline_idx, task_idx]
            current_standard_dev = final_data_standard_dev[baseline_idx, task_idx]

            if current_mean + current_standard_dev >= top_mean \
                    or current_mean >= top_mean - top_standard_dev:
                final_data_optimality[baseline_idx, task_idx] += 1.0
                optimal_tasks[baseline].append(task)

    final_data_optimality = final_data_optimality.sum(axis=1)

    print()
    print("Number Of Optimal Tasks: ")
    for baseline_idx, baseline in enumerate(baselines):
        print(f"{baseline} = ", int(final_data_optimality[baseline_idx]), "/ 7", optimal_tasks[baseline])


@cli.command()
@click.option('--dir', type=str)
@click.option('--percentile', type=str, default="100th")
@click.option('--modifier', type=str, default="")
def stochasticity_table(dir, percentile, modifier):

    import glob
    import os
    import tensorflow as tf
    import tqdm
    import numpy as np
    import pandas as pd
    import itertools
    import scipy.stats as stats

    tasks = [
        "gfp",
        "tf-bind-8",
        "utr",
        "hopper",
        "superconductor",
        "chembl",
        "ant",
        "dkitty"
    ]

    baselines = [
        "autofocused-cbas",
        "cbas",
        "bo-qei",
        "cma-es",
        "gradient-ascent",
        "gradient-ascent-min-ensemble",
        "gradient-ascent-mean-ensemble",
        "mins",
        "reinforce"
    ]

    metrics = [
        "rank-correlation",
        "max-shift",
        "avg-shift"
    ]

    baseline_to_tag = {
        "autofocused-cbas": f"score/{percentile}",
        "cbas": f"score/{percentile}",
        "bo-qei": f"score/{percentile}",
        "cma-es": f"score/{percentile}",
        "gradient-ascent": f"score/{percentile}",
        "gradient-ascent-min-ensemble": f"score/{percentile}",
        "gradient-ascent-mean-ensemble": f"score/{percentile}",
        "mins": f"exploitation/actual_ys/{percentile}",
        "reinforce": f"score/{percentile}"
    }

    baseline_to_iteration = {
        "autofocused-cbas": 20,
        "cbas": 20,
        "bo-qei": 10,
        "cma-es": 0,
        "gradient-ascent": 200,
        "gradient-ascent-min-ensemble": 200,
        "gradient-ascent-mean-ensemble": 200,
        "mins": 0,
        "reinforce": 200
    }

    performance = dict()
    for task in tqdm.tqdm(tasks):
        performance[task] = dict()
        for baseline in baselines:
            performance[task][baseline] = list()

            dirs = [d for d in glob.glob(os.path.join(
                dir, f"{baseline}{modifier}-{task}/*/*")) if os.path.isdir(d)]

            for d in dirs:
                for f in glob.glob(os.path.join(d, '*/events.out*')):
                    for e in tf.compat.v1.train.summary_iterator(f):
                        for v in e.summary.value:
                            if v.tag == baseline_to_tag[baseline] \
                                    and e.step == baseline_to_iteration[baseline]:
                                performance[task][baseline].append(
                                    tf.make_ndarray(v.tensor))

    final = [[list() for t in tasks] for m in metrics]
    table0_df = [[None for t in tasks] for b in baselines]
    table1_df = [[None for t in tasks] for b in baselines]

    for i, task in enumerate(tasks):
        for j, baseline in enumerate(baselines):

            data = np.array(performance[task][baseline])
            size = data.shape[0]
            np.random.shuffle(data)
            table0_data = data[:size // 2]
            table1_data = data[size // 2:]

            mean0 = 0.0
            if table0_data.shape[0] > 0:
                mean0 = np.mean(table0_data)

            mean1 = 0.0
            if table1_data.shape[0] > 0:
                mean1 = np.mean(table1_data)

            table0_df[j][i] = mean0
            table1_df[j][i] = mean1

    table0_df = pd.DataFrame(data=table0_df, columns=tasks, index=baselines)
    table1_df = pd.DataFrame(data=table1_df, columns=tasks, index=baselines)

    for i, task in enumerate(tasks):

        table0_rank = table0_df[task].to_numpy()
        table1_rank = table1_df[task].to_numpy()

        for j, metric in enumerate(metrics):

            if metric == "rank-correlation":
                rho = stats.spearmanr(table0_rank, table1_rank)[0]
                final[j][i].append(rho)

            elif metric == "max-shift":
                table0_index = table0_rank.argsort().argsort()
                table1_index = table1_rank.argsort().argsort()
                final[j][i].append(np.abs(table0_index - table1_index).max())

            elif metric == "avg-shift":
                table0_index = table0_rank.argsort().argsort()
                table1_index = table1_rank.argsort().argsort()
                final[j][i].append(np.abs(table0_index - table1_index).mean())

    for i, task in enumerate(tasks):
        for j, metric in enumerate(metrics):
            final[j][i] = np.mean(final[j][i])

    final = pd.DataFrame(data=final, columns=tasks, index=metrics)
    print(final.to_latex())
    final.to_csv(f"self-correlation{modifier}.csv")


@cli.command()
@click.option('--dir', type=str)
@click.option('--name', type=str)
@click.option('--tag', type=str)
@click.option('--max-iterations', type=int)
def plot_sample_size(dir,
                     name,
                     tag,
                     max_iterations):

    from collections import defaultdict
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    import os
    import re
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import tqdm
    import pickle as pkl

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

    dirs = [d for d in glob.glob(
        os.path.join(dir, '*'))
        if pattern.search(d) is not None]

    # get the hyper parameters for each experiment
    params = []
    for d in dirs:
        with open(os.path.join(d, 'params.pkl'), 'rb') as f:
            params.append(pkl.load(f))

    task_to_ylabel = {
        'HopperController-v0': "Effective sample size"}

    fig, axes = plt.subplots(
        nrows=1, ncols=1, figsize=(8.0, 8.0))

    task_to_axis = {
        'HopperController-v0': axes}

    sp_to_alpha = {
        10: 0.5,
        20: 1.0,
        30: 0.5,
        40: 0.0,
        50: 0.0,
        60: 0.0,
        70: 0.0,
        80: 0.0,
        90: 0.0,
        100: 0.0
    }

    for task in [
            'HopperController-v0']:

        # read data from tensor board
        ylabel = task_to_ylabel[task]
        data = pd.DataFrame(columns=[
            'Algorithm',
            'Importance sampling iteration',
            ylabel])

        sp_to_score = {}
        for d, p in tqdm.tqdm(zip(dirs, params)):

            if p["task_kwargs"]["split_percentile"] not in sp_to_score:
                sp_to_score[p["task_kwargs"]["split_percentile"]] = defaultdict(list)

            for f in glob.glob(os.path.join(d, '*/events.out*')):
                for e in tf.compat.v1.train.summary_iterator(f):
                    for v in e.summary.value:
                        if v.tag == tag and e.step < max_iterations:

                            score = tf.make_ndarray(v.tensor).tolist()
                            if p["task_kwargs"]["split_percentile"] == 20:
                                data = data.append({
                                    'Importance sampling iteration': e.step,
                                    'Algorithm': name,
                                    ylabel: score}, ignore_index=True)
                            else:
                                sp_to_score[p["task_kwargs"]["split_percentile"]][e.step].append(score)

        axis = task_to_axis[task]

        axis = sns.lineplot(
            x='Importance sampling iteration',
            y=ylabel,
            hue='Algorithm',
            data=data,
            ax=axis,
            linewidth=4,
            legend=False)

        for lr, plot_data in sp_to_score.items():
            xs = np.array(list(plot_data.keys()))
            ys = np.array([np.mean(l) for l in plot_data.values()])
            indices = np.argsort(xs)
            xs = xs[indices]
            ys = ys[indices]

            axis.plot(xs,
                      ys,
                      linestyle='--',
                      linewidth=2,
                      alpha=sp_to_alpha[lr],
                      color=color_palette[0])

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel(r'\textbf{Importance sampling iteration}', fontsize=24)
        axis.set_ylabel(r'\textbf{' + ylabel + '}', fontsize=24)
        axis.set_title(r'\textbf{' + task + '}', fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=2)

    plt.legend([r'\textbf{' + name.capitalize() + '}'],
               ncol=1,
               loc='lower left',
               fontsize=20,
               fancybox=True)
    plt.tight_layout()
    fig.savefig('plot_sample_size.pdf')


@cli.command()
@click.option('--dir1', type=str)
@click.option('--dir2', type=str)
@click.option('--name1', type=str)
@click.option('--name2', type=str)
@click.option('--tag', type=str)
@click.option('--max-iterations', type=int)
def plot_two_sweeps(dir1,
                    dir2,
                    name1,
                    name2,
                    tag,
                    max_iterations):

    from collections import defaultdict
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    import os
    import re
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import tqdm
    import pickle as pkl

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

    dir1 = [d for d in glob.glob(
        os.path.join(dir1, '*'))
        if pattern.search(d) is not None]
    dir2 = [d for d in glob.glob(
        os.path.join(dir2, '*'))
        if pattern.search(d) is not None]

    # get the hyper parameters for each experiment
    params1 = []
    for d in dir1:
        with open(os.path.join(d, 'params.pkl'), 'rb') as f:
            params1.append(pkl.load(f))

    # get the hyper parameters for each experiment
    params2 = []
    for d in dir2:
        with open(os.path.join(d, 'params.pkl'), 'rb') as f:
            params2.append(pkl.load(f))

    task_to_ylabel = {
        'HopperController-v0': "Average return"}

    fig, axes = plt.subplots(
        nrows=1, ncols=1, figsize=(8.0, 8.0))

    task_to_axis = {
        'HopperController-v0': axes}

    lr1_to_alpha = {
        0.1: 0.0,
        0.05: 0.0,
        0.02: 0.5,
        0.01: 1.0,
        0.005: 0.5,
        0.002: 0.0,
        0.001: 0.0,
        0.0005: 0.0
    }

    lr2_to_alpha = {
        0.00005: 0.0,
        0.00002: 0.0,
        0.00001: 0.0,
        0.000005: 0.0,
        0.000002: 0.5,
        0.000001: 1.0,
        0.0000005: 0.5,
        0.0000002: 0.0
    }

    for task in [
            'HopperController-v0']:

        # read data from tensor board
        ylabel = task_to_ylabel[task]
        data = pd.DataFrame(columns=[
            'Algorithm',
            'Gradient ascent steps',
            ylabel])

        lr1_to_score = {}
        for d, p in tqdm.tqdm(zip(dir1, params1)):

            if p["solver_lr"] not in lr1_to_score:
                lr1_to_score[p["solver_lr"]] = defaultdict(list)

            for f in glob.glob(os.path.join(d, '*/events.out*')):
                for e in tf.compat.v1.train.summary_iterator(f):
                    for v in e.summary.value:
                        if v.tag == tag and e.step < max_iterations:

                            score = tf.make_ndarray(v.tensor).tolist()
                            if p["solver_lr"] == 0.01:
                                data = data.append({
                                    'Gradient ascent steps': e.step,
                                    'Algorithm': name1,
                                    ylabel: score}, ignore_index=True)
                            else:
                                lr1_to_score[p["solver_lr"]][e.step].append(score)

        lr2_to_score = dict()
        for d, p in tqdm.tqdm(zip(dir2, params2)):

            if p["solver_lr"] not in lr2_to_score:
                lr2_to_score[p["solver_lr"]] = defaultdict(list)

            for f in glob.glob(os.path.join(d, '*/events.out*')):
                for e in tf.compat.v1.train.summary_iterator(f):
                    for v in e.summary.value:
                        if v.tag == tag and e.step < max_iterations:

                            score = tf.make_ndarray(v.tensor).tolist()
                            if p["solver_lr"] == 0.000001:
                                data = data.append({
                                    'Gradient ascent steps': e.step,
                                    'Algorithm': name2,
                                    ylabel: score}, ignore_index=True)
                            else:
                                lr2_to_score[p["solver_lr"]][e.step].append(score)

        axis = task_to_axis[task]

        axis = sns.lineplot(
            x='Gradient ascent steps',
            y=ylabel,
            hue='Algorithm',
            data=data,
            ax=axis,
            linewidth=4,
            legend=False)

        for lr, plot_data in lr1_to_score.items():
            xs = np.array(list(plot_data.keys()))
            ys = np.array([np.mean(l) for l in plot_data.values()])
            indices = np.argsort(xs)
            xs = xs[indices]
            ys = ys[indices]

            axis.plot(xs,
                      ys,
                      linestyle='--',
                      linewidth=2,
                      alpha=lr1_to_alpha[lr],
                      color=color_palette[0])

        for lr, plot_data in lr2_to_score.items():
            xs = np.array(list(plot_data.keys()))
            ys = np.array([np.mean(l) for l in plot_data.values()])
            indices = np.argsort(xs)
            xs = xs[indices]
            ys = ys[indices]

            axis.plot(xs,
                      ys,
                      linestyle='--',
                      linewidth=2,
                      alpha=lr2_to_alpha[lr],
                      color=color_palette[1])

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel(r'\textbf{Gradient ascent steps}', fontsize=24)
        axis.set_ylabel(r'\textbf{' + ylabel + '}', fontsize=24)
        axis.set_title(r'\textbf{' + task + '}', fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=2)

    plt.legend([r'\textbf{' + name1.capitalize() + '}',
                r'\textbf{' + name2.capitalize() + '}'],
               ncol=1,
               loc='lower left',
               fontsize=20,
               fancybox=True)
    plt.tight_layout()
    fig.savefig('plot_two_sweeps.pdf')


@cli.command()
@click.option('--dir1', type=str)
@click.option('--dir2', type=str)
@click.option('--name1', type=str)
@click.option('--name2', type=str)
@click.option('--tag', type=str)
@click.option('--max-iterations', type=int)
def plot_two_exp(dir1,
                 dir2,
                 name1,
                 name2,
                 tag,
                 max_iterations):

    from collections import defaultdict
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    import os
    import re
    import pandas as pd
    import numpy as np
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

    dir1 = [d for d in glob.glob(
        os.path.join(dir1, '*'))
        if pattern.search(d) is not None]
    dir2 = [d for d in glob.glob(
        os.path.join(dir2, '*'))
        if pattern.search(d) is not None]

    task_to_ylabel = {
        'HopperController-v0': "Average return"}

    fig, axes = plt.subplots(
        nrows=1, ncols=1, figsize=(8.0, 8.0))

    task_to_axis = {
        'HopperController-v0': axes}

    for task in [
            'HopperController-v0']:

        # read data from tensor board
        ylabel = task_to_ylabel[task]
        data = pd.DataFrame(columns=[
            'Algorithm',
            'Gradient ascent steps',
            ylabel])

        for d in tqdm.tqdm(dir1):
            for f in glob.glob(os.path.join(d, '*/events.out*')):
                for e in tf.compat.v1.train.summary_iterator(f):
                    for v in e.summary.value:
                        if v.tag == tag and e.step < max_iterations:

                            data = data.append({
                                'Algorithm': name1,
                                'Gradient ascent steps': e.step,
                                ylabel: tf.make_ndarray(v.tensor).tolist(),
                                }, ignore_index=True)

        for d in tqdm.tqdm(dir2):
            for f in glob.glob(os.path.join(d, '*/events.out*')):
                for e in tf.compat.v1.train.summary_iterator(f):
                    for v in e.summary.value:
                        if v.tag == tag and e.step < max_iterations:

                            data = data.append({
                                'Algorithm': name2,
                                'Gradient ascent steps': e.step,
                                ylabel: tf.make_ndarray(v.tensor).tolist(),
                                }, ignore_index=True)

        axis = task_to_axis[task]

        axis = sns.lineplot(
            x='Gradient ascent steps',
            y=ylabel,
            hue='Algorithm',
            data=data,
            ax=axis,
            linewidth=4,
            legend=False)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel(r'\textbf{Gradient ascent steps}', fontsize=24)
        axis.set_ylabel(r'\textbf{' + ylabel + '}', fontsize=24)
        axis.set_title(r'\textbf{' + task + '}', fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=2)

    plt.legend([r'\textbf{' + name1.capitalize() + '}',
                r'\textbf{' + name2.capitalize() + '}'],
               ncol=1,
               loc='upper center',
               fontsize=20,
               fancybox=True)
    plt.tight_layout()
    fig.savefig('plot_two_exp.pdf')


"""

design-baselines compare-runs \
    --hopper ~/neurips-round1/coms-hopper-demo/coms-hopper-cons/ \
    --hopper ~/neurips-round1/coms-hopper-demo/coms-hopper-over/ \
    --utr ~/neurips-round1/coms-utr-demo/coms-utr-cons/ \
    --utr ~/neurips-round1/coms-utr-demo/coms-utr-over/ \
    --names "COMs" \
    --names "Gradient Ascent" \
    --tag "score/100th" \
    --max-iterations 50

"""


@cli.command()
@click.option('--hopper', multiple=True)
@click.option('--utr', multiple=True)
@click.option('--names', multiple=True)
@click.option('--tag', type=str)
@click.option('--max-iterations', type=int)
def compare_runs(hopper,
                 utr,
                 names,
                 tag,
                 max_iterations):

    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    import os
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

    name_to_dir = {}

    for (hopper_i,
         utr_i,
         names_i) in zip(
            hopper,
            utr,
            names):

        hopper_dir = [d for d in glob.glob(
            os.path.join(hopper_i, '*')) if os.path.isdir(d)]
        utr_dir = [d for d in glob.glob(
            os.path.join(utr_i, '*')) if os.path.isdir(d)]

        name_to_dir[names_i] = {
            'hopper': hopper_dir, 'utr': utr_dir}

    task_to_ylabel = {
        'hopper': "Average Return",
        'utr': "Ribosome Loading"}

    task_to_title = {
        'hopper': "Hopper Controller",
        'utr': "UTR"}

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12.5, 5.0))

    task_to_axis = {'hopper': axes[0], 'utr': axes[1]}

    for task in ['hopper', 'utr']:
        title = task_to_title[task]

        # read data from tensor board
        ylabel = task_to_ylabel[task]
        data = pd.DataFrame(columns=[
            'Algorithm', 'Gradient ascent steps', ylabel])

        for name, task_to_dir_i in name_to_dir.items():
            for d in tqdm.tqdm(task_to_dir_i[task]):
                for f in glob.glob(os.path.join(d, 'events.out*')):
                    for e in tf.compat.v1.train.summary_iterator(f):
                        for v in e.summary.value:
                            if v.tag == tag and e.step < max_iterations:
                                data = data.append({
                                    'Algorithm': name,
                                    'Gradient ascent steps': e.step,
                                    ylabel: tf.make_ndarray(
                                        v.tensor).tolist()}, ignore_index=True)

        axis = task_to_axis[task]

        axis = sns.lineplot(
            x='Gradient ascent steps',
            y=ylabel,
            hue='Algorithm',
            data=data,
            ax=axis,
            linewidth=4,
            legend=False)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel(r'\textbf{Gradient ascent steps}', fontsize=24)
        axis.set_ylabel(r'\textbf{' + ylabel + '}', fontsize=24)
        axis.set_title(r'\textbf{' + title + '}', fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=2)

    new_axes = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    for x in name_to_dir.keys():
        new_axes.plot([0], [0], color=(1.0, 1.0, 1.0, 0.0), label=x)
    leg = new_axes.legend([r'\textbf{ ' + x + '}' for x in name_to_dir.keys()],
                          ncol=len(name_to_dir.keys()),
                          loc='lower center',
                          bbox_to_anchor=(0.5, 0.0, 0.0, 0.0),
                          fontsize=20,
                          fancybox=True)
    leg.legendHandles[0].set_color(color_palette[0])
    leg.legendHandles[0].set_linewidth(4.0)
    leg.legendHandles[1].set_color(color_palette[1])
    leg.legendHandles[1].set_linewidth(4.0)
    new_axes.patch.set_alpha(0.0)
    fig.subplots_adjust(bottom=0.3)
    fig.savefig('compare_runs.pdf')


@cli.command()
@click.option('--hopper')
@click.option('--superconductor')
@click.option('--tag', type=str)
@click.option('--max-iterations', type=int)
def ablate_beta(hopper,
                superconductor,
                tag,
                max_iterations):

    from collections import defaultdict
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    import os
    import re
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import tqdm
    import json

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

    hopper_dir = [d for d in glob.glob(
        os.path.join(hopper, '*'))
        if pattern.search(d) is not None]
    superconductor_dir = [d for d in glob.glob(
        os.path.join(superconductor, '*'))
        if pattern.search(d) is not None]

    name_to_dir = {
        'HopperController-v0': hopper_dir,
        'Superconductor-v0': superconductor_dir}

    task_to_ylabel = {
        'HopperController-v0': "Average return",
        'Superconductor-v0': "Critical temperature"}

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12.5, 5.0))

    task_to_axis = {
        'HopperController-v0': axes[0],
        'Superconductor-v0': axes[1]}

    for task in [
            'HopperController-v0',
            'Superconductor-v0']:

        # read data from tensor board
        ylabel = task_to_ylabel[task]
        data = pd.DataFrame(columns=[
            'Beta',
            'Gradient ascent steps',
            ylabel])

        for d in tqdm.tqdm(name_to_dir[task]):
            for f in glob.glob(os.path.join(d, '*/events.out*')):
                params = os.path.join(d, 'params.json')
                with open(params, "r") as pf:
                    params = json.load(pf)
                for e in tf.compat.v1.train.summary_iterator(f):
                    for v in e.summary.value:
                        if v.tag == tag and e.step < max_iterations:
                            data = data.append({
                                'Beta': f'{params["solver_beta"]}',
                                'Gradient ascent steps': e.step,
                                ylabel: tf.make_ndarray(v.tensor).tolist(),
                                }, ignore_index=True)

        axis = task_to_axis[task]

        palette = {"0.0": "C0", "0.1": "C1", "0.3": "C2",
                   "0.7": "C3", "0.9": "C4", "1.0": "C5"}

        axis = sns.lineplot(
            x='Gradient ascent steps',
            y=ylabel,
            hue='Beta',
            data=data,
            ax=axis,
            linewidth=4,
            legend=False,
            palette=palette)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel(r'\textbf{Gradient ascent steps}', fontsize=24)
        axis.set_ylabel(r'\textbf{' + ylabel + '}', fontsize=24)
        axis.set_title(r'\textbf{' + task + '}', fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=2)

    new_axes = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    for x in [0.0, 0.1, 0.3, 0.7, 0.9, 1.0]:
        new_axes.plot([0], [0], color=(1.0, 1.0, 1.0, 0.0), label=r"$\beta$" + f" = {x}")
    leg = new_axes.legend([r"$\beta$" + f" = {x}" for x in [0.0, 0.1, 0.3, 0.7, 0.9, 1.0]],
                          ncol=3,
                          loc='lower center',
                          bbox_to_anchor=(0.5, 0.0, 0.0, 0.0),
                          fontsize=20,
                          fancybox=True)
    leg.legendHandles[0].set_color(color_palette[0])
    leg.legendHandles[0].set_linewidth(4.0)
    leg.legendHandles[1].set_color(color_palette[1])
    leg.legendHandles[1].set_linewidth(4.0)
    leg.legendHandles[2].set_color(color_palette[2])
    leg.legendHandles[2].set_linewidth(4.0)
    leg.legendHandles[3].set_color(color_palette[3])
    leg.legendHandles[3].set_linewidth(4.0)
    leg.legendHandles[4].set_color(color_palette[4])
    leg.legendHandles[4].set_linewidth(4.0)
    leg.legendHandles[5].set_color(color_palette[5])
    leg.legendHandles[5].set_linewidth(4.0)
    new_axes.patch.set_alpha(0.0)
    fig.subplots_adjust(bottom=0.4)
    fig.savefig('ablate_beta.pdf')


@cli.command()
@click.option('--hopper')
@click.option('--utr')
@click.option('--tag', type=str)
@click.option('--max-iterations', type=int)
def ablate_tau(hopper,
               utr,
               tag,
               max_iterations):

    from collections import defaultdict
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    import os
    import re
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import tqdm
    import json

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

    hopper_dir = [d for d in glob.glob(
        os.path.join(hopper, '*'))
        if os.path.isdir(d)]
    utr_dir = [d for d in glob.glob(
        os.path.join(utr, '*'))
        if os.path.isdir(d)]

    name_to_dir = {
        'hopper': hopper_dir,
        'utr': utr_dir}

    task_to_ylabel = {
        'hopper': "Average Return",
        'utr': "Ribosome Loading"}

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12.5, 5.0))

    task_to_axis = {
        'hopper': axes[0],
        'utr': axes[1]}

    task_to_name = {
        'hopper': 'Hopper Controller',
        'utr': 'UTR'}

    for task in [
            'hopper',
            'utr']:

        # read data from tensor board
        ylabel = task_to_ylabel[task]
        data = pd.DataFrame(columns=[
            'Tau',
            'Gradient ascent steps',
            ylabel])

        for d in tqdm.tqdm(name_to_dir[task]):
            for f in glob.glob(os.path.join(d, '*/events.out*')):
                params = os.path.join(os.path.dirname(f), 'params.json')
                with open(params, "r") as pf:
                    params = json.load(pf)
                for e in tf.compat.v1.train.summary_iterator(f):
                    for v in e.summary.value:
                        if v.tag == tag and e.step < max_iterations:
                            data = data.append({
                                'Tau': f'{params["forward_model_overestimation_limit"]}',
                                'Gradient ascent steps': e.step,
                                ylabel: tf.make_ndarray(v.tensor).tolist(),
                                }, ignore_index=True)

        axis = task_to_axis[task]

        axis = sns.lineplot(
            x='Gradient ascent steps',
            y=ylabel,
            hue='Tau',
            data=data,
            ax=axis,
            linewidth=2)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel(r'\textbf{Gradient ascent steps}', fontsize=24)
        axis.set_ylabel(r'\textbf{' + ylabel + '}', fontsize=24)
        axis.set_title(r'\textbf{' + task_to_name[task] + '}', fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=2)

    fig.subplots_adjust(bottom=0.4)
    fig.savefig('ablate_tau.pdf')


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
--max-iterations 500

design-baselines ablate-architecture \
--hopper ~/final-results/online/gradient-ascent-hopper/gradient_ascent/ \
--superconductor ~/final-results/online/gradient-ascent-superconductor/gradient_ascent/ \
--gfp ~/final-results/online/gradient-ascent-gfp/gradient_ascent/ \
--molecule ~/final-results/online/gradient-ascent-molecule/gradient_ascent/ \
--tag 'oracle_0/prediction/mean' \
--evaluator-one 'oracle/min_of_mean/mean' \
--evaluator-two 'oracle/same_architecture/min_of_mean/mean' \
--max-iterations 500

"""


@cli.command()
@click.option('--hopper', type=str)
@click.option('--superconductor', type=str)
@click.option('--gfp', type=str)
@click.option('--molecule', type=str)
@click.option('--tag', type=str)
@click.option('--evaluator-one', type=str)
@click.option('--evaluator-two', type=str)
@click.option('--max-iterations', type=int)
def ablate_architecture(hopper,
                        superconductor,
                        gfp,
                        molecule,
                        tag,
                        evaluator_one,
                        evaluator_two,
                        max_iterations):

    from collections import defaultdict
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    import os
    import re
    import pandas as pd
    import numpy as np
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

    hopper_dir = [d for d in glob.glob(
        os.path.join(hopper, '*'))
        if pattern.search(d) is not None]
    superconductor_dir = [d for d in glob.glob(
        os.path.join(superconductor, '*'))
        if pattern.search(d) is not None]
    gfp_dir = [d for d in glob.glob(
        os.path.join(gfp, '*'))
        if pattern.search(d) is not None]
    molecule_dir = [d for d in glob.glob(
        os.path.join(molecule, '*'))
        if pattern.search(d) is not None]

    task_to_dir = {
        'HopperController-v0': hopper_dir,
        'Superconductor-v0': superconductor_dir,
        'GFP-v0': gfp_dir,
        'MoleculeActivity-v0': molecule_dir}

    task_to_ylabel = {
        'HopperController-v0': "Predicted return",
        'Superconductor-v0': "Predicted temperature",
        'GFP-v0': "Predicted fluorescence",
        'MoleculeActivity-v0': "Predicted activity"}

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
            'Gradient ascent steps',
            ylabel])

        it_to_eval_one = defaultdict(list)
        it_to_eval_two = defaultdict(list)

        for d in tqdm.tqdm(task_to_dir[task]):
            for f in glob.glob(os.path.join(d, '*/events.out*')):
                for e in tf.compat.v1.train.summary_iterator(f):
                    for v in e.summary.value:
                        if v.tag == tag and e.step < max_iterations:

                            data = data.append({
                                'Gradient ascent steps': e.step,
                                ylabel: tf.make_ndarray(v.tensor).tolist(),
                                }, ignore_index=True)

                        if v.tag == evaluator_one and e.step < max_iterations:
                            it_to_eval_one[e.step].append(
                                tf.make_ndarray(v.tensor).tolist())

                        if v.tag == evaluator_two and e.step < max_iterations:
                            it_to_eval_two[e.step].append(
                                tf.make_ndarray(v.tensor).tolist())

        if len(it_to_eval_one.keys()) == 0:
            print(task, 'A')
            exit()
        if len(it_to_eval_two.keys()) == 0:
            print(task, 'B')
            exit()

        axis = task_to_axis[task]

        axis = sns.lineplot(
            x='Gradient ascent steps',
            y=ylabel,
            data=data,
            ax=axis,
            linewidth=4,
            legend=False)

        original_data = data[ylabel].to_numpy()
        y_min = original_data.min()
        y_max = original_data.max()

        xs, ys = zip(*it_to_eval_one.items())
        xs = np.array(xs)
        ys = np.array([np.mean(yi) for yi in ys])

        ys = (ys - ys.min()) / (ys.max() - ys.min())
        ys = ys * (y_max - y_min) + y_min

        indices = np.argsort(xs)
        xs = xs[indices]
        ys = ys[indices]

        axis.plot(xs,
                  ys,
                  linestyle='--',
                  linewidth=4,
                  color=color_palette[0])

        xs, ys = zip(*it_to_eval_two.items())
        xs = np.array(xs)
        ys = np.array([np.mean(yi) for yi in ys])

        ys = (ys - ys.min()) / (ys.max() - ys.min())
        ys = ys * (y_max - y_min) / 2 + y_min

        indices = np.argsort(xs)
        xs = xs[indices]
        ys = ys[indices]

        axis.plot(xs,
                  ys,
                  linestyle='--',
                  linewidth=4,
                  color=color_palette[1])

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel(r'\textbf{Gradient ascent steps}', fontsize=24)
        axis.set_ylabel(r'\textbf{' + ylabel + '}', fontsize=24)
        axis.set_title(r'\textbf{' + task + '}', fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=2)

    plt.legend([r'\textbf{Training Prediction}',
                r'\textbf{Naive Ensemble}',
                r'\textbf{Varying Activations}'],
               ncol=4,
               loc='lower center',
               bbox_to_anchor=(-1.4, -0.5),
               fontsize=20,
               fancybox=True)
    fig.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    fig.savefig('ablate_architecture.pdf')


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--xlabel', type=str)
@click.option('--ylabel', type=str)
@click.option('--separate-runs', is_flag=True)
@click.option('--max-iterations', type=int, default=999999)
@click.option('--lower-limit', type=float, default=-999999.)
@click.option('--upper-limit', type=float, default=999999.)
@click.option('--norm', type=str, default='none')
def plot(dir, tag, xlabel, ylabel, separate_runs,
         max_iterations, lower_limit, upper_limit, norm):

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
    import numpy as np

    plt.rcParams['text.usetex'] = False
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

    import design_bench
    params[0]['task_kwargs'].pop('for_validation', None)
    task = design_bench.make(params[0]['task'],
                             **params[0]['task_kwargs'])
    dim_x = float(task.x.shape[1])

    # read data from tensor board
    data = pd.DataFrame(columns=['id', xlabel, ylabel] + params_of_variation)
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag and e.step < max_iterations:
                        y_vals = tf.make_ndarray(v.tensor)
                        y_vals = np.clip(y_vals, lower_limit, upper_limit)
                        if norm == 'sqrt':
                            y_vals /= np.sqrt(dim_x)
                        if norm == 'full':
                            y_vals /= dim_x
                        row = {'id': i,
                               ylabel: y_vals.tolist(),
                               xlabel: e.step}
                        for key in params_of_variation:
                            row[key] = f'{pretty(key)} = {p[key]}'
                        data = data.append(row, ignore_index=True)

    if separate_runs:
        params_of_variation.append('id')

    # save a separate plot for every hyper parameter
    print(data)
    for key in params_of_variation:
        plt.clf()
        g = sns.relplot(x=xlabel, y=ylabel, hue=key, data=data,
                        kind="line", height=5, aspect=2,
                        facet_kws={"legend_out": True})
        g.set(title=f'Evaluating {pretty(algo_name)} On {task_name}')
        plt.savefig(f'{algo_name}_{task_name}_{key}_{tag.replace("/", "_")}_{norm}.png',
                    bbox_inches='tight')


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--xlabel', type=str)
@click.option('--ylabel', type=str)
@click.option('--cbar-label', type=str)
@click.option('--iteration', type=int, default=999999)
@click.option('--lower-limit', type=float, default=-999999.)
@click.option('--upper-limit', type=float, default=999999.)
def plot_heatmap(dir, tag, xlabel, ylabel, cbar_label,
                 iteration, lower_limit, upper_limit):

    from collections import defaultdict
    import glob
    import os
    import re
    import pickle as pkl
    import tensorflow as tf
    import tqdm
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

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
    while "evaluate_steps" in params_of_variation:
        params_of_variation.remove("evaluate_steps")

    assert len(params_of_variation) == 2, \
        f"only two parameters can vary: {params_of_variation}"

    # read data from tensor board
    data_dict = defaultdict(list)
    p0_keys = set()
    p1_keys = set()
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag and e.step == iteration:
                        y_vals = tf.make_ndarray(v.tensor)
                        y_vals = np.clip(y_vals, lower_limit, upper_limit)
                        p0_keys.add(p[params_of_variation[0]])
                        p1_keys.add(p[params_of_variation[1]])
                        data_dict[(
                            p[params_of_variation[0]],
                            p[params_of_variation[1]])].append(y_vals)

    p0_keys = sorted(list(p0_keys))
    p0_map = {p0: i for i, p0 in enumerate(p0_keys)}
    p1_keys = sorted(list(p1_keys))
    p1_map = {p1: i for i, p1 in enumerate(p1_keys)}
    print(p0_keys, p1_keys)

    data = np.zeros([len(p1_keys), len(p0_keys)])
    for p0 in p0_keys:
        for p1 in p1_keys:
            data[p1_map[p1], p0_map[p0]] = np.mean(data_dict[(p0, p1)])
            print(f"{(p0, p1)} = {np.mean(data_dict[(p0, p1)])}")

    # save a separate plot for every hyper parameter
    plt.clf()

    g = sns.heatmap(data,
                    xticklabels=[r"$\infty$" if x == 10.0 else f"{x}" for x in p0_keys],
                    yticklabels=[r"$\infty$" if x == 10.0 else f"{x}" for x in p1_keys],
                    square=True,
                    cbar_kws={'label': r"$\textbf{" + cbar_label + r"}$"})
    plt.title(r"$\textbf{" + task_name + r"}$")
    plt.xlabel(r"$\textbf{" + xlabel + r"}$")
    plt.ylabel(r"$\textbf{" + ylabel + r"}$")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(f'{algo_name}_{task_name}_{tag.replace("/", "_")}_heatmap.pdf',
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
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
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
@click.option('--hopper', type=str)
@click.option('--tag', type=str)
@click.option('--param', type=str)
@click.option('--xlabel', type=str)
@click.option('--iteration', type=int)
def plot_comparison(hopper, tag, param, xlabel, iteration):

    from collections import defaultdict
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
    import matplotlib

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

    hopper_dir = [d for d in glob.glob(
        os.path.join(hopper, '*'))
        if pattern.search(d) is not None]

    task_to_dir = {
        'HopperController-v0': hopper_dir}

    task_to_ylabel = {
        'HopperController-v0': "Average return"}

    fig, axes = plt.subplots(
        nrows=1, ncols=1, figsize=(7, 5.0))

    task_to_axis = {
        'HopperController-v0': axes}

    for task in [
            'HopperController-v0']:

        ylabel = task_to_ylabel[task]
        dirs = task_to_dir[task]

        # get the hyper parameters for each experiment
        params = []
        for d in dirs:
            with open(os.path.join(d, 'params.pkl'), 'rb') as f:
                params.append(pkl.load(f))

        # read data from tensor board
        data = pd.DataFrame(columns=[xlabel, ylabel])
        it_to_tag = defaultdict(list)
        it_to_p = defaultdict(list)
        for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
            for f in glob.glob(os.path.join(d, '*/events.out*')):
                for e in tf.compat.v1.train.summary_iterator(f):
                    for v in e.summary.value:
                        if v.tag == tag:
                            it_to_tag[e.step].append(
                                tf.make_ndarray(v.tensor).tolist())
                            it_to_p[e.step].append(p[param])

        for score, p in zip(it_to_tag[iteration], it_to_p[iteration]):
            data = data.append({
                ylabel: score,
                xlabel: p}, ignore_index=True)

        axis = task_to_axis[task]
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        sns.lineplot(x=xlabel,
                     y=ylabel,
                     data=data,
                     ax=axis,
                     linewidth=4,
                     legend=False)
        axis.set_xlabel(r'\textbf{' + xlabel + '}', fontsize=24)
        axis.set_ylabel(r'\textbf{' + ylabel + '}', fontsize=24)
        axis.set_xscale('log')
        axis.set_title(r'\textbf{' + task + '}', fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=2)

    fig.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    fig.savefig('ablate_tau.pdf')


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--eval-tag', type=str)
def evaluate_offline_per_seed(dir, tag, eval_tag):

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

    scores = []
    its = []
    tag_set = set()

    # read data from tensor board
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        it_to_tag = defaultdict(list)
        it_to_eval_tag = defaultdict(list)
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag not in tag_set:
                        tag_set.add(v.tag)
                    if v.tag == tag and e.step < 500:
                        it_to_tag[e.step].append(tf.make_ndarray(v.tensor).tolist())
                    if v.tag == eval_tag and e.step < 500:
                        it_to_eval_tag[e.step].append(tf.make_ndarray(v.tensor).tolist())

        if len(it_to_eval_tag.keys()) > 0:
            keys, values = zip(*it_to_eval_tag.items())
            values = [np.mean(vs) for vs in values]
            iteration = keys[int(np.argmax(values))]
            scores.append(it_to_tag[iteration])
            its.append(iteration)

    if len(scores) == 0:
        print(dir, tag, eval_tag, tag_set)
        exit()
    mean = np.mean(scores)
    std = np.std(scores)

    # save a separate plot for every hyper parameter
    print(f'Evaluate {task_name} At {np.mean(its)}\n\t{mean} +- {std}')


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
@click.option('--confidence', is_flag=True)
def evaluate_fixed(dir, tag, iteration, confidence):

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

    import numpy as np
    import scipy.stats

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        se = scipy.stats.sem(a)
        return se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

    if iteration in it_to_tag:
        mean = np.mean(it_to_tag[iteration])
        std = np.std(it_to_tag[iteration])
        if confidence:
            ci90 = mean_confidence_interval(np.array(it_to_tag[iteration]), confidence=0.90)
            ci95 = mean_confidence_interval(np.array(it_to_tag[iteration]), confidence=0.95)
            ci99 = mean_confidence_interval(np.array(it_to_tag[iteration]), confidence=0.99)
            #print(f'Evaluate {task_name} At {iteration}\n\t{mean} : ci90={ci90} ci95={ci95} ci99={ci99}')
            print(f'{task_name}, {algo_name}, {mean}, {ci90}, {ci95}, {ci99}')
        else:
            #print(f'Evaluate {task_name} At {iteration}\n\t{mean} +- {std}')
            print(f'{task_name}, {algo_name}, {mean}, {std}')


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--distance', type=float)
@click.option('--distance-tag', type=str, default='distance/travelled')
@click.option('--norm', type=str, default='full')
@click.option('--confidence', is_flag=True)
def evaluate_fixed_distance(dir, tag, distance,
                            distance_tag, norm, confidence):

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

    import design_bench
    task = design_bench.make(params[0]['task'],
                             **params[0]['task_kwargs'])
    dim_x = float(task.x.shape[1])

    # read data from tensor board
    it_to_tag = defaultdict(list)
    it_to_distance = defaultdict(list)
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag:
                        it_to_tag[e.step].append(
                            tf.make_ndarray(v.tensor).tolist())
                    if v.tag == distance_tag:
                        y_vals = tf.make_ndarray(v.tensor)
                        if norm == 'sqrt':
                            y_vals /= np.sqrt(dim_x)
                        if norm == 'full':
                            y_vals /= dim_x
                        it_to_distance[e.step].append(y_vals.tolist())

    iterations, distances = zip(*list(it_to_distance.items()))
    distances = np.array([np.mean(dl) for dl in distances])
    distances = np.where(distances < distance, distances, -999999.)
    iteration = iterations[np.argmax(distances)]

    import numpy as np
    import scipy.stats

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        se = scipy.stats.sem(a)
        return se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

    if iteration in it_to_tag:
        mean = np.mean(it_to_tag[iteration])
        std = np.std(it_to_tag[iteration])
        if confidence:
            ci90 = mean_confidence_interval(np.array(it_to_tag[iteration]), confidence=0.90)
            ci95 = mean_confidence_interval(np.array(it_to_tag[iteration]), confidence=0.95)
            ci99 = mean_confidence_interval(np.array(it_to_tag[iteration]), confidence=0.99)
            #print(f'Evaluate {task_name} At {iteration}\n\t{mean} : ci90={ci90} ci95={ci95} ci99={ci99}')
            print(f'{task_name}, {algo_name}, {mean}, {ci90}, {ci95}, {ci99}')
        else:
            #print(f'Evaluate {task_name} At {iteration}\n\t{mean} +- {std}')
            print(f'{task_name}, {algo_name}, {mean}, {std}')


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


@cli.command()
@click.option('--dir', type=str)
@click.option('--tag', type=str)
@click.option('--distance', type=float)
@click.option('--distance-tag', type=str, default='distance/travelled')
@click.option('--norm', type=str, default='full')
def evaluate_distance(dir, tag, distance, distance_tag, norm):

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

    import design_bench
    task = design_bench.make(params[0]['task'],
                             **params[0]['task_kwargs'])
    dim_x = float(task.x.shape[1])

    # read data from tensor board
    param_to_it_scores = defaultdict(lambda: defaultdict(list))
    param_to_it_distances = defaultdict(lambda: defaultdict(list))
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag:
                        for key in params_of_variation:
                            key = f'{pretty(key)} = {p[key]}'
                            param_to_it_scores[key][e.step].append(
                                tf.make_ndarray(v.tensor).tolist())
                    if v.tag == distance_tag:
                        for key in params_of_variation:
                            key = f'{pretty(key)} = {p[key]}'
                            ds = tf.make_ndarray(v.tensor)
                            if norm == 'sqrt':
                                ds /= np.sqrt(dim_x)
                            if norm == 'full':
                                ds /= dim_x
                            param_to_it_distances[key][e.step].append(ds)

    # return the mean score and standard deviation
    for key in param_to_it_scores:
        step_0 = list(param_to_it_scores[key].keys())[0]
        if len(param_to_it_scores[key][step_0]) > 0:
            iterations, distances = zip(*list(param_to_it_distances[key].items()))
            distances = np.array([np.mean(dl) for dl in distances])
            distances = np.where(distances < distance, distances, -999999.)
            iteration = iterations[np.argmax(distances)]
            scores = np.array(param_to_it_scores[key][iteration])
            mean = np.mean(scores)
            std = np.std(scores - mean)
            print(f"key: {key}\n\tmean: {mean}\n\tstd: {std}")


@cli.command()
@click.option('--dir', type=str)
@click.option('--iteration', type=int)
@click.option('--lower-k', type=int, default=1)
@click.option('--upper-k', type=int, default=128)
def evaluate_budget(dir, iteration, lower_k, upper_k):

    from collections import defaultdict
    import pickle as pkl
    import glob
    import os
    import re
    import numpy as np
    import tensorflow as tf
    import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

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

    # get the task and algorithm name
    task_name = params[0]['task']
    algo_name = matches[0].group(1)

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
    data = pd.DataFrame(columns=['id', "Budget", "Score"] + params_of_variation)
    for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for key in params_of_variation:

                try:

                    scores = np.load(os.path.join(os.path.dirname(f), 'scores.npy'))
                    predictions = np.load(os.path.join(os.path.dirname(f), 'predictions.npy'))
                    if len(predictions.shape) > 2:
                        predictions = predictions[:, :, 0]
                    print(predictions.shape)
                    print(scores.shape)
                    for limit in range(lower_k, upper_k):
                        top_k = np.argsort(predictions[:, iteration])[::-1][:limit]
                        data = data.append({"id": i, "Budget": limit,
                                            "Score": np.max(scores[:, iteration][top_k]),
                                            key: f'{pretty(key)} = {p[key]}'}, ignore_index=True)

                except FileNotFoundError:
                    pass

    # save a separate plot for every hyper parameter
    for key in params_of_variation:
        plt.clf()
        g = sns.relplot(x="Budget", y="Score", hue=key, data=data,
                        kind="line", height=5, aspect=2,
                        facet_kws={"legend_out": True})
        g.set(title=f'Stability Of {pretty(algo_name)} On {task_name}')
        plt.savefig(f'{algo_name}_{task_name}_{key}_stability.png',
                    bbox_inches='tight')

"""

design-baselines compare-budget --hopper ~/grad-kun-final/hopper/gradient_ascent/ --hopper ~/coms-kun-icml/online-hopper-particle/online/ --superconductor ~/grad-kun-final/superconductor/gradient_ascent/ --superconductor ~/coms-kun-icml/online-superconductor-particle/online/ --names 'naive gradient ascent' --names 'coms (ours)' --iteration 450

"""


@cli.command()
@click.option('--hopper', multiple=True)
@click.option('--utr', multiple=True)
@click.option('--names', multiple=True)
@click.option('--iteration', type=int)
@click.option('--upper-k', type=int, default=128)
def compare_budget(hopper,
                   utr,
                   names,
                   iteration,
                   upper_k):

    from collections import defaultdict
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    import os
    import re
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import tqdm
    import json

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

    name_to_dir = {}

    for (hopper_i,
         utr_i,
         names_i) in zip(
            hopper,
            utr,
            names):

        hopper_dir = [d for d in glob.glob(
            os.path.join(hopper_i, '*'))
            if os.path.isdir(d)]
        utr_dir = [d for d in glob.glob(
            os.path.join(utr_i, '*'))
            if os.path.isdir(d)]

        name_to_dir[names_i] = {
            'Hopper Controller': hopper_dir,
            'UTR': utr_dir}

    task_to_ylabel = {
        'Hopper Controller': "Average Return",
        'UTR': "Ribosome Loading"}

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12.5, 5.0))

    task_to_axis = {
        'Hopper Controller': axes[0],
        'UTR': axes[1]}

    for task in [
            'Hopper Controller',
            'UTR']:

        # read data from tensor board
        ylabel = task_to_ylabel[task]
        data = pd.DataFrame(columns=[
            'Algorithm',
            'Budget',
            ylabel])

        for name, task_to_dir_i in name_to_dir.items():
            for d in tqdm.tqdm(task_to_dir_i[task]):
                for f in glob.glob(os.path.join(d, 'events.out*')):

                    try:

                        scores = np.load(os.path.join(os.path.dirname(f), 'scores.npy'))
                        predictions = np.load(os.path.join(os.path.dirname(f), 'predictions.npy'))

                        if len(predictions.shape) > 2:
                            predictions = predictions[:, :, 0]

                        for limit in range(1, upper_k):
                            top_k = np.argsort(predictions[:, iteration])[::-1][:limit]
                            data = data.append({"Budget": limit,
                                                ylabel: np.max(scores[:, iteration][top_k]),
                                                'Algorithm': name}, ignore_index=True)

                    except FileNotFoundError:
                        pass

        axis = task_to_axis[task]

        axis = sns.lineplot(
            x='Budget',
            y=ylabel,
            hue='Algorithm',
            data=data,
            ax=axis,
            linewidth=4,
            legend=False)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel(r'\textbf{Evaluation budget}', fontsize=24)
        axis.set_ylabel(r'\textbf{' + ylabel + '}', fontsize=24)
        axis.set_title(r'\textbf{' + task + '}', fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=2)

    new_axes = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    for x in name_to_dir.keys():
        new_axes.plot([0], [0], color=(1.0, 1.0, 1.0, 0.0), label=x)
    leg = new_axes.legend([r'\textbf{ ' + x + '}' for x in name_to_dir.keys()],
                          ncol=len(name_to_dir.keys()),
                          loc='lower center',
                          bbox_to_anchor=(0.5, 0.0, 0.0, 0.0),
                          fontsize=20,
                          fancybox=True)
    leg.legendHandles[0].set_color(color_palette[0])
    leg.legendHandles[0].set_linewidth(4.0)
    leg.legendHandles[1].set_color(color_palette[1])
    leg.legendHandles[1].set_linewidth(4.0)
    new_axes.patch.set_alpha(0.0)
    fig.subplots_adjust(bottom=0.3)
    fig.savefig('compare_budget.pdf')
