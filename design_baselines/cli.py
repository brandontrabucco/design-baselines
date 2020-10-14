import click


@click.group()
def cli():
    """A group of potential sub methods that are available for use through
    a command line interface
    """


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

design-baselines ablate-architecture \
--hopper ~/final-results/online/gradient-ascent-hopper/gradient_ascent/ \
--superconductor ~/final-results/online/gradient-ascent-superconductor/gradient_ascent/ \
--gfp ~/final-results/online/gradient-ascent-gfp/gradient_ascent/ \
--molecule ~/final-results/online/gradient-ascent-molecule/gradient_ascent/ \
--tag 'score/100th' \
--tag 'score/100th' \
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
@click.option('--dashed-tag', type=str)
@click.option('--max-iterations', type=int)
def compare_runs(hopper,
                 superconductor,
                 gfp,
                 molecule,
                 names,
                 tag,
                 dashed_tag,
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
        'HopperController-v0': "Average return",
        'Superconductor-v0': "Critical temperature",
        'GFP-v0': "Protein fluorescence",
        'MoleculeActivity-v0': "Drug activity"}

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

        name_to_eval = dict()

        for name, task_to_dir_i in name_to_dir.items():
            it_to_eval_tag = defaultdict(list)

            for d in tqdm.tqdm(task_to_dir_i[task]):
                for f in glob.glob(os.path.join(d, '*/events.out*')):
                    for e in tf.compat.v1.train.summary_iterator(f):
                        for v in e.summary.value:
                            if v.tag == tag and e.step < max_iterations:

                                data = data.append({
                                    'Algorithm': name,
                                    'Gradient ascent steps': e.step,
                                    ylabel: tf.make_ndarray(v.tensor).tolist(),
                                    }, ignore_index=True)

                            if v.tag == dashed_tag and e.step < max_iterations:
                                it_to_eval_tag[e.step].append(
                                    tf.make_ndarray(v.tensor).tolist())

            name_to_eval[name] = it_to_eval_tag

        axis = task_to_axis[task]

        axis = sns.lineplot(
            x='Gradient ascent steps',
            y=ylabel,
            hue='Algorithm',
            data=data,
            ax=axis,
            linewidth=4,
            legend=False)

        iteration = -1
        for name, it_to_eval_tag in name_to_eval.items():
            iteration += 1
            original_data = data.loc[data['Algorithm']
                                     == name][ylabel].to_numpy()
            y_min = original_data.min()
            y_max = original_data.max()

            xs, ys = zip(*it_to_eval_tag.items())
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
                      color=color_palette[iteration])

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

    plt.legend([r'\textbf{' + x.capitalize() + '}' for x in name_to_dir.keys()] +
               [r'\textbf{Prediction ' + x.lower() + '}' for x in name_to_eval.keys()],
               ncol=len(name_to_dir.keys()) + len(name_to_eval.keys()),
               loc='lower center',
               bbox_to_anchor=(-1.4, -0.5),
               fontsize=20,
               fancybox=True)
    fig.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    fig.savefig('compare_runs.pdf')


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
@click.option('--superconductor', type=str)
@click.option('--molecule', type=str)
@click.option('--tag', type=str)
@click.option('--param', type=str)
@click.option('--xlabel', type=str)
@click.option('--eval-tag', type=str)
def plot_comparison(superconductor, molecule, tag, param, xlabel, eval_tag):

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

    superconductor_dir = [d for d in glob.glob(
        os.path.join(superconductor, '*'))
        if pattern.search(d) is not None]
    molecule_dir = [d for d in glob.glob(
        os.path.join(molecule, '*'))
        if pattern.search(d) is not None]

    task_to_dir = {
        'Superconductor-v0': superconductor_dir,
        'MoleculeActivity-v0': molecule_dir}

    task_to_ylabel = {
        'Superconductor-v0': "Critical temperature",
        'MoleculeActivity-v0': "Drug activity"}

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12.5, 5.0))

    task_to_axis = {
        'Superconductor-v0': axes[0],
        'MoleculeActivity-v0': axes[1]}

    for task in [
            'Superconductor-v0',
            'MoleculeActivity-v0']:

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
        it_to_eval_tag = defaultdict(list)
        it_to_p = defaultdict(list)
        for i, (d, p) in enumerate(tqdm.tqdm(zip(dirs, params))):
            for f in glob.glob(os.path.join(d, '*/events.out*')):
                for e in tf.compat.v1.train.summary_iterator(f):
                    for v in e.summary.value:
                        if v.tag == tag:
                            it_to_tag[e.step].append(
                                tf.make_ndarray(v.tensor).tolist())
                            it_to_p[e.step].append(p[param])
                        if v.tag == eval_tag:
                            it_to_eval_tag[e.step].append(
                                tf.make_ndarray(v.tensor).tolist())

        if len(it_to_eval_tag) > 0:
            eval_position = int(np.argmax(
                [np.mean(vals) for vals in it_to_eval_tag.values()]))
            iteration = list(it_to_eval_tag.keys())[eval_position]
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
        axis.set_title(r'\textbf{' + task + '}', fontsize=24)
        axis.grid(color='grey',
                  linestyle='dotted',
                  linewidth=2)

    fig.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    fig.savefig('plot_comparison.pdf')


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
