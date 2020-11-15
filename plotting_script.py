import matplotlib

matplotlib.use('Agg')
import argparse

import matplotlib.pyplot as plt
import os

import numpy as np

import seaborn as sns

sns.set_style('white')

# Plotting utilities specific to diagnosing_q
import scripts.plot_utils as plot_utils
from rlutil.logging import log_processor
import collections
import scipy.stats


def configure_matplotlib(matplotlib):
    plt.rcParams['text.usetex'] = False

    matplotlib.rc('font', family='serif', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    # matplotlib.rcParams['font.weight']= 'heavy'

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 13}

    matplotlib.rc('font', **font)


configure_matplotlib(matplotlib)

color_palette = ['#EE7733', '#0077BB', '#33BBEE', '#009988', '#CC3311', '#EE3377', '#BBBBBB', '#000000']

palette = sns.color_palette(color_palette)
sns.palplot(palette)
sns.set_palette(palette)

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default="./data", type=str)
parser.add_argument('--env_name', default='grid16smoothobs', type=str)
parser.add_argument('--layer_size', default='(64, 64)', type=str)
parser.add_argument('--weighting_scheme', default='none', type=str)
parser.add_argument('--sampling_type', default='pi', type=str)
args = parser.parse_args()

import json, os


############################################
#   PLOTTING SPECIFIC FUNCTIONS
############################################
def make_paper_ready(ax, labelsize='medium'):
    ax.yaxis.set_tick_params(labelsize=labelsize)
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


##############################################
#   DIAGNOSING_Q SPECIFIC FUNCTIONS
##############################################

def reduce_mean_partitions_over_time(partitions, split_key, expected_len=600):
    for partition_key in partitions:
        if isinstance(split_key, (list, tuple)):
            new_params = dict(zip(split_key, partition_key))
        else:
            new_params = {split_key: partition_key}

        exps = partitions[partition_key]
        new_data = {}
        for data_key in exps[0].progress.keys():
            agg_data = [exp.progress[data_key][:expected_len] for exp in exps]
            filtered_agg_data = agg_data
            new_data[data_key] = scipy.stats.trim_mean(filtered_agg_data, 0.1, axis=0)
        yield log_processor.ExperimentLog(new_params, new_data, None)


def reduce_std_partitions_over_time(partitions, split_key, expected_len=600):
    for partition_key in partitions:
        if isinstance(split_key, (list, tuple)):
            new_params = dict(zip(split_key, partition_key))
        else:
            new_params = {split_key: partition_key}

        exps = partitions[partition_key]
        new_data = {}
        for data_key in exps[0].progress.keys():
            agg_data = [exp.progress[data_key][:expected_len] for exp in exps]
            filtered_agg_data = agg_data
            new_data[data_key] = np.std(filtered_agg_data, axis=0)
        yield log_processor.ExperimentLog(new_params, new_data, None)


def filter_fn(params):
    if params['log_sampling_type'] != 'buffer128':
        return False
    if params['env_name'] != args.env_name:
        return False
    if params['weighting_scheme'] != args.weighting_scheme:
        return False
    if params['sampling_policy'] != args.sampling_type:
        return False
    if params['layers'] != args.layer_size:
        return False
    return True


def main(log_dir):
    all_exps = list(plot_utils.max_itr(log_processor.iterate_experiments(log_dir, filter_fn=filter_fn), n=20000))
    plot(all_exps)


def plot(all_exps):
    frame = log_processor.to_data_frame(all_exps)

    # Mean across weighting scheme, layers, env_name
    frame = log_processor.reduce_mean_keys(frame, col_keys=('max_project_steps', 'layers',))

    # mean datasframe
    split_exps = log_processor.partition_params(all_exps, ('max_project_steps', 'layers'))
    mean_exps = list(reduce_mean_partitions_over_time(split_exps, ('max_project_steps', 'layers'), expected_len=500))
    std_exps = list(reduce_std_partitions_over_time(split_exps, ('max_project_steps', 'layers'), expected_len=500))

    # ALso split them across axis to make individual lines
    split_exps = log_processor.partition_params(all_exps, ('max_project_steps', 'layers', 'uuid'))
    exps = list(reduce_mean_partitions_over_time(split_exps, ('max_project_steps', 'layers', 'uuid'), expected_len=500))

    frame = log_processor.timewise_data_frame(exps, time_min=0, time_max=500, ignore_params=('__clsname__',))
    mean_df = log_processor.timewise_data_frame(mean_exps, time_min=0, time_max=500)
    std_df = log_processor.timewise_data_frame(std_exps, time_min=0, time_max=500)

    # import ipdb; ipdb.set_trace()
    unique_max_steps = frame['max_project_steps'].unique()
    unique_max_steps = [unique_max_steps.min(), unique_max_steps.max()]

    # plt.figure(figsize=(10,4))
    fig, axes = plt.subplots(nrows=2, figsize=(5.0, 8.0))

    colors = ['#CC3311', '#0077BB', '#EE7733', ]
    plt_keys = ['actual rank 0.01', 'returns_normalized_50_step_mean']

    y_ax_dict = {
        'actual rank 0.01': r'srank$_\delta(\Phi), \delta=0.01$',
        'stable rank': r'Lower bound on rank$(\Phi)$',
        'returns_normalized_50_step_mean': r'Normalized Return',
    }

    axes[0].plot(
        np.arange(500), np.ones(500) * mean_df['[proj] actual rank 0.01'].mean(), color='#000000', linewidth=3,
        linestyle='--', label='Supervised'
    )
    # axes[1].plot(
    #     np.arange(500), np.ones(500) * mean_df['[proj] stable rank'].mean(), color='#000000', linewidth=3, linestyle='--', label='Supervised'
    # )
    axes[1].plot(
        np.arange(500), np.ones(500), color='#000000', linewidth=3, linestyle='--', label='Supervised'
    )

    if '64' in args.layer_size:
        ranges = [(10, 64), (0.0, 1.0)]
    else:
        ranges = [(10, 150), (0.0, 1.0)]

    for mdx in range(2):
        plt_key = plt_keys[mdx]
        ax = axes[mdx]
        ax.grid(color='grey', linestyle='dotted', linewidth=1)
        for idx, max_step_idx in enumerate(unique_max_steps):
            print(max_step_idx)
            temp_frame = (mean_df['max_project_steps'] == max_step_idx)
            mean_val = mean_df[temp_frame]

            temp_frame = (std_df['max_project_steps'] == max_step_idx)
            std_val = std_df[temp_frame]

            ax.plot(
                np.arange(500), mean_val[plt_key], linewidth=3, color=colors[idx],
                label=r'T={abc}'.format(abc=str(max_step_idx))
            )
            if mdx > 0:
                ax.fill_between(
                    np.arange(500), mean_val[plt_key] - 0.5 * std_val[plt_key],
                                    mean_val[plt_key] + 0.5 * std_val[plt_key], alpha=0.2, facecolor=colors[idx]
                )

            frame_temp = (frame['max_project_steps'] == max_step_idx)
            frame_local = frame[frame_temp]
            unique_uuids = frame_local['uuid'].unique()

            correlation = 0.0
            for jdx, uuid_jdx in enumerate(unique_uuids):
                temp_frame2 = (frame_local['uuid'] == uuid_jdx)
                frame2 = frame_local[temp_frame2]

                # Compute correlation
                r, p = scipy.stats.pearsonr(frame2[plt_keys[0]], frame2[plt_keys[1]])
                correlation += r
                ax.plot(np.arange(500), frame2[plt_key], linewidth=2, color=colors[idx], alpha=0.3, label='')
            correlation = correlation / jdx

        make_paper_ready(ax, labelsize='large')
        ax.set_ylabel(y_ax_dict[plt_key], fontweight='bold', fontsize=15)
        if mdx == 1:
            if args.sampling_type == 'random':
                ax.set_xlabel(r'Fitting Iterations', fontweight='bold', fontsize=15)
            else:
                ax.set_xlabel(r'Fitting Iterations', fontweight='bold', fontsize=15)
        if mdx == 1 or mdx == 0:
            ax.legend(fontsize=15, loc='best')
        if mdx == 0:
            if args.sampling_type == 'random':
                title_text = r'GridWorld'  # % correlation
            else:
                title_text = r'Gridworld'  # % correlation
            ax.set_title(title_text, fontsize=16, fontweight='bold', loc='center')

        ax.set_xlim(0, 500)
        ax.set_ylim(ranges[mdx][0], ranges[mdx][1])

    plt.tight_layout()
    file_name = 'env_name_' + args.env_name + '_layers_' + args.layer_size + '_weighting_' + args.weighting_scheme + '_sampling_' + args.sampling_type + '_final_final.pdf'
    plt.savefig(os.path.join(args.log_dir, file_name))
    # plt.savefig('temp.png')
    print('Saved : ', os.path.join(args.log_dir, file_name))


###############################################

main(args.log_dir)