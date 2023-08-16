from matplotlib import pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np
from fau_colors import cmaps
from matplotlib import patches as mpatches
colors = cmaps.faculties_all


def main():
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        type=str,
                        required=True,
                        help="Path to dir with files.")
    parser.add_argument("--plot",
                        type=str,
                        required=True,
                        help="Which plot?")
    args = parser.parse_args()

    if args.plot == 'imp_per_version':
        plot_imp_per_version(args.dir, 'r1')
        plot_imp_per_version(args.dir, 'map')
    elif args.plot == '5_fold_per_zoo':
        plot_5_fold_per_zoo(args.dir, 'r1')
        plot_5_fold_per_zoo(args.dir, 'map')
    elif args.plot == 'all_versions_per_epoch':
        plot_all_versions_per_epoch(args.dir, 'rank-1')
        plot_all_versions_per_epoch(args.dir, 'map')
    elif args.plot == 'comb_per_version':
        plot_comb_per_version(args.dir, 'r1')
        plot_comb_per_version(args.dir, 'map')
    elif args.plot == 'train_loss_acc':
        plot_train(args.dir, 'loss')
        plot_train(args.dir, 'acc')


def plot_train(dir, metric):
    # load files
    metric_per_epoch = {}
    c_counter = [4, 5, 6, 7]
    versions = ['color_back', 'color_pad', 'bw_back', 'bw_pad']
    legend_patches = []
    for v in versions:
        metric_per_epoch[v] = pd.read_csv(os.path.join(dir, v, 'training_statistics.csv'))
    epochs = metric_per_epoch['color_back']['epoch']

    # set plot
    _, ax = plt.subplots()
    if metric == 'acc':
        plt.ylabel('Accuracy (%)')
        ax.set_ylim([0.0, 1.0])
        ax.set_yticks([i / 100 for i in range(0, 101, 10)])
        ax.set_yticklabels(range(0, 101, 10))
    else:
        plt.ylabel('Loss')
        ax.set_ylim([0.0, 2.6])

    ax.set_axisbelow(True)
    plt.xlabel("Epochs")
    plt.grid()
    ax.set_xticks(range(0, 251, 25))
    ax.set_xlim([0, 250])

    # plot data
    for i, v in enumerate(versions):
        plt.plot(epochs, metric_per_epoch[v][metric], color=colors[c_counter[i]])
        legend_patches.append(mpatches.Patch(color=colors[c_counter[i]], label=f'{v}'))

    # def labels
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.78)
    plt.show()


def plot_comb_per_version(file_path, metric):
    # load files
    base = pd.read_csv(os.path.join(file_path, '5_fold_eval', f'5_fold_{metric}.csv'))
    comb_simple = pd.read_csv(os.path.join(file_path, 'combined', f'comb_simple_{metric}.csv'))
    comb_adv = pd.read_csv(os.path.join(file_path, 'combined', f'comb_adv_{metric}.csv'))
    versions = ['color_back', 'color_pad', 'bw_back', 'bw_pad']
    legend_patches = []

    # def graph
    _, ax = plt.subplots()
    ax = set_y_ax(ax, metric)
    plt.xlabel("Datasets")
    plt.grid(axis='y')
    ax.set_xticks(range(4))
    ax.set_xticklabels(versions)

    # plot data
    off = np.array([-0.25, 0, 0.25])
    i = np.array(range(4))
    base_mean, base_std = get_mean_std_by_v(base)
    simple_mean, simple_std = get_mean_std_by_v(comb_simple)
    adv_mean, adv_std = get_mean_std_by_v(comb_adv)

    # plot bars
    plt.bar(i + off[0], base_mean, yerr=base_std, align='center', width=0.25, color=colors[2], ecolor='black', capsize=5)
    legend_patches.append(mpatches.Patch(color=colors[2], label='Unimproved'))
    plt.bar(i + off[1], simple_mean, yerr=simple_std, align='center', width=0.25, color=colors[4], ecolor='black', capsize=5)
    legend_patches.append(mpatches.Patch(color=colors[4], label='Only + simple'))
    plt.bar(i + off[2], adv_mean, yerr=adv_std, align='center', width=0.25, color=colors[6], ecolor='black', capsize=5)
    legend_patches.append(mpatches.Patch(color=colors[6], label='Only + adv.'))

    # set legend
    ax.set_ylim([0.5, 1.0])
    ax.set_yticks([i / 100 for i in range(50, 101, 5)])
    ax.set_yticklabels(range(50, 101, 5))
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title='Improvements')
    plt.subplots_adjust(right=0.755)

    plt.show()


def plot_imp_per_version(file_path, metric):
    # load files
    base = pd.read_csv(os.path.join(file_path, '5_fold_eval', f'5_fold_{metric}.csv'))
    only_zoo = pd.read_csv(os.path.join(file_path, 'improvements', f'only_zoo_{metric}.csv'))
    re_simple_5 = pd.read_csv(os.path.join(file_path, 'improvements', f're_rank_simple_k5_{metric}.csv'))
    re_adv = pd.read_csv(os.path.join(file_path, 'improvements', f're_rank_adv_{metric}.csv'))
    versions = ['color_back', 'color_pad', 'bw_back', 'bw_pad']
    legend_patches = []

    # def graph
    _, ax = plt.subplots()

    # plot data
    off = np.array([-0.3, -0.1, 0.1, 0.3])
    i = np.array(range(4))
    base_mean, base_std = get_mean_std_by_v(base)
    only_mean, only_std = get_mean_std_by_v(only_zoo)
    re5_mean, re5_std = get_mean_std_by_v(re_simple_5)
    ra_mean, ra_std = get_mean_std_by_v(re_adv)

    # plot bars
    plt.bar(i + off[0], base_mean, yerr=base_std, align='center', width=0.2, color=colors[2], ecolor='black', capsize=5)
    legend_patches.append(mpatches.Patch(color=colors[2], label='Unimproved'))
    plt.bar(i + off[1], only_mean, yerr=only_std, align='center', width=0.2, color=colors[4], ecolor='black', capsize=5)
    legend_patches.append(mpatches.Patch(color=colors[4], label='Only zoo'))
    plt.bar(i + off[2], re5_mean, yerr=re5_std, align='center', width=0.2, color=colors[8], ecolor='black', capsize=5)
    legend_patches.append(mpatches.Patch(color=colors[8], label='Simple re-rank'))
    plt.bar(i + off[3], ra_mean, yerr=ra_std, align='center', width=0.2, color=colors[10], ecolor='black', capsize=5)
    legend_patches.append(mpatches.Patch(color=colors[10], label='Adv. re-rank'))

    # set legend
    ax = set_y_ax(ax, metric)
    ax.set_ylim([0.5, 1.0])
    ax.set_yticks([i / 100 for i in range(50, 101, 5)])
    ax.set_yticklabels(range(50, 101, 5))
    ax.set_axisbelow(True)
    plt.xlabel("Datasets")
    plt.grid(axis='y')
    ax.set_xticks(range(4))
    ax.set_xticklabels(versions)
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title='Improvements')
    plt.subplots_adjust(right=0.749)

    plt.show()


def plot_5_fold_per_zoo(dir, metric):
    # load data
    base = pd.read_csv(os.path.join(dir, '5_fold_eval', f'5_fold_{metric}.csv'))
    zoos = ['Nürnberg', 'Berlin', 'Schönbrunn', 'Mulhouse']
    legend_patches = []
    c_counter = [4, 5, 6, 7]

    # def graph
    _, ax = plt.subplots()
    ax = set_y_ax(ax, metric)
    plt.xlabel("Zoos")
    plt.grid(axis='y')
    ax.set_xticks(range(4))
    ax.set_xticklabels(zoos)

    # set data
    off = np.array([-0.3, -0.1, 0.1, 0.3])
    i = np.array(range(4))
    std = base.groupby('version', as_index=False).std(ddof=0)
    mean = base.groupby('version', as_index=False).mean()
    print(mean)

    for idx, v in enumerate(['color_back', 'color_pad', 'bw_back', 'bw_pad']):
        v_mean = mean[mean['version'] == v]
        v_std = std[std['version'] == v]
        plt.bar(i + off[idx], v_mean[zoos].values[0], yerr=v_std[zoos].values[0], align='center', width=0.2, color=colors[c_counter[idx]],
                ecolor='black', capsize=5)
        legend_patches.append(mpatches.Patch(color=colors[c_counter[idx]], label=f'{v}'))

    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title='Datasets')
    plt.subplots_adjust(right=0.78)

    plt.show()


def plot_all_versions_per_epoch(dir, metric):
    # load files
    map_per_epoch = {}
    c_counter = [4, 5, 6, 7, 8, 9, 10, 11]
    versions = ['color_back', 'color_pad', 'bw_back', 'bw_pad']
    legend_patches = []
    for v in versions:
        if metric == 'map':
            map_per_epoch[v] = pd.read_csv(os.path.join(dir, v, f'{metric}.csv'))['mAP all']
        else:
            map_per_epoch[v] = pd.read_csv(os.path.join(dir, v, f'{metric}.csv'))[f'{metric} all']
    epochs = [1] + list(range(10, 251, 10))

    # set plot
    _, ax = plt.subplots()
    ax = set_y_ax(ax, metric)
    ax.set_ylim([0.4, 1.0])
    ax.set_yticks([i / 100 for i in range(40, 101, 5)])
    ax.set_yticklabels(range(40, 101, 5))
    plt.xlabel("Epochs")
    plt.grid()
    ax.set_xticks(range(0, 251, 25))
    ax.set_xlim([0, 250])

    # plot data
    for i, v in enumerate(versions):
        plt.plot(epochs, map_per_epoch[v], color=colors[c_counter[i]])
        legend_patches.append(mpatches.Patch(color=colors[c_counter[i]], label=f'{v}'))

    # def labels
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title='Datasets')
    plt.subplots_adjust(right=0.78)
    plt.show()


def set_y_ax(ax, metric):
    if metric == 'r1' or metric == 'rank-1':
        plt.ylabel('Rank-1 (%)')
    else:
        plt.ylabel('mAP (%)')
    plt.xlabel("Datasets")
    ax.set_ylim([0.0, 1.0])
    ax.set_yticks([i / 100 for i in range(0, 101, 10)])
    ax.set_yticklabels(range(0, 101, 10))
    ax.set_axisbelow(True)
    return ax


def get_mean_std_by_v(df):
    std_fold = df.groupby('version').std(ddof=0)['mean']
    mean_fold = df.groupby('version').mean()['mean']
    std_fold = [std_fold[i] for i in ['color_back', 'color_pad', 'bw_back', 'bw_pad']]
    mean_fold = [mean_fold[i] for i in ['color_back', 'color_pad', 'bw_back', 'bw_pad']]
    return mean_fold, std_fold


def get_mean_std_by_zoo(df, v):
    z = ['Nürnberg', 'Berlin', 'Schönbrunn', 'Mulhouse']
    df = df[df['version'] == v]
    std_fold = df.groupby('version').std(ddof=0)[z]
    mean_fold = df.groupby('version').mean()[z]
    std_fold = [std_fold[i].values[0] for i in z]
    mean_fold = [mean_fold[i].values[0] for i in z]
    return mean_fold, std_fold


if __name__ == "__main__":
    main()
