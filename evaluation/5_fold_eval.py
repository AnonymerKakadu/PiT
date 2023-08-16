'''
    File name: 5_fold_eval.py
    Author: Richard Dirauf
    Python Version: 3.8
    Description: Evaluate the 5 fold cross validation.
'''

import argparse
import os
import pandas as pd
import numpy as np
from matplotlib import patches, pyplot as plt
from fau_colors import cmaps

from pb_evaluation import PBEvaluation
colors = cmaps.faculties_all


def main():
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        help="Path to datasets.")
    parser.add_argument("--features",
                        type=str,
                        help="Path to fold feature dir.")
    parser.add_argument("--plot",
                        type=str,
                        help="Dir of fixed train-test eval file.")
    args = parser.parse_args()
    fold_file_r1, fold_file_map = None, None
    versions = [['Nürnberg', 0, 1], ['Berlin', 2, 3], ['Schönbrunn', 4, 5], ['Mulhouse', 6, 7]]
    r1_data, map_data = [], []

    # should be plotted?
    if args.plot:
        plot_comp_mars('r1', args.plot)
        plot_comp_mars('map', args.plot)
        plot_5_r2('r2', args.plot)
        plot_5('map', args.plot)
        plot_5('r1', args.plot)
        plot_all('map', args.plot)
        plot_all('r1', args.plot)
        return

    # go through versions
    for ver in ['color_back', 'color_pad', 'bw_back', 'bw_pad']:
        print(f'Version {ver}')
        mAPs, rank1s = [], []

        for i in range(5):
            # read fold
            print(f'\nEvaluate Fold {i + 1}')
            fold_dir = os.path.join(args.features, ver, f'fold_{i + 1}')
            fold_info_file = f'track_fold_info_{i + 1}.csv'
            if ver[-3:] == 'pad':
                dataset = os.path.join(args.dataset, '10_all_pad')
            else:
                dataset = os.path.join(args.dataset, '9_all_back')

            # create eval class
            pb_eval = PBEvaluation(dataset, fold_dir, fold_info_file)
            mAP, CMC, mAP_per_id, CMC_per_id = pb_eval.eval_single_feat_file(250, f'features_pb_fold_{i + 1}')
            mAPs.append(mAP)
            rank1s.append(CMC[0])

            # add to data arrays
            map_data.append([ver, i, mAP] + [val for _, val in sorted(mAP_per_id.items())] + [val for _, val in pb_eval.map_per_zoo.items()])
            r1_data.append([ver, i, CMC[0]] + [val[0] for _, val in sorted(CMC_per_id.items())] + [val for _, val in pb_eval.r1_per_zoo.items()])

        print('\nMean values of all 5 folds:')
        fold_mAP = np.sum(mAPs) / len(mAPs)
        fold_r1 = np.sum(rank1s) / len(rank1s)
        print(f'mAP: {fold_mAP}')
        print(f'R1: {fold_r1}')

    # save data
    col = ['version', 'fold', 'mean'] + [str(a_id) for a_id in range(8)] + [v[0] for v in versions]
    fold_file_r1 = pd.DataFrame(r1_data, columns=col)
    fold_file_map = pd.DataFrame(map_data, columns=col)

    fold_file_r1.to_csv('5_fold_r1.csv', index=False)
    fold_file_map.to_csv('5_fold_map.csv', index=False)
    print('--- saved ---')


def plot_all(metric, path):
    fold = pd.read_csv(os.path.join(path, '5_fold_eval', f'5_fold_{metric}.csv'))
    versions = fold['version'].unique()
    legend_patches = []

    # plot graph
    _, ax = plt.subplots()
    if metric == 'r1':
        plt.ylabel('Rank-1 (%)')
    else:
        plt.ylabel('mAP (%)')
    plt.xlabel("Datasets")
    ax.set_axisbelow(True)
    plt.grid(axis='y')

    # set data
    off = np.array([-0.3, -0.15, 0, 0.15, 0.3])
    x = np.array(range(4))
    colors_n = colors[:4] + [colors[6]]

    for i in range(5):
        f_data = fold[fold['fold'] == i]['mean']
        plt.bar(x + off[i], f_data, align='center', width=0.15, color=colors_n[i])
        legend_patches.append(patches.Patch(color=colors_n[i], label=f'{i + 1}'))

    # set x values
    ax.set_xticks(range(4))
    ax.set_xticklabels(versions)
    ax.set_ylim([0.5, 1.0])
    ax.set_yticks([i / 100 for i in range(50, 101, 5)])
    ax.set_yticklabels(range(50, 101, 5))
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title='Fold')
    plt.subplots_adjust(right=0.88)

    plt.show()


def plot_5_r2(metric, path):
    fold_r1 = pd.read_csv(os.path.join(path, '5_fold_eval', '5_fold_r1.csv'))
    fold_r2 = pd.read_csv(os.path.join(path, '5_fold_eval', f'5_fold_{metric}.csv'))
    legend_patches = []

    # plot graph
    _, ax = plt.subplots()
    plt.ylabel('Rank-k (%)')
    plt.xlabel("Datasets")
    ax.set_axisbelow(True)
    plt.grid(axis='y')
    versions = ['color_back', 'color_pad', 'bw_back', 'bw_pad']

    # set data
    off = 0.25
    i = np.array(range(4))
    std_r1 = fold_r1.groupby('version').std(ddof=0)['mean']
    mean_r1 = fold_r1.groupby('version').mean()['mean']
    std_r1 = [std_r1[i] for i in ['color_back', 'color_pad', 'bw_back', 'bw_pad']]
    mean_r1 = [mean_r1[i] for i in ['color_back', 'color_pad', 'bw_back', 'bw_pad']]
    std_r2 = fold_r2.groupby('version').std(ddof=0)['mean']
    mean_r2 = fold_r2.groupby('version').mean()['mean']
    std_r2 = [std_r2[i] for i in ['color_back', 'color_pad', 'bw_back', 'bw_pad']]
    mean_r2 = [mean_r2[i] for i in ['color_back', 'color_pad', 'bw_back', 'bw_pad']]

    plt.bar(i - 0.15, mean_r1, yerr=std_r1, align='center', width=off, color=colors[0], ecolor='black', capsize=10)
    legend_patches.append(patches.Patch(color=colors[0], label='Rank-1'))
    plt.bar(i + 0.15, mean_r2, yerr=std_r2, align='center', width=off, color=colors[2], ecolor='black', capsize=10)
    legend_patches.append(patches.Patch(color=colors[2], label='Rank-2'))

    # set x values
    ax.set_xticks(range(4))
    ax.set_xticklabels(versions)
    ax.set_ylim([0.5, 1.0])
    ax.set_yticks([i / 100 for i in range(50, 101, 5)])
    ax.set_yticklabels(range(50, 101, 5))
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)

    plt.show()


def plot_5(metric, path):
    fold = pd.read_csv(os.path.join(path, '5_fold_eval', f'5_fold_{metric}.csv'))
    legend_patches = []
    metric_per_epoch = pd.DataFrame([[i] for i in [1] + list(range(10, 251, 10))], columns=['epoch'])

    # plot graph
    _, ax = plt.subplots()
    for v in ['color_back', 'color_pad', 'bw_back', 'bw_pad']:
        if metric == 'r1':
            plt.ylabel('Rank-1 (%)')
            metric_per_epoch[v] = pd.read_csv(os.path.join(path, v, 'rank-1.csv'))['rank-1 all']
        else:
            plt.ylabel('mAP (%)')
            metric_per_epoch[v] = pd.read_csv(os.path.join(path, v, 'map.csv'))['mAP all']
    plt.xlabel("Datasets")
    ax.set_axisbelow(True)
    plt.grid(axis='y')
    versions = metric_per_epoch.columns[1:]

    # set data
    off = 0.25
    i = np.array(range(4))
    fixed = metric_per_epoch[metric_per_epoch['epoch'] == 250].values[0][1:]
    std_fold = fold.groupby('version').std(ddof=0)['mean']
    mean_fold = fold.groupby('version').mean()['mean']
    std_fold = [std_fold[i] for i in ['color_back', 'color_pad', 'bw_back', 'bw_pad']]
    mean_fold = [mean_fold[i] for i in ['color_back', 'color_pad', 'bw_back', 'bw_pad']]

    plt.bar(i - 0.15, fixed, align='center', width=off, color=colors[0])
    legend_patches.append(patches.Patch(color=colors[0], label='Fixed split'))
    plt.bar(i + 0.15, mean_fold, yerr=std_fold, align='center', width=off, color=colors[2], ecolor='black', capsize=10)
    legend_patches.append(patches.Patch(color=colors[2], label='5-Fold'))

    # set x values
    ax.set_xticks(range(4))
    ax.set_xticklabels(versions)
    ax.set_ylim([0.5, 1.0])
    ax.set_yticks([i / 100 for i in range(50, 101, 5)])
    ax.set_yticklabels(range(50, 101, 5))

    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)

    plt.show()


def plot_comp_mars(metric, path):
    fold = pd.read_csv(os.path.join(path, '5_fold_eval', f'5_fold_{metric}.csv'))
    mars = {'r1': 0.8702, 'map': 0.7847}

    # plot graph
    _, ax = plt.subplots()
    if metric == 'r1':
        plt.ylabel('Rank-1 (%)')
    else:
        plt.ylabel('mAP (%)')
    plt.xlabel("Datasets")
    ax.set_axisbelow(True)
    plt.grid(axis='y')

    # set data
    std_fold = fold.groupby('version').std(ddof=0)['mean']['color_pad']
    mean_fold = fold.groupby('version').mean()['mean']['color_pad']

    plt.bar(0, mean_fold, yerr=std_fold, align='center', width=0.3, color=colors[2], ecolor='black', capsize=10)
    plt.bar(1, mars[metric], align='center', width=0.3, color=colors[3])

    # set x values
    ax.set_xticks(range(2))
    ax.set_xticklabels(['Polar Bear', 'MARS'])
    ax.set_ylim([0.0, 1.0])
    ax.set_yticks([i / 100 for i in range(0, 101, 10)])
    ax.set_yticklabels(range(0, 101, 10))

    plt.show()


if __name__ == "__main__":
    main()
