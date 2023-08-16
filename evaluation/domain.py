'''
    File name: domain.py
    Author: Richard Dirauf
    Python Version: 3.8
    Description: Evaluate the domain adaptation.
'''

import argparse
import os
import pandas as pd
import numpy as np
from matplotlib import patches, pyplot as plt
from fau_colors import cmaps
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

from pb_evaluation import PBEvaluation
import plot
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
                        help="Dir to the domain tables.")
    args = parser.parse_args()

    # should be plotted?
    if args.plot:
        plot_scores('precision', args.plot)
        plot_scores('recall', args.plot)
        plot_scores('f1', args.plot)
        plot_domain('map', args.plot)
        plot_domain('r1', args.plot)
        plot_kappas('jonas', args.plot)
        plot_kappas('sklearn', args.plot)
        plot_zoos('map', args.plot)
        plot_zoos('r1', args.plot)
        return

    domain_file_r1, domain_file_map = None, None
    zoos = [['Nürnberg', 0, 1], ['Berlin', 2, 3], ['Schönbrunn', 4, 5], ['Mulhouse', 6, 7]]
    r1_data, map_data, kappa_data, scores = [], [], [], []
    # pe = [126 / 206, 78 / 140, 95 / 182, 58 / 90]

    # go through versions
    for ver in ['color_back', 'color_pad', 'bw_back', 'bw_pad']:
        print(f'Version {ver}')

        for idx, z in enumerate(zoos):
            if False and z[0] == 'Berlin':
                # Domain with or without Berlin
                continue
            # read zoo features
            print(f'\nEvaluate domain adapt on {z[0]}')
            domain_dir = os.path.join(args.features, ver, z[0])
            if ver[-3:] == 'pad':
                dataset = os.path.join(args.dataset, '10_all_pad')
            else:
                dataset = os.path.join(args.dataset, '9_all_back')

            # create eval class
            pb_eval = PBEvaluation(dataset, domain_dir, 'track_info.csv', zoo_ids=z[1:])
            pb_eval.track_test_info.reset_index(inplace=True)
            mAP, CMC, mAP_per_id, CMC_per_id = pb_eval.eval_single_feat_file(250, f'features_pb_{z[0]}')

            y_label_idx, y_pred_idx = np.array(pb_eval.sorted_idx_per_epoch).T[:2]
            y_true = pb_eval.track_test_info.loc[y_label_idx]['id'].values
            y_pred = pb_eval.track_test_info.loc[y_pred_idx]['id'].values

            # calc precision
            precision = precision_score(y_true, y_pred, labels=z[1:], average='weighted')
            recall = recall_score(y_true, y_pred, labels=z[1:], average='weighted')
            score = f1_score(y_true, y_pred, labels=z[1:], average='weighted')

            # calc kappa vals w sklearn
            kappa = cohen_kappa_score(y_true, y_pred, )

            # calc kappa vals w best possible classifier
            # kappa = (CMC[0] - pe[idx]) / (1 - pe[idx])

            # add to data arrays
            kappa_data.append([ver, z[0], kappa])
            scores.append([ver, z[0], precision, recall, score])
            map_data.append([ver, z[0], mAP] + [val for _, val in sorted(mAP_per_id.items())])
            r1_data.append([ver, z[0], CMC[0]] + [val[0] for _, val in sorted(CMC_per_id.items())])

    # save data
    col = ['version', 'domain', 'mean', 'ID A', 'ID B']
    domain_file_r1 = pd.DataFrame(r1_data, columns=col)
    domain_file_map = pd.DataFrame(map_data, columns=col)
    scores_file = pd.DataFrame(scores, columns=['version', 'domain', 'precision', 'recall', 'f1'])
    kappa_file = pd.DataFrame(kappa_data, columns=['version', 'domain', 'kappa'])

    domain_file_r1.to_csv('domain_r1.csv', index=False)
    domain_file_map.to_csv('domain_map.csv', index=False)
    scores_file.to_csv('scores.csv', index=False)
    kappa_file.to_csv('kappa_vals_jonas.csv', index=False)
    print('--- saved ---')


def plot_zoos(metric, path):
    # load data
    base = pd.read_csv(os.path.join(path, f'domain_{metric}.csv'))
    zoos = ['Nürnberg', 'Berlin', 'Schönbrunn', 'Mulhouse']
    legend_patches = []
    c_counter = [4, 5, 6, 7]

    # def graph
    _, ax = plt.subplots()
    ax = plot.set_y_ax(ax, metric)
    plt.grid(axis='y')
    ax.set_xticks(range(4))
    ax.set_xticklabels(zoos)
    plt.xlabel("Zoos")
    print(base.groupby('version', as_index=False).mean())

    # set data
    off = np.array([-0.3, -0.1, 0.1, 0.3])
    i = np.array(range(4))

    for idx, v in enumerate(['color_back', 'color_pad', 'bw_back', 'bw_pad']):
        data = base[base['version'] == v]
        assert list(data['domain'].values) == zoos
        plt.bar(i + off[idx], data['mean'], align='center', width=0.2, color=colors[c_counter[idx]])
        legend_patches.append(patches.Patch(color=colors[c_counter[idx]], label=f'{v}'))

    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.78)

    plt.show()


def plot_kappas(metric, path):
    # load data
    base = pd.read_csv(os.path.join(path, f'kappa_vals_{metric}_w_berlin.csv'))
    zoos = ['Nürnberg', 'Berlin', 'Schönbrunn', 'Mulhouse']
    legend_patches = []
    c_counter = [4, 5, 6, 7]

    # def graph
    _, ax = plt.subplots()
    plt.ylabel('Kappa')
    plt.xlabel("Zoos")
    ax.set_ylim([-1.0, 1.0])
    ax.set_axisbelow(True)
    plt.grid(axis='y')
    ax.set_xticks(range(4))
    ax.set_xticklabels(zoos)

    # set data
    off = np.array([-0.3, -0.1, 0.1, 0.3])
    i = np.array(range(4))

    for idx, v in enumerate(['color_back', 'color_pad', 'bw_back', 'bw_pad']):
        data = base[base['version'] == v]
        assert list(data['domain'].values) == zoos
        plt.bar(i + off[idx], data['kappa'], align='center', width=0.2, color=colors[c_counter[idx]])
        legend_patches.append(patches.Patch(color=colors[c_counter[idx]], label=f'{v}'))

    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.78)

    plt.show()


def plot_scores(metric, path):
    # load data
    base = pd.read_csv(os.path.join(path, 'scores_wo_berlin.csv'))
    zoos = ['Nürnberg', 'Schönbrunn', 'Mulhouse']
    legend_patches = []
    c_counter = [4, 5, 6, 7]

    # def graph
    _, ax = plt.subplots()
    if metric == 'precision':
        plt.ylabel('Precision (%)')
    elif metric == 'recall':
        plt.ylabel('Recall (%)')
    else:
        plt.ylabel('F1 (%)')
    plt.xlabel("Zoos")
    ax.set_ylim([0.0, 1.0])
    ax.set_yticks([i / 100 for i in range(0, 101, 10)])
    ax.set_yticklabels(range(0, 101, 10))
    ax.set_axisbelow(True)
    plt.grid(axis='y')
    ax.set_xticks(range(3))
    ax.set_xticklabels(zoos)

    # set data
    off = np.array([-0.3, -0.1, 0.1, 0.3])
    i = np.array(range(3))

    for idx, v in enumerate(['color_back', 'color_pad', 'bw_back', 'bw_pad']):
        data = base[base['version'] == v]
        assert list(data['domain'].values) == zoos
        plt.bar(i + off[idx], data[metric], align='center', width=0.2, color=colors[c_counter[idx]])
        legend_patches.append(patches.Patch(color=colors[c_counter[idx]], label=f'{v}'))

    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.78)

    plt.show()


def plot_domain(metric, path):
    # load data
    base = pd.read_csv(os.path.join(path, f'domain_{metric}.csv'))
    simple = pd.read_csv(os.path.join(path, f'domain_simple_{metric}.csv'))
    adv = pd.read_csv(os.path.join(path, f'domain_adv_{metric}.csv'))
    versions = ['color_back', 'color_pad', 'bw_back', 'bw_pad']
    legend_patches = []

    # def graph
    _, ax = plt.subplots()
    ax = plot.set_y_ax(ax, metric)
    plt.grid(axis='y')
    ax.set_xticks(range(4))
    ax.set_xticklabels(versions)

    # plot data
    off = 0.25
    i = np.array(range(4))
    base_mean, base_std = get_mean_std_domain(base)
    simple_mean, simple_std = get_mean_std_domain(simple)
    adv_mean, adv_std = get_mean_std_domain(adv)

    # plot bars
    plt.bar(i - off, base_mean, yerr=base_std, align='center', width=0.25, color=colors[2], ecolor='black', capsize=5)
    legend_patches.append(patches.Patch(color=colors[2], label='Unimproved'))
    plt.bar(i, simple_mean, yerr=simple_std, align='center', width=0.25, color=colors[4], ecolor='black', capsize=5)
    legend_patches.append(patches.Patch(color=colors[4], label='Simple re-rank'))
    plt.bar(i + off, adv_mean, yerr=adv_std, align='center', width=0.25, color=colors[8], ecolor='black', capsize=5)
    legend_patches.append(patches.Patch(color=colors[8], label='Adv. re-rank'))

    # set legend
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title='Improvements')
    plt.subplots_adjust(right=0.749)

    plt.show()


def get_mean_std_domain(df):
    std = df.groupby('version').std(ddof=0)['mean']
    mean = df.groupby('version').mean()['mean']
    std = [std[i] for i in ['color_back', 'color_pad', 'bw_back', 'bw_pad']]
    mean = [mean[i] for i in ['color_back', 'color_pad', 'bw_back', 'bw_pad']]
    return mean, std


if __name__ == "__main__":
    main()
