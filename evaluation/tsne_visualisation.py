'''
    File name: tsne_visualisation.py
    Author: Richard Dirauf
    Python Version: 3.8
    Description: Visualize the features with t-SNE.
'''

import argparse
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from fau_colors import cmaps
from matplotlib import patches as mpatches

from sklearn.manifold import TSNE
colors = cmaps.faculties_all


def main():
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to dataset.")
    parser.add_argument("--features",
                        type=str,
                        required=True,
                        help="Path to feature file.")
    args = parser.parse_args()

    # load db
    track_test_info = pd.read_csv(os.path.join(args.dataset, 'track_test_info.csv'))
    # track_test_info = track_test_info[track_test_info['id'].isin(zoo_ids)]
    features_test = np.loadtxt(args.features)
    print('Loaded features: ' + str(features_test.shape))

    tsne = TSNE(n_components=2, verbose=1, perplexity=60, n_iter=1500, init='pca', learning_rate='auto')
    features_test = tsne.fit_transform(features_test)
    print('Dim reduction of features: ' + str(features_test.shape))

    # get only ids from track info
    pids = track_test_info['id']
    # plot features
    plotme(features_test, pids)


def plotme(features, pids):
    c_counter = [4, 5, 6, 7, 8, 9, 10, 11]
    legend_patches = []

    # go through the animals
    for j in np.unique(pids):
        # create mask where tracklets of that animal are true
        mask = (pids == j)

        # only take features of that animal
        subset_fe = features[mask]

        # plot points and lines between them
        plt.plot(subset_fe[:, 0], subset_fe[:, 1], color=colors[c_counter[j]], zorder=1, linewidth=0.5)
        plt.scatter(subset_fe[:, 0], subset_fe[:, 1], 20., c=colors[c_counter[j]], zorder=2)
        legend_patches.append(mpatches.Patch(color=colors[c_counter[j]], label=f'{j}'))

    # plot things
    plt.ylabel('Y')
    plt.xlabel("X")
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title='ID')
    plt.subplots_adjust(right=0.85)
    plt.show()


if __name__ == "__main__":
    main()
