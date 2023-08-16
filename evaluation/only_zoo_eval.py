'''
    File name: only_zoo_eval.py
    Author: Richard Dirauf
    Python Version: 3.8
    Description: Evaluate the zoos one by one.
'''

import argparse
import os
import pandas as pd
import numpy as np
from fau_colors import cmaps

from pb_evaluation import PBEvaluation
colors = cmaps.faculties_all


def main():
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to datasets.")
    parser.add_argument("--features",
                        type=str,
                        required=True,
                        help="Path to fold feature dir.")
    args = parser.parse_args()
    fold_file_r1, fold_file_map = None, None
    versions = [['Nürnberg', 0, 1], ['Berlin', 2, 3], ['Schönbrunn', 4, 5], ['Mulhouse', 6, 7]]
    r1_data, map_data = [], []

    # go through versions
    for ver in ['color_back', 'color_pad', 'bw_back', 'bw_pad']:
        print(f'Version {ver}')

        for i in range(5):
            # def save arrays
            only_map_per_id, only_r1_per_id, only_map_per_zoo, only_r1_per_zoo = [], [], [], []
            all_map, all_r1 = [], []

            # read fold
            print(f'\nEvaluate Fold {i + 1}')
            fold_dir = os.path.join(args.features, ver, f'fold_{i + 1}')
            fold_info_file = f'track_fold_info_{i + 1}.csv'
            if ver[-3:] == 'pad':
                dataset = os.path.join(args.dataset, '10_all_pad')
            else:
                dataset = os.path.join(args.dataset, '9_all_back')

            # take only zoo
            for z in versions:
                # create eval class
                pb_eval = PBEvaluation(dataset, fold_dir, fold_info_file, zoo_ids=z[1:])
                _, _, mAP_per_id, CMC_per_id = pb_eval.eval_single_feat_file(250, f'features_pb_fold_{i + 1}')

                # add to arrays
                only_map_per_id += [val for _, val in sorted(mAP_per_id.items())]
                only_r1_per_id += [val[0] for _, val in sorted(CMC_per_id.items())]
                only_map_per_zoo += [val for _, val in pb_eval.map_per_zoo.items()]
                only_r1_per_zoo += [val for _, val in pb_eval.r1_per_zoo.items()]
                all_map += list(pb_eval.ap)
                all_r1 += pb_eval.r1

            only_map = np.sum(all_map) / len(all_map)
            only_r1 = np.sum(all_r1) / len(all_r1)

            # add to data arrays
            map_data.append([ver, i, only_map] + only_map_per_id + only_map_per_zoo)
            r1_data.append([ver, i, only_r1] + only_r1_per_id + only_r1_per_zoo)

    # save data
    col = ['version', 'fold', 'mean'] + [str(a_id) for a_id in range(8)] + [v[0] for v in versions]
    fold_file_r1 = pd.DataFrame(r1_data, columns=col)
    fold_file_map = pd.DataFrame(map_data, columns=col)

    fold_file_r1.to_csv('only_zoo_r1.csv', index=False)
    fold_file_map.to_csv('only_zoo_map.csv', index=False)
    print('--- saved ---')


if __name__ == "__main__":
    main()
