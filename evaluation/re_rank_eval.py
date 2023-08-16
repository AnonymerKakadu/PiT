'''
    File name: re_rank_eval.py
    Author: Richard Dirauf
    Python Version: 3.8
    Description: Evaluate the if re ranking can improve things.
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
    parser.add_argument("--rank",
                        type=str,
                        required=True,
                        help="Re-rank option. simple or adv.")
    args = parser.parse_args()
    fold_file_r1, fold_file_map = None, None
    versions = [['Nürnberg', 0, 1], ['Berlin', 2, 3], ['Schönbrunn', 4, 5], ['Mulhouse', 6, 7]]
    r1_data, map_data = [], []

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
            pb_eval = PBEvaluation(dataset, fold_dir, fold_info_file, re_rank=args.rank)
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

    fold_file_r1.to_csv(f're_rank_{args.rank}_r1.csv', index=False)
    fold_file_map.to_csv(f're_rank_{args.rank}_map.csv', index=False)
    print('--- saved ---')


if __name__ == "__main__":
    main()
