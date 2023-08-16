'''
    File name: main_pb_eval.py
    Author: Richard Dirauf
    Python Version: 3.8
    Description: Main function to do the evaluation and other stuff.
'''


import argparse
import os
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from fau_colors import cmaps
from matplotlib import patches as mpatches
from sklearn.metrics import ConfusionMatrixDisplay

from pb_evaluation import PBEvaluation
colors = cmaps.faculties_all


def plot(epochs, mAPs, rank1s, mAPs_per_id, CMCs_per_id, mAPs_per_zoo, r1s_per_zoo):
    # plot diagrams: mAP
    _, ax = plt.subplots()
    c_counter = [4, 5, 6, 7, 8, 9, 10, 11]
    legend_patches = []

    # plot lines
    for key, value in sorted(mAPs_per_id.items()):
        plt.plot(epochs, value, color=colors[c_counter[key]])
        legend_patches.append(mpatches.Patch(color=colors[c_counter[key]], label=f'ID {key}'))

    plt.plot(epochs, mAPs, color='black', linewidth=4)
    legend_patches.append(mpatches.Patch(color='black', label='All'))

    # def labels and axes
    plt.ylabel('mAP')
    plt.xlabel("Epochs")
    plt.grid()
    ax.set_xticks(range(0, 251, 50))
    ax.set_xlim([0, 250])
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)
    plt.show()

    # plot diagrams: rank-1
    _, ax = plt.subplots()
    legend_patches = []

    # plot lines
    for key, value in sorted(CMCs_per_id.items()):
        value = [s_cmc[0] for s_cmc in value]
        plt.plot(epochs, value, color=colors[c_counter[key]])
        legend_patches.append(mpatches.Patch(color=colors[c_counter[key]], label=f'ID {key}'))

    plt.plot(epochs, rank1s, color='black', linewidth=4)
    legend_patches.append(mpatches.Patch(color='black', label='All'))

    # def labels and axes
    plt.ylabel('Rank-1')
    plt.xlabel("Epochs")
    plt.grid()
    ax.set_xticks(range(0, 251, 50))
    ax.set_xlim([0, 250])
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)
    plt.show()

    # save data to csv
    map_csv = [epochs, mAPs] + [v for _, v in sorted(mAPs_per_id.items())] + [val for _, val in mAPs_per_zoo.items()]
    map_csv = np.array(map_csv).T
    map_csv = pd.DataFrame(map_csv, columns=['epoch', 'mAP all'] + list(sorted(mAPs_per_id.keys())) + list(mAPs_per_zoo.keys()))
    cmc_csv = [epochs, rank1s] + [np.array(v)[:, 0] for _, v in sorted(CMCs_per_id.items())] + [val for _, val in r1s_per_zoo.items()]
    cmc_csv = np.array(cmc_csv).T
    cmc_csv = pd.DataFrame(cmc_csv, columns=['epoch', 'rank-1 all'] + list(sorted(CMCs_per_id.keys())) + list(r1s_per_zoo.keys()))
    map_csv.to_csv('map.csv', index=False)
    cmc_csv.to_csv('rank-1.csv', index=False)


def get_img(df, dataset_dir):
    a_id, cam, tracklet, start, end = df
    basename = '{:03d}'.format(a_id) + 'C{:02d}'.format(cam) + 'T{:03d}'.format(tracklet) + 'F{:03d}'.format(int((end - start) / 2)) + '.jpg'
    path = os.path.join(dataset_dir, '{:03d}'.format(a_id), basename)
    return cv2.imread(path)


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
                        help="Path to feature dir.")
    parser.add_argument("--feature_file",
                        type=str,
                        help="If only one specific feature file should be read.")
    parser.add_argument("--query_id",
                        type=int,
                        help="Display missdetections from this specific id.")
    args = parser.parse_args()
    # get values
    dataset = os.path.abspath(args.dataset)
    features = os.path.abspath(args.features)

    pb_eval = PBEvaluation(dataset, features)

    if args.feature_file is None:
        # review all epochs
        epochs, mAPs, rank1s, mAPs_per_id, CMCs_per_id, mAPs_per_zoo, r1s_per_zoo = pb_eval.eval_feat_dir()
        plot(epochs, mAPs, rank1s, mAPs_per_id, CMCs_per_id, mAPs_per_zoo, r1s_per_zoo)
        return

    # only one file with missdetection showing
    f_epoch = int(args.feature_file.split('_')[-1])
    pb_eval.eval_single_feat_file(f_epoch, args.feature_file)

    # create confusion matrix
    y_label_idx, y_pred_idx = np.array(pb_eval.sorted_idx_per_epoch).T[:2]
    y_label = pb_eval.track_test_info.loc[y_label_idx]['id'].values
    y_pred = pb_eval.track_test_info.loc[y_pred_idx]['id'].values
    # show confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_label, y_pred, cmap='GnBu')
    plt.ylabel('True label (ID)')
    plt.xlabel("Predicted label (ID)")
    plt.show()

    if args.query_id is None or args.query_id not in list(range(8)):
        raise Exception()

    q_id = args.query_id
    empty = np.zeros((128, 10, 3), dtype=np.uint8)
    miss = False
    img_list = None

    # get through sorted idx
    for q in range(121):  # querys
        # get query infos
        q_idx = pb_eval.sorted_idx_per_epoch[q][0]

        if pb_eval.track_test_info.iloc[q_idx]['id'] == q_id:
            # is desired idx

            # print values
            print(f'\nquery_idx: {q_idx}, query_id: {q_id}')
            for i in range(1, 4):  # sorted idx
                found_idx = pb_eval.sorted_idx_per_epoch[q][i]
                found_id, cam, track, _, _ = pb_eval.track_test_info.iloc[found_idx]
                print(f'          {found_idx},           {found_id}')

                if i == 1 and found_id != q_id:
                    miss = True

                if miss:
                    # show missdetected images
                    if img_list is None:
                        img_list = get_img(pb_eval.track_test_info.iloc[q_idx], dataset)
                        _, q_cam, q_track, _, _ = pb_eval.track_test_info.iloc[q_idx]
                        texty = f'{q_id}_C{q_cam}_T{q_track}'
                        img_list = cv2.putText(img_list, texty, (0, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

                    # add zeros
                    img_list = np.hstack((img_list, empty))

                    # add img
                    tmp_img = get_img(pb_eval.track_test_info.iloc[found_idx], dataset)
                    texty = f'{found_id}_C{cam}_T{track}'
                    if found_id != q_id:
                        tmp_img = cv2.putText(tmp_img, texty, (0, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        tmp_img = cv2.putText(tmp_img, texty, (0, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

                    img_list = np.hstack((img_list, tmp_img))

            if miss:
                # show wrong identification
                cv2.imshow(f'query: {q_id}', img_list)
                if cv2.waitKey() == ord('q'):
                    miss = False
                    img_list = None
                    continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
