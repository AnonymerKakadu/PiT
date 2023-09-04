'''
    File name: pb_evaluation.py
    Author: Richard Dirauf
    Python Version: 3.8
'''

import os
from typing import List, Tuple
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy.spatial.distance import cdist
import sys

sys.path.append("..")
from gltr_error import GLTRError
from evaluation.re_ranking.re_ranking_ranklist import re_ranking


class PBEvaluation:
    """Class to handle the evaluation of polar bear test. Better useable than the matlab files.\n

    Attributes:
        * track_test_info -- track_test_info as a pandas object.
        * feat_files -- List of names of the features files.
        * feat_dir -- Path to the feature directory.
        * sorted_idx_per_epoch -- Sorted index for each query over each epoch (1210, 121)
        * re_rank -- Should re_ranking be applied?
        * zoos -- All current zoos in the dataset.
        * map_per_zoo -- variable containing the metric for each zoo.
        * r1_per_zoo -- variable containing the metric for each zoo.
    """

    track_test_info = None
    feat_files = None
    feat_dir = None
    sorted_idx_per_epoch = []
    re_rank = False
    simpl_re_rank_count = 0

    zoos = None  # [['Nürnberg', 0, 1], ['Berlin', 2, 3], ['Schönbrunn', 4, 5], ['Mulhouse', 6, 7]]
    map_per_zoo = {}
    r1_per_zoo = {}

    def __init__(self, dataset_dir: str, features_dir: str, csv_file: str = 'track_test_info.csv',
                 zoo_ids: List = None, re_rank: str = None) -> None:
        """Initialize the evaluation object.

        Args:
            dataset_dir (str): Path to the dataset.
            features_dir (str): Path to the feature directory.
            csv_file (str): Info file with the track infos.
            zoo_ids (List): Listo of IDs that should be evaluated. If none, all are evaluated.
            re_rank (str): Should re_ranking be applied? (None, simple, adv)

        Raises:
            GLTRError: If an error occured.
        """
        # check values
        track_test_file = os.path.join(dataset_dir, csv_file)

        if not os.path.isdir(dataset_dir) or not os.path.isfile(track_test_file):
            raise GLTRError('PBEvaluation', 'The dataset does not exist.')
        if not os.path.isdir(features_dir):
            raise GLTRError('PBEvaluation', 'The feature dir does not exist.')
        if re_rank is not None and re_rank not in ['simple', 'adv']:
            raise GLTRError('PBEvaluation', 're-rank not defined.')

        # load db
        self.track_test_info = pd.read_csv(track_test_file)

        # load zoos
        animal_db = pd.read_csv(os.path.join(dataset_dir, 'animal_db.csv'))
        animal_db = animal_db.groupby('zoo')['id'].apply(lambda x: list(x)).reset_index()
        self.zoos = list(animal_db.apply(lambda x: [x['zoo']] + x['id'], axis=1))

        if zoo_ids is not None:
            self.track_test_info = self.track_test_info[self.track_test_info['id'].isin(zoo_ids)]
            # only take the 1 zoo
            self.zoos = [z for z in self.zoos if z[1:] == zoo_ids]

        # read features from dir
        self.feat_files = os.listdir(features_dir)
        self.feat_files.sort()
        self.feat_dir = features_dir
        self.re_rank = re_rank

        # init dirs
        self.map_per_zoo = {}
        self.r1_per_zoo = {}
        self.sorted_idx_per_epoch = []

    def eval_feat_dir(self) -> List:
        """Computes CMC and mAP for all feature files.

        Returns:
            epochs, mAPs, rank1s, mAPs_per_id, CMCs_per_id, mAPs_per_zoo, r1s_per_zoo: In same order.
        """
        # def save values
        epochs, mAPs, rank1s = [], [], []
        mAPs_per_id, CMCs_per_id = {}, {}
        mAPs_per_zoo, r1s_per_zoo = {}, {}
        for i in self.track_test_info['id'].unique():
            mAPs_per_id[i] = []
            CMCs_per_id[i] = []
        for z in self.zoos:
            mAPs_per_zoo[z[0]] = []
            r1s_per_zoo[z[0]] = []

        # go through feature files
        for f in self.feat_files:
            # get epoch
            epoch = int(f.split('_')[2])
            epochs.append(epoch)

            # evaluate feat
            mAP, CMC, mAP_per_id, CMC_per_id = self.eval_single_feat_file(epoch, f)
            mAPs.append(mAP)
            rank1s.append(CMC[0])
            for i in self.track_test_info['id'].unique():
                mAPs_per_id[i].append(mAP_per_id[i])
                CMCs_per_id[i].append(CMC_per_id[i])
            for z in self.zoos:
                mAPs_per_zoo[z[0]].append(self.map_per_zoo[z[0]])
                r1s_per_zoo[z[0]].append(self.r1_per_zoo[z[0]])

        # find max
        imap = mAPs.index(max(mAPs))
        irank1 = rank1s.index(max(rank1s))
        if imap == irank1:
            print(f'\nMax. mAP ({round(max(mAPs), 4)}) and rank-1 ({max(rank1s)}) at epoch {epochs[imap]}\n')
        else:
            print(f'\nMax. mAP ({round(max(mAPs), 4)}) at epoch {epochs[imap]}')
            print(f'Max. rank-1 ({round(max(rank1s), 4)}) at epoch {epochs[irank1]}\n')

        return epochs, mAPs, rank1s, mAPs_per_id, CMCs_per_id, mAPs_per_zoo, r1s_per_zoo

    @staticmethod
    def process_box_feat(features: np.ndarray) -> np.ndarray:
        """Normalize the features.

        Args:
            features (ndarray): Feature matrix.

        Returns:
            ndarray: Normalized feature matrix.
        """
        sum_val = np.sqrt(np.sum(features**2, axis=0))
        for i, f in enumerate(features):
            features[i] = f / sum_val
        return features

    def eval_single_feat_file(self, epoch: int, feat_file: str) -> List:
        """Computes CMC and mAP for a single feature file.

        Args:
            epoch (int): Epoch number.
            feat_file (str): Name of the feature file.

        Returns:
            mAP, CMC, mAP_per_id, CMC_per_id: Of one epoch.
        """
        # read feature file
        box_feature_test = np.loadtxt(os.path.join(self.feat_dir, feat_file))


        # assert len(box_feature_test) == len(self.track_test_info)
        box_feature_test = box_feature_test.T  # transpose data, one feature vector per column

        # normalize features
        video_feat_test = self.process_box_feat(box_feature_test)

        # compute the distance between all test samples, euclidean by default
        distance = cdist(video_feat_test.T, video_feat_test.T)

        # calc the restult, distance matrix is symmetrical so no transpose
        assert (distance.T == distance).all()
        mAP, CMC, mAP_per_id, CMC_per_id = self.evaluation(distance)

        # print restults for current epoch
        print(f'\nEpoch {epoch}')
        print('-----------------------------')
        print(f'Overall:   mAP = {round(mAP, 4)}, r1 precision = {round(CMC[0], 4)}')

        print('Divided by:  ID      mAP     rank-1')
        ids = self.track_test_info['id'].unique()
        ids.sort()
        for i in ids:
            print(f'             {i}      {str(mAP_per_id[i])[:4]}      {str(CMC_per_id[i][0])[:4]}')
        for z in self.map_per_zoo:
            print(f'         {z}      {str(self.map_per_zoo[z])[:4]}      {str(self.r1_per_zoo[z])[:4]}')
        if self.re_rank == 'simple':
            print(f'Re-ranks without majority: {self.simpl_re_rank_count}')
            self.simpl_re_rank_count = 0

        return mAP, CMC, mAP_per_id, CMC_per_id

    def evaluation(self, distance: np.ndarray) -> List:
        """Computes CMC and mAP given the pairwise distance.

        Args:
            distance (ndarray): Pairwise distance between the features.

        Returns:
            mAP, CMC, mAP_per_id, CMC_per_id: Of one epoch.
        """
        # define values
        ap = np.zeros(len(self.track_test_info))  # average precision
        CMC = []  # Cumulative Matching Characteristics

        ap_per_id, mAP_per_id = {}, {}  # avaerage precision per id
        cmc_per_id, CMC_per_id = {}, {}  # Cumulative Matching Characteristics per id
        q_counter = 0  # count querys
        for i in self.track_test_info['id'].unique():
            ap_per_id[i] = []
            cmc_per_id[i] = []

        # go through the queries, each test element is a query once, rest is gallery
        for k, score in enumerate(distance):
            # get info for query
            try:
                q_label = self.track_test_info['id'].loc[[k]].values[0]
                q_track = self.track_test_info['tracklet'].loc[[k]].values[0]
            except KeyError:  # skip this query
                continue

            # differentiate between hits and misses
            good_image = self.track_test_info[self.track_test_info['id'] == q_label]  # get elements with the same label
            good_image = good_image[good_image['tracklet'] != q_track]  # remove the query

            # apply advanced re-ranking
            if self.re_rank == 'adv':
                # split query and gallery
                q_q_dist = np.reshape(score[k], (1, 1))
                q_g_dist = np.reshape(np.delete(score, k), (1, len(score) - 1))
                g_g_dist = np.delete(np.delete(distance, k, axis=0), k, axis=1)

                # re rank
                final_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=3, k2=2, lambda_value=0.1)
                # add query with 0 again
                final_dist = np.insert(final_dist[0], k, 0.)

                # checks
                assert len(final_dist) == len(score)
                assert final_dist[k] == score[k]

                score = final_dist

            # sort distance after smallest one
            sorted_idx = np.argsort(score)

            # remove idx that are not in table
            if len(sorted_idx) != len(self.track_test_info):
                sorted_idx = sorted_idx[np.isin(sorted_idx, self.track_test_info.index)]

            assert sorted_idx[0] == self.track_test_info.loc[[k]].index
            self.sorted_idx_per_epoch.append(sorted_idx)
            sorted_idx = sorted_idx[1:]  # remove first instance, is query

            # calc ap and r1 score
            ap_query, cmc_query = self.compute_AP(good_image, sorted_idx)
            ap[q_counter] = ap_query
            q_counter += 1
            CMC.append(cmc_query)

            # diff between the ids
            ap_per_id[q_label].append(ap_query)
            cmc_per_id[q_label].append(cmc_query)

        # calc the mean over the whole epoch
        self.r1 = [q[0] for q in CMC]
        CMC = np.array(CMC)

        CMC = np.sum(CMC, axis=0) / len(CMC)  # rank accuracies
        mAP = np.sum(ap) / len(ap)  # mean avaerage presicion

        # calc mAP and CMC divided by the different ids
        for key, value in ap_per_id.items():
            mAP_per_id[key] = np.sum(value) / len(value)
        for key, value in cmc_per_id.items():
            CMC_per_id[key] = np.sum(np.array(value), axis=0) / len(value)

        # calc map and r1 divided by zoo
        for z in self.zoos:
            value_ap, value_cmc = [], []
            
            # go through individuals
            for j in range(1, len(z)):
                value_ap += ap_per_id[z[j]]
                value_cmc += cmc_per_id[z[j]]

            # compute map, r1
            self.map_per_zoo[z[0]] = np.sum(value) / len(value)
            value = np.array(value_cmc).T[0]
            self.r1_per_zoo[z[0]] = np.sum(value) / len(value)

        # return ap and r1 per query for the improvement calc
        self.ap = ap

        return mAP, CMC, mAP_per_id, CMC_per_id

    def compute_AP(self, good_image: DataFrame, sorted_idx: np.ndarray) -> Tuple:
        """Computes CMC and mAP of one query.

        Args:
            good_image (DataFrame): Images with the same label in the gallery.
            sorted_idx (ndarray): Index array from smallest to highest distance.

        Returns:
            ap, cmc: For query.
        """
        # def values
        cmc = np.zeros(len(sorted_idx))  # set cmc to 120
        ngood = len(good_image)  # set to number of possible hits
        old_recall = 0
        old_precision = 1.0
        ap = 0
        good_now = 0  # counter of already found good images

        if self.re_rank == 'simple':
            sorted_idx = self.apply_simple_re_rank(sorted_idx)

        # go through the sorted index array
        for n, idx in enumerate(sorted_idx):

            if idx in good_image.index:
                # this gallery seq and the query are from the same id
                cmc[n:] = 1  # CMC is 1 from this seq on
                good_now += 1

            # calculate recall and precision
            recall = good_now / ngood
            precision = good_now / (n + 1)

            # calculate average precision
            ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
            old_recall = recall
            old_precision = precision

            # found all true positives
            if good_now == ngood:
                break

        return ap, cmc

    def apply_simple_re_rank(self, sorted_idx: np.ndarray) -> np.ndarray:
        """Applys a simple re-ranking. The correct id is chosen through the majority of the first k elements.

        Args:
            sorted_idx (ndarray): Index array from smallest to highest distance.

        Returns:
            sorted_idx: New sorted idx.
        """
        # def k and conter array and ids array
        k = 5
        count = np.zeros(8)
        inital_ids = []

        # go through the sorted index array
        for n, idx in enumerate(sorted_idx):
            # get id
            g_id = self.track_test_info['id'].loc[[idx]].values[0]
            inital_ids.append(g_id)

            if n < k:  # first k elements
                count[g_id] += 1

        # find majority
        majority = np.sort(count)[::-1]
        if majority[0] == majority[1]:
            # no majority could be choosen, sorted idx stays
            self.simpl_re_rank_count += 1
            return sorted_idx
        elif majority[0] == k:
            # or first k elements are the same, no re-ranking nessesary
            return sorted_idx

        # get id of majority
        id_major = np.argsort(count)[-1]

        # re-rank by swapping elements of the end to the front
        for n in range(k):
            g_id = inital_ids[n]
            if g_id != id_major:
                swap_idx = len(inital_ids) - inital_ids[::-1].index(id_major) - 1
                sorted_idx[n], sorted_idx[swap_idx] = sorted_idx[swap_idx], sorted_idx[n]
                inital_ids[n], inital_ids[swap_idx] = inital_ids[swap_idx], inital_ids[n]

        return sorted_idx
