from itertools import groupby
import os
import os.path as osp
import re
from scipy.io import loadmat
import numpy as np
import pandas as pd
from .bases import BaseImageDataset
from datasets.dataset_image import Dataset_Image


class PolarBearVidID(BaseImageDataset):
    """
    PolarBearVidID

    Reference:
    PolarBearVidID: A Video-based Re-Identification Benchmark Dataset for Polar Bears

    Class for loading the PolarBearVidID dataset.
    """

    def __init__(self, root='/data/datasets/', split_id=1):
        '''
        Initialize the Dataset Object

        Args:
            root: the root directory of the dataset

        Returns:
            None
        '''
        if split_id == 1:
            print(r"""
            
        ____        __           ____                 _    ___     __________ 
       / __ \____  / /___ ______/ __ )___  ____ _____| |  / (_)___/ /  _/ __ \
      / /_/ / __ \/ / __ `/ ___/ __  / _ \/ __ `/ ___/ | / / / __  // // / / /
     / ____/ /_/ / / /_/ / /  / /_/ /  __/ /_/ / /   | |/ / / /_/ // // /_/ / 
    /_/    \____/_/\__,_/_/  /_____/\___/\__,_/_/    |___/_/\__,_/___/_____/  
                                                                                                                                                        
            """)
            print('Initializing PolarBearVidID Dataset - This may take a while...')
        # Create info file path
        self.root = osp.join(root, 'PolarBearVidID')
        self.track_fold_info_1 = osp.join(self.root, 'track_fold_info_1.csv')
        self.track_fold_info_2 = osp.join(self.root, 'track_fold_info_2.csv')
        self.track_fold_info_3 = osp.join(self.root, 'track_fold_info_3.csv')
        self.track_fold_info_4 = osp.join(self.root, 'track_fold_info_4.csv')
        self.track_fold_info_5 = osp.join(self.root, 'track_fold_info_5.csv')
        self.animal_db = osp.join(self.root, 'animal_db.csv')
        self.track_info = osp.join(self.root, 'track_info.csv')

        track_info = pd.read_csv(self.track_info)
        all_images = self._get_dataset_images(track_info)

        # Check if the info file exists only at first run
        if split_id == 1:
            self._check_before_run(all_images)

        # Load meta data
        track_fold_info = {}
        
        track_fold_info[split_id] = pd.read_csv(getattr(self, f'track_fold_info_{split_id}'))

        # Create dict with names
        test_images = {}
        train_images = {}
        all_images_set = set(all_images)

        # For all Folds get the names of the test tracklets and split them into train and test
        
        test_images[split_id] = self._get_dataset_images(track_fold_info[split_id])

        # get tracklet and id combination of test_images[split_id]
        temp_test_images = test_images[split_id]

        temp_test_images_set = set(temp_test_images)
        train_images[split_id] = list(all_images_set - temp_test_images_set)
        
        train = {}
        num_train_tracklets = {}
        num_train_pids = {}
        num_train_imgs = {}
        test = {}
        num_test_tracklets = {}
        num_test_pids = {}
        num_test_imgs = {}
        train_img = {}
        test_img = {}
        num_imgs_per_tracklet = {}
        total_num = {}
        min_num = {}
        max_num = {}
        avg_num = {}
        num_total_pids = {}
        num_total_tracklets = {}

        
        train[split_id], num_train_tracklets[split_id], num_train_pids[split_id], num_train_imgs[split_id] = \
        self._process_data(train_images[split_id])

        test[split_id], num_test_tracklets[split_id], num_test_pids[split_id], num_test_imgs[split_id] = \
        self._process_data(test_images[split_id])

        train_img[split_id], _, _ = \
        self._extract_1stfeame(train_images[split_id])

        test_img[split_id], _, _ = \
        self._extract_1stfeame(test_images[split_id])

        num_imgs_per_tracklet[split_id] = num_train_imgs[split_id] + num_test_imgs[split_id]

        total_num[split_id] = np.sum(num_imgs_per_tracklet[split_id])
        min_num[split_id] = np.min(num_imgs_per_tracklet[split_id])
        max_num[split_id] = np.max(num_imgs_per_tracklet[split_id])
        avg_num[split_id] = np.mean(num_imgs_per_tracklet[split_id])

        num_total_pids[split_id] = max(num_train_pids[split_id], num_test_pids[split_id])
        num_total_tracklets[split_id] = num_train_tracklets[split_id] +  + num_test_tracklets[split_id]

        if split_id == 1:
            print("=> PolarBearVidID loaded")
            print("Dataset statistics:")
            print("  ------------------------------------------")
            print("  Subset    | # Ids | # Tracklets | # Images")
            print("  ------------------------------------------")

        
        print("  {}. Fold:".format(split_id))
        print("  ------------------------------------------")
        print("  Train {}   | {:5d} | {:8d} | {:8d}".format(split_id, num_train_pids[split_id], num_train_tracklets[split_id], np.sum(num_train_imgs[split_id])))
        print("  Test  {}   | {:5d} | {:8d} | {:8d}".format(split_id, num_test_pids[split_id], num_test_tracklets[split_id], np.sum(num_test_imgs[split_id])))
        print("  ------------------------------------------")
        print("  Total     | {:5d} | {:8d} | {:8d}".format(num_total_pids[split_id], num_total_tracklets[split_id], total_num[split_id]))
        print("  ------------------------------------------")
        print("  Number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num[split_id], max_num[split_id], avg_num[split_id]))

        if(len(train_images[split_id])+len(test_images[split_id]) == len(all_images)):
            print("  Image numbers match")
        else:
            print("  WARNING: Image numbers do not match")
        print("  ------------------------------------------")



        self.train = train[split_id]
        self.test = test[split_id]

        self.train_img = train_img[split_id]
        self.test_img = test_img[split_id]

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_test_pids, self.num_test_imgs, self.num_test_cams, self.num_test_vids = self.get_imagedata_info(
            self.test)
        
    def _getIDX_from_names(self, names):
        """
        Get the ID and the tracklet number from the name
        Args:
            names: list of names
        Returns:
            idx: list of tuples (ID, tracklet number)
        """
        idx = []
        for name in names:
            entry = (int(name[0:2]),int(name[7:9]))
            if entry not in idx:
                idx.append(entry) # ID and tracklet number tuple
        return idx
    
    def _check_before_run(self, all_images):
        """
        Check if all files are available before going deeper
        """
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.track_fold_info_1):
            raise RuntimeError("'{}' is not available".format(self.track_fold_info_1))
        if not osp.exists(self.track_fold_info_2):
            raise RuntimeError("'{}' is not available".format(self.track_fold_info_2))
        if not osp.exists(self.track_fold_info_3):
            raise RuntimeError("'{}' is not available".format(self.track_fold_info_3))
        if not osp.exists(self.track_fold_info_4):
            raise RuntimeError("'{}' is not available".format(self.track_fold_info_4))  
        if not osp.exists(self.track_fold_info_5):
            raise RuntimeError("'{}' is not available".format(self.track_fold_info_5))
        if not osp.exists(self.animal_db):
            raise RuntimeError("'{}' is not available".format(self.animal_db))
        if not osp.exists(self.track_info):
            raise RuntimeError("'{}' is not available".format(self.track_info))
        for image in all_images:
            if not osp.exists(str(image)):
                raise RuntimeError("'{}' is not available".format(image))

    def _get_dataset_images(self, track_info):
        '''
        Gets all the dataset images in PolarBearVidID

        Args:
            track_info: info file for all included dataset images
        
        Returns:
            names: list of all the dataset images in track info
        '''
        images = []
        for index, row in track_info.iterrows():
            for i in range(row["start"], row["end"] + 1):
                name = str(row["id"]).zfill(3) + 'C' + str(row["cam"]).zfill(2) + 'T' + str(row["tracklet"]).zfill(3) + 'F' + str(i).zfill(3) + '.jpg'
                images.append(Dataset_Image(
                    name,
                    os.path.join(self.root, str(row["id"]).zfill(3), name),
                    row["tracklet"],
                    row["id"],
                    row["cam"],
                    i))
        return images
    
    def _get_fold_names(self, fold_info):
        '''
        Gets the names of all the images for each fold

        Args:
            fold_info: dataframe containing the information of each fold
        
        Returns:
            fold_names: list of names of all the images in the folder
        '''
        fold_names = []
        for index, row_series in fold_info.iterrows():
            for i in range(row_series["start"], row_series["end"] + 1):
                fold_names.append(str(row_series["id"]).zfill(3) + 'C' + str(row_series["cam"]).zfill(2) + 'T' + str(row_series["tracklet"]).zfill(3) + 'F' + str(i).zfill(3) + '.jpg')
        return fold_names

    def _process_data(self, images):
        '''
        Processes the data and returns the tracklets, the number of tracklets the number of pids and the number of images per tracklet

        Args:
            images: list of all the images
        Returns:
            tracklets: list of tracklets
            num_tracklets: number of tracklets
            num_pids: number of pids
            num_imgs_per_tracklet: number of images per tracklet
        '''
        tracklets = []
        num_imgs_per_tracklet = []
        idlist = []

        images_sorted = sorted(images, key=lambda x: x.uidt)
        images_grouped = [list(group) for key, group in groupby(images_sorted, key=lambda x: x.uidt)]
        for individual_tracklet in images_grouped:
            image_list = []
            camid = -1
            id = -1
            individual_tracklet = sorted(individual_tracklet, key=lambda x: x.frame)
            for frame in individual_tracklet:
                if camid == -1:
                    camid = frame.camera -1 # camera 1 is 0 according to previous code
                if id == -1:
                    id = frame.id
                image_list.append(frame.path)
            idlist.append(id)
            image_tuple = tuple(image_list)
            tracklets.append((image_tuple, id, camid, 1))
            num_imgs_per_tracklet.append(len(image_list))
        
        num_tracklets = len(tracklets)
        num_pids = len(set(idlist))

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
    
    def _extract_1stfeame(self, images):
        '''
        Extracts the first frame of each tracklet
        Args:
            images: list of all the images
        Returns:
            imgs: list of all the first frames, the id and the camera
            num_imgs: number of all images
            num_pids: number of all pids
        '''
        imgs = []
        idlist = []
        images_sorted = sorted(images, key=lambda x: x.uidt)
        images_grouped = [list(group) for key, group in groupby(images_sorted, key=lambda x: x.uidt)]
        for individual_tracklet in images_grouped:
            individual_tracklet = sorted(individual_tracklet, key=lambda x: x.frame)
            idlist.append(individual_tracklet[0].id)
            imgs.append(([individual_tracklet[0].path], individual_tracklet[0].id, individual_tracklet[0].camera - 1))
            
        num_pids = len(set(idlist))
        num_imgs = len(imgs)

        return imgs, num_imgs, num_pids
