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

    def __init__(self, root='/data/datasets/', min_seq_len=0):
        '''
        Initialize the Dataset Object

        Args:
            root: the root directory of the dataset

        Returns:
            None
        '''
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

        # Check if the info file exists
        self._check_before_run(all_images)

        fold_amount = 5 # TODO change to 5
        # Load meta data
        track_fold_info = {}
        for i in range(1, fold_amount+1):
            track_fold_info[i] = pd.read_csv(getattr(self, f'track_fold_info_{i}'))

        # Create dict with names
        test_images = {}
        train_images = {}
        query_images = {}
        gallery_images = {}
        all_images_set = set(all_images)

        # For all Folds get the names of the test tracklets and split them into query and gallery
        for i in range(1, fold_amount+1):
            print(f"Loading fold {i}")
            test_images[i] = self._get_dataset_images(track_fold_info[i])
    
            # get tracklet and id combination of test_images[i]
            temp_test_images = test_images[i]

            temp_test_images_set = set(temp_test_images)
            train_images[i] = list(all_images_set - temp_test_images_set)

            # TODO: Query should get assigned dynamically
            setlist = []
            for instance in test_images[i]:
                setlist.append(instance.get_individual())
            setlist = list(set(setlist))
            split_index = round(len(setlist) * 0.2)

            querylist = []
            gallerylist = []
            for instance in test_images[i]:
                if instance.get_individual() in setlist[:split_index]:
                    querylist.append(instance)
                else:
                    gallerylist.append(instance)

            gallery_images[i] = gallerylist
            query_images[i] = querylist
        


        # Sanity check for debugging
        '''
        separator = '-' * 50
        print(separator)
        print("Evaluating fold values:")

        for i in range(1, fold_amount+1):
            print("Fold: ", i)
            print(f"All images: {len(all_images):>9}")
            print(f"Train images: {len(train_images[i]):>9}")
            print(f"Test images: {len(test_images[i]):>9}")
            print(f"Query images: {len(query_images[i]):>7}")
            print(f"Gallery images: {len(gallery_images[i]):>5}")
            if(len(train_images[i])+len(test_images[i]) == len(all_images)):
                print("Image numbers match")
            else:
                print("WARNING: Image numbers do not match")
            print(separator)
        '''

        train = {}
        num_train_tracklets = {}
        num_train_pids = {}
        num_train_imgs = {}
        query = {}
        num_query_tracklets = {}
        num_query_pids = {}
        num_query_imgs = {}
        gallery = {}
        num_gallery_tracklets = {}
        num_gallery_pids = {}
        num_gallery_imgs = {}
        train_img = {}
        query_img = {}
        gallery_img = {}
        num_imgs_per_tracklet = {}
        total_num = {}
        min_num = {}
        max_num = {}
        avg_num = {}
        num_total_pids = {}
        num_total_tracklets = {}

        for i in range(1, fold_amount+1):
            train[i], num_train_tracklets[i], num_train_pids[i], num_train_imgs[i] = \
            self._process_data(train_images[i])

            query[i], num_query_tracklets[i], num_query_pids[i], num_query_imgs[i] = \
            self._process_data(query_images[i])

            gallery[i], num_gallery_tracklets[i], num_gallery_pids[i], num_gallery_imgs[i] = \
            self._process_data(gallery_images[i])

            train_img[i], _, _ = \
            self._extract_1stfeame(train_images[i])

            query_img[i], _, _ = \
            self._extract_1stfeame(query_images[i])

            gallery_img[i], _, _ = \
            self._extract_1stfeame(gallery_images[i])

            num_imgs_per_tracklet[i] = num_train_imgs[i] + num_gallery_imgs[i] + num_query_imgs[i]

            total_num[i] = np.sum(num_imgs_per_tracklet[i])
            min_num[i] = np.min(num_imgs_per_tracklet[i])
            max_num[i] = np.max(num_imgs_per_tracklet[i])
            avg_num[i] = np.mean(num_imgs_per_tracklet[i])

            num_total_pids[i] = max(num_train_pids[i], num_query_pids[i])
            num_total_tracklets[i] = num_train_tracklets[i] + num_gallery_tracklets[i] + num_query_tracklets[i]

        print("=> PolarBearVidID loaded")
        print("Dataset statistics:")
        print("  ------------------------------------------")
        print("  Subset    | # Ids | # Tracklets | # Images")
        print("  ------------------------------------------")

        for i in range(1, fold_amount+1):
            print("  {}. Fold:".format(i))
            print("  ------------------------------------------")
            print("  Train {}   | {:5d} | {:8d} | {:8d}".format(i, num_train_pids[i], num_train_tracklets[i], np.sum(num_train_imgs[i])))
            print("  Query {}   | {:5d} | {:8d} | {:8d}".format(i, num_query_pids[i], num_query_tracklets[i], np.sum(num_query_imgs[i])))
            print("  Gallery {} | {:5d} | {:8d} | {:8d}".format(i, num_gallery_pids[i], num_gallery_tracklets[i], np.sum(num_gallery_imgs[i])))
            print("  ------------------------------------------")
            print("  Total     | {:5d} | {:8d} | {:8d}".format(num_total_pids[i], num_total_tracklets[i], total_num[i]))
            print("  ------------------------------------------")
            print("  Number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num[i], max_num[i], avg_num[i]))

            if(len(train_images[i])+len(test_images[i]) == len(all_images)):
                print("  Image numbers match")
            else:
                print("  WARNING: Image numbers do not match")
            print("  ------------------------------------------")


        # TODO for now fixed to the first fold later get all folds

        self.train = train[1]
        self.query = query[1]
        self.gallery = gallery[1]

        self.train_img = train_img[1]
        self.query_img = query_img[1]
        self.gallery_img = gallery_img[1]

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)
        
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
                    camid = frame.camera
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
            imgs.append(([individual_tracklet[0].path], individual_tracklet[0].id, individual_tracklet[0].camera))
            
        num_pids = len(set(idlist))
        num_imgs = len(imgs)

        return imgs, num_imgs, num_pids
