import os
import os.path as osp
import re
from scipy.io import loadmat
import numpy as np
import pandas as pd

from .bases import BaseImageDataset

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
            min_seq_len: the minimum length of a tracklet

        Returns:
            None
        '''

        # Create info file path
        self.root = osp.join(root, 'PolarBearVidID')
        self.track_fold_info_1 = osp.join(self.root, 'track_fold_info_1.csv')
        self.track_fold_info_2 = osp.join(self.root, 'track_fold_info_2.csv')
        self.track_fold_info_3 = osp.join(self.root, 'track_fold_info_3.csv')
        self.track_fold_info_4 = osp.join(self.root, 'track_fold_info_4.csv')
        self.track_fold_info_5 = osp.join(self.root, 'track_fold_info_5.csv')
        self.animal_db = osp.join(self.root, 'animal_db.csv')
        self.track_info = osp.join(self.root, 'track_info.csv')

        # Check if the info file exists
        self._check_before_run()

        # Load meta data
        track_fold_info = {}
        for i in range(1, 6):
            track_fold_info[i] = pd.read_csv(getattr(self, f'track_fold_info_{i}'))

        animal_db = pd.read_csv(self.animal_db)
        track_info = pd.read_csv(self.track_info)
        all_names = self._get_names(self.root)

        # Create dict with names
        test_names = {}
        query_names = {}
        gallery_names = {}

        # For all Folds get the names of the test tracklets and split them into query and gallery
        for i in range(1, 6):
            test_names[i] = self._get_fold_names(track_fold_info[i])
            # Get individual tracklets for the test tracklets and split them into query and gallery 80:20(tupel (id,tracklet))
            # TODO: Query should get assigned dynamically
            individual_tracklets = self._getIDX_from_names(test_names[i])
            split_index = round(len(individual_tracklets) * 0.2)
            gallery_list = []
            query_list = []
            queryIDX = individual_tracklets[:split_index]
            galleryIDX = individual_tracklets[split_index:]
            for name in test_names[i]:
                match = re.search(r'(\d+)C.+T(\d+)F', name)
                if match:
                    if (int(match.group(1)),int(match.group(2))) in queryIDX:
                        query_list.append(name)
                    else:
                        gallery_list.append(name)
            query_names[i] = query_list
            gallery_names[i] = gallery_list


        train_names = {}
        for i in range(1, 6):
            train_names[i] = list(set(all_names) - set(test_names[i]))
        


        # Sanity check
        print("-"*50)
        print("Evaluating fold values:")
        for i in range(1, 6):
            print("Fold: ", i)
            print("Allnames: ", len(all_names))
            print("Trainnames: ", len(train_names[i]))
            print("Testnames: ", len(test_names[i]))
            print("Querynames: ", len(query_names[i]))
            print("Gallerynames: ", len(gallery_names[i]))
            if(len(train_names[i])+len(test_names[i]) == len(all_names)):
                print("Image numbers match")
            else:
                print("WARNING: Image numbers do not match")
            print("-"*50)

        separator = '-' * 50
        print(separator)
        print("Evaluating fold values:")
        for i in range(1, 6):
            print(f"Fold: {i}")
            print(f"Allnames: {len(all_names):>10}")
            print(f"Trainnames: {len(train_names[i]):>9}")
            print(f"Testnames: {len(test_names[i]):>10}")
            print(f"Querynames: {len(query_names[i]):>10}")
            print(f"Gallerynames: {len(gallery_names[i]):>8}")
            if(len(train_names[i])+len(test_names[i]) == len(all_names)):
                print("Image numbers match")
            else:
                print("WARNING: Image numbers do not match")
            print(separator)

            

        # id cam tracklet start end

        # Bauen von train, num_train_tracklets, num_train_pids, num_train_imgs
        # train = Trainingsdaten
        # train = array mit tracklet elementnamen = trackletnummer inhalt pfad zu allen bildern im tracklet 
        # tuple(img_paths) + pid + camid + 1
        # Get rows of track_info that are not in track_fold_info_x
        # TODO: Hier ist noch ein Fehler
        #test_fold_1 = self._get_df_dif(track_fold_info_1,track_info)
        #test_fold_2 etc


        
        # train, num_train_tracklets, num_train_pids, num_train_imgs = \
        #   self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)


        # num_train_tracklets = länge von train
        # num_train_pids = anzahl der ids in train TODO
        # num_train_imgs = liste wie viele bilder in jedem tracklet sind
        
        # query = Testdaten
        # gallery = Testdaten TODO Unterschied zu query?


        # train_names = self._get_names(self.train_name_path)
        # test_names = self._get_names(self.test_name_path)
        # track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        # track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        # query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        # query_IDX -= 1 # index from 0
        # track_query = track_test[query_IDX,:]
        # gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        # track_gallery = track_test[gallery_IDX,:]
        # track_gallery = track_test

        
        # _extract 1st frame  returns a list: [img_path first jpg, pid, camid]
        # _process_data returns a list: [img_path, pid, camid, tracklet_id]

        '''
        
        [start, end, label, cam]
        array([[     1,     16,      1,      1],
            [    17,     95,      1,      1],
            [    96,    110,      1,      1],
            ...,
            [509821, 509844,   1499,      5],
            [509845, 509864,   1499,      5],
            [509865, 509914,   1499,      5]], dtype=int32)

        track_test
        array([[     1,     24,     -1,      1],
            [    25,     34,     -1,      1],
            [    35,     49,     -1,      1],
            ...,
            [680962, 680984,   1500,      1],
            [680985, 681071,   1500,      1],
            [681072, 681089,   1500,      5]], dtype=int32)
            
        query_IDX - tracklet numbers??
        array([ 4129,  4137,  4145, ..., 12176, 12178, 12179], dtype=uint16)

        train:
        tuple 0000
                    tuple 0: 00: 'data/mars/bbox_train/0001/0001C1T0001F001.jpg'
                             01: 'data/mars/bbox_train/0001/0001C1T0001F002.jpg'
                             02: 'data/mars/bbox_train/0001/0001C1T0001F003.jpg'
                          1:
                          2:
                          3:
        '''

        

        train_1, num_train_tracklets_1, num_train_pids_1, num_train_imgs_1 = \
          self._process_data(train_names, track_info, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        train_img, _, _ = \
          self._extract_1stfeame(train_names, track_train, home_dir='bbox_train', relabel=True)

        query_img, _, _ = \
          self._extract_1stfeame(test_names, track_query, home_dir='bbox_test', relabel=False)

        gallery_img, _, _ = \
          self._extract_1stfeame(test_names, track_gallery, home_dir='bbox_test', relabel=False)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        total_num = np.sum(num_imgs_per_tracklet)
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> PolarBearVidID loaded")
        print("Dataset statistics:")
        print("  -----------------------------------------")
        print("  subset    | # ids | # tracklets | # images")
        print("  -----------------------------------------")
        print("  First Fold:")
        print("  -----------------------------------------")
        print("  train_1   | {:5d} | {:8d} | {:8d}".format(num_train_pids, num_train_tracklets, np.sum(num_train_imgs)))
        print("  query_1   | {:5d} | {:8d} | {:8d}".format(num_query_pids, num_query_tracklets, np.sum(num_query_imgs)))
        print("  gallery_1 | {:5d} | {:8d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets, np.sum(num_gallery_imgs)))
        print("  -----------------------------------------")
        print("  total    | {:5d} | {:8d} | {:8d}".format(num_total_pids, num_total_tracklets, total_num))
        print("  -----------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  -----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.train_img = train_img
        self.query_img = query_img
        self.gallery_img = gallery_img

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
    
    def _check_before_run(self):
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

    def _get_names(self, fpath):
        '''
        Gets the names of all the images in the folder

        Args:
            fpath: path to the folder
        
        Returns:
            names: list of names of all the images in the folder
        '''
        names = []
        for dirs in os.walk(fpath):
            for dir in dirs[1]:
                dir_path = os.path.join(fpath, dir)
                for file in os.listdir(dir_path):
                    names.append(file)
        return names
    
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

    def _process_data(self, names, track_info):
        '''
        Processes the data and returns the tracklets, the number of tracklets and the number of pids

        Args:
            names: list of names of all the images
        Returns:
            tracklets: list of tracklets
            num_tracklets: number of tracklets
            num_pids: number of pids
        '''
        pid_tid_tuple = self._getIDX_from_names(names)
        # Get the overall loaded tracklet count
        num_tracklets = len(set(pid_tid_tuple))

        # Get the overall loaded pid count
        # num_pids = len(set(track_info.iloc[:, 0].values)) That works for csv files but not for sorted names in query and gallery
        num_pids = len(set(pid_tid_tuple[0]))

        num_tracklets = len(names)
        pid_list = []
        for name in names:
            pid_list.append(name.split('C')[0])
        pid_list = list(set(pid_list))
        num_pids = len(pid_list)
        tracklets = []

        for name in names:
            tracklet_id = name.split("T")[1][:3]
            found = False
            for tracklet in tracklets:
                if tracklet[0].split("T")[1][:3] == tracklet_id:
                    tracklet.append(name)
                    found = True
                    break
            if not found:
                tracklets.append([name])
        
        # tracklets ist ne liste von tracklets die erste stelle ist ein tracklet mit den einzelnen bildern die zweite stelle ist die 


        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
    
    def _process_data_old(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid, 1))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def _extract_1stfeame(self, names, meta_data, home_dir=None, relabel=False):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        imgs = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_name = names[start_index-1]

            # append image names with directory information
            img_path = osp.join(self.root, home_dir, img_name[:4], img_name)
            
            imgs.append(([img_path], pid, camid))

        num_imgs = len(imgs)

        return imgs, num_imgs, num_pids
