import os
import os.path
import os.path as osp

import pandas as pd

def get_all_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(file)
    return file_list

#get current folder path
print("Programm start")
print("______________________________________________________")
folder_path = os.path.dirname(os.path.realpath(__file__))
#bbox_test folder path
folder_path = os.path.join(folder_path, 'data/PolarbearVidID')

track_fold_info_1 = osp.join(folder_path, 'track_fold_info_1.csv')
track_fold_info_2 = osp.join(folder_path, 'track_fold_info_2.csv')
track_fold_info_3 = osp.join(folder_path, 'track_fold_info_3.csv')
track_fold_info_4 = osp.join(folder_path, 'track_fold_info_4.csv')
track_fold_info_5 = osp.join(folder_path, 'track_fold_info_5.csv')

'''id,cam,tracklet,start,end
0,1,0,0,99
0,1,4,0,99
0,1,12,0,99
0,1,15,0,99
0,1,19,0,99
0,2,24,0,99
0,1,30,17,99
0,1,31,0,74'''

fold_info = {}
fold_info[0] = pd.read_csv(track_fold_info_1)
fold_info[1] = pd.read_csv(track_fold_info_2)
fold_info[2] = pd.read_csv(track_fold_info_3)
fold_info[3] = pd.read_csv(track_fold_info_4)
fold_info[4] = pd.read_csv(track_fold_info_5)

for i in range(len(fold_info)):
    print("fold: {}".format(i+1))
    print("total tracklets: {}".format(fold_info[i].shape[0]))
    for id in range(13):
        for cam in range(3):
            print("total tracklets with id {} and cam {}: {}".format(id, cam+1, len(fold_info[i][(fold_info[i]['id'] == id) & (fold_info[i]['cam'] == cam+1)])))
    
    print("______________________________________________________")

print("Programm end")