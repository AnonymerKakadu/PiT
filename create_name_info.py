import os

def get_all_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(file)
    return file_list

#get current folder path
print("Programm start")
folder_path = os.path.dirname(os.path.realpath(__file__))
#bbox_test folder path
folder_path = os.path.join(folder_path, 'data/mars')
bbox_test_path = os.path.join(folder_path, 'bbox_test')
bbox_train_path = os.path.join(folder_path, 'bbox_train')

test_file_list = get_all_files_in_folder(bbox_test_path)
# remove .DS_Store file
test_file_list = [file for file in test_file_list if file != '.DS_Store']
with open(folder_path + '/info/train_name.txt', 'w') as f:
    for file in test_file_list:
        f.write(file + '\n')


train_file_list = get_all_files_in_folder(bbox_train_path)
# remove .DS_Store file
train_file_list = [file for file in test_file_list if file != '.DS_Store']
with open(folder_path + '/info/test_name.txt', 'w') as f:
    for file in train_file_list:
        f.write(file + '\n')



# make sure image names correspond to the same person
pnames = [img_name[:4] for img_name in test_file_list]
assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

# make sure all images are captured under the same camera
camnames = [img_name[5] for img_name in train_file_list]
assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"