import os
import cv2

# Helper function to flip all images in polar bear dataset
def get_all_jpgs_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                file_list.append(os.path.join(root, file))
    return file_list

#get current folder path
print("Programm start")
folder_path = os.path.dirname(os.path.realpath(__file__))
# current folder path

og_folder_path = os.path.join(folder_path, 'PolarBearVidID_og')
flip_folder_path = os.path.join(folder_path, 'PolarBearVidID_flip')

if not os.path.exists(flip_folder_path):
    os.makedirs(flip_folder_path)

file_list = get_all_jpgs_in_folder(os.path.join(folder_path, 'PolarBearVidID_og'))

for file in file_list:
    # read image and rotate it 90 degrees clockwise
    img = cv2.imread(file)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # save image
    new_file_path = file.replace('PolarBearVidID_og', 'PolarBearVidID_flip')
    if not os.path.exists(os.path.dirname(new_file_path)):
        os.makedirs(os.path.dirname(new_file_path))
    cv2.imwrite(os.path.join(folder_path, new_file_path), img)
    print("Image saved: " + new_file_path)    

print("Programm end")