from skimage.util.shape import view_as_windows
from skimage import img_as_ubyte

import os
import sys
import glob
import scipy.io
import numpy as np
import cv2 
from natsort import natsorted
from tqdm import tqdm

from utils import createDir

from sklearn.model_selection import train_test_split

patch_width = 256
patch_height = 256

dataset_dir = "Dataset/Otherdata/"
img_dir = "cpm17/test/Images"
label_dir = "cpm17/test/Labels"

prepared_dataset_dir = "prepared_dataset_cpm17"
#print()
image_list = natsorted(glob.glob(os.path.join(dataset_dir,img_dir)+"/*.png"))
#print(image_list)
createDir([os.path.join(dataset_dir,prepared_dataset_dir), os.path.join(dataset_dir,prepared_dataset_dir,"TissueImages"),os.path.join(dataset_dir,prepared_dataset_dir,"GroundTruth")])

for idx, img_path in tqdm(enumerate(image_list)):
    #image = io.imread(img_path)
    image = cv2.imread(img_path)
    
    label_path = glob.glob(os.path.join(dataset_dir,label_dir,os.path.splitext(os.path.basename(img_path))[0])+"*")[0]
    #print(label_path)
    label = scipy.io.loadmat(label_path)
    label = label['inst_map']
    label = label.reshape(label.shape[0],label.shape[1],1)
    #print(image.shape,label.shape)
    for i in range(0,label.shape[0]):
        for j in range(0,label.shape[1]):
            # change all non black pixel to white
            if label[i][j][0] != 0:
                label[i][j][0] = 255


    cv2.imwrite(os.path.join(dataset_dir,prepared_dataset_dir,"TissueImages",str(idx)+".png"),image)
    cv2.imwrite(os.path.join(dataset_dir,prepared_dataset_dir,"GroundTruth",str(idx)+"_bin_mask.png"),label)
    