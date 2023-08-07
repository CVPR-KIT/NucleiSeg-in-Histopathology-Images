import os
import cv2
from tqdm import tqdm
import numpy as np
from natsort import natsorted
import sys

from auxilary.utils import *

'''
Run this file for Sliding image augmentation.
'''
# Preserve the directory name in the given order as it is associated with makeVal.bash

# Read Config
config = readConfig()
tile_width = config["tileHeight"]
tile_height = config["tileWidth"]
slidingTile = config["slidingSize"]
#print(config)


log_dir = config["log"]
createDir([log_dir])

f = open(log_dir +  "slidingAugLog.txt", "w")

if not os.path.exists(config["out_dir"]):
        os.mkdir(config["out_dir"])


image_dir = config["to_be_aug"]+'GroundTruth/'

# Read json file
metadata = readJson(config["to_be_aug"]+"metadata.json")

#print(metadata)
out_dir_img = os.path.join(config["out_dir"], "images/")
out_dir_label = os.path.join(config["out_dir"], "labels/")
createDir([out_dir_img, out_dir_label])

print("Augmenting images in ", config["out_dir"], " directory")
f.write("Augmenting images in "+ config["out_dir"]+ " directory\n")
cnt = 0

for imageidx, imagePath in tqdm(enumerate(metadata["images"])):
    f.write("imagePath: "+imagePath+"\n")
    image = cv2.imread(imagePath)
    # read binary image
    f.write("labelPath: "+metadata["labels"][imageidx]+ "\n")
    label = cv2.imread(metadata["labels"][imageidx], cv2.IMREAD_GRAYSCALE)
    if label is None:
        f.write("label is None\n")

    for x in range(0, image.shape[0]-tile_height, slidingTile):
        for y in range(0, image.shape[1]-tile_width,slidingTile):
            f.write("("+str(x)+","+str(y)+")\n")
            imageTile = image[x:x + tile_height, y:y + tile_width]
            labelTile = label[x:x + tile_height, y:y + tile_width]
            cv2.imwrite(os.path.join(out_dir_img,str(cnt)+".png"), imageTile)
            cv2.imwrite(os.path.join(out_dir_label,str(cnt)+".png"), labelTile)
            cnt += 1


print("Sliding Augmentation done")
f.write("Sliding Augmentation done\n")
f.close()