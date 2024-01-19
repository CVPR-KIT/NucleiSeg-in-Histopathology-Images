import os
import cv2
from tqdm import tqdm
import numpy as np
from natsort import natsorted
import sys
import argparse

from auxilary.utils import *

'''
Run this file for Sliding image augmentation.
'''
# Preserve the directory name in the given order as it is associated with makeVal.bash


def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='none', help='Path to the config file.')
    return parser.parse_args()

# sliding image augmentation
def slidingAugment(indir, outdir, tile_width, tile_height, slidingTile):
    cnt = 0
    for i in tqdm(range(1, len(os.listdir(indir))//2)):
        image = cv2.imread(os.path.join(indir, str(i)+".png"))
        # read binary image
        label = cv2.imread(os.path.join(indir, str(i)+"_label.png"), cv2.IMREAD_GRAYSCALE)
        if label is None:
            print("label is None")

        for x in range(0, image.shape[0]-tile_height, slidingTile):
            for y in range(0, image.shape[1]-tile_width,slidingTile):
                imageTile = image[x:x + tile_height, y:y + tile_width]
                labelTile = label[x:x + tile_height, y:y + tile_width]
                cv2.imwrite(os.path.join(outdir,str(cnt)+".png"), imageTile)
                cv2.imwrite(os.path.join(outdir,str(cnt)+"_label.png"), labelTile)
                cnt += 1


if __name__ == '__main__':

    args = arg_init()
    config = args.config

    if config == 'none':
        print("Please provide the path to the config file")
        exit()

    config = readConfig(config)
    tile_width = config["tileHeight"]
    tile_height = config["tileWidth"]
    slidingTile = config["slidingSize"]


    log_dir = config["log"]
    createDir([log_dir])

    f = open(log_dir +  "logs-pre-training.txt", "a")
    f.write("Sliding Augmentation\n")

    if not os.path.exists(config["out_dir"]):   
        os.mkdir(config["out_dir"])

    createDir([config["out_dir"]+"training/", config["out_dir"]+"validation/"])

    image_dir = config["split_dir"]

    print("Sliding Augmentation of Training Images")
    f.write("Sliding Augmentation of Training Images\n")
    out_dir_img = config["out_dir"]+"training/"    
    slidingAugment(image_dir+"train/", out_dir_img, tile_width, tile_height, slidingTile)
    print("Sliding Augmentation of Val Labels")
    f.write("Sliding Augmentation of Val Labels\n")
    out_dir_label = config["out_dir"]+"validation/"    
    slidingAugment(image_dir+"val/", out_dir_label, tile_width, tile_height, slidingTile)

    print("Sliding Augmentation done")
    f.write("Sliding Augmentation done\n")
    f.close()