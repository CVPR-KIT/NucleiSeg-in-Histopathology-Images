import os
import cv2
from tqdm import tqdm
from natsort import natsorted
import argparse
import sys
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auxilary.utils import *

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='none', help='Path to the config file.')
    return parser.parse_args()

def organiseImages(metadata, trainsplitLen, valsplitLen, f):
    
        images = metadata["images"]
        labels = metadata["labels"]

        f.write("Copying Train images\n")
    
        for i in tqdm(range(trainsplitLen)):
            shutil.copy(images[i], config["split_dir"]+"train/"+str(i)+".png")
            shutil.copy(labels[i], config["split_dir"]+"train/"+str(i)+"_label.png")
            f.write("train/"+str(i)+".png\n")

        f.write("Copying Val images\n")
    
        for i in tqdm(range(trainsplitLen, trainsplitLen+valsplitLen)):
            shutil.copy(images[i], config["split_dir"]+"val/"+str(i-trainsplitLen)+".png")
            shutil.copy(labels[i], config["split_dir"]+"val/"+str(i-trainsplitLen)+"_label.png")
            f.write("val/"+str(i-trainsplitLen)+".png\n")
    
        #print("Images moved")


if __name__ == '__main__':

    args = arg_init()
    config = args.config

    if config == 'none':
        print("Please provide the path to the config file")
        exit()

    config = readConfig(config)

    log_dir = config["log"]
    createDir([log_dir])

    f = open(log_dir +  "logs-pre-training.txt", "w")

    if not os.path.exists(config["split_dir"]):
        os.mkdir(config["split_dir"])

    createDir([config["split_dir"]+"train/", config["split_dir"]+"val/"])

    metadata = readJson(config["to_be_aug"]+"metadata.json")
    assert metadata["numImages"] == metadata["numLabels"], "Number of images and labels are not equal"

    numImages = metadata["numImages"]
    splitRatio = config["splitRatio"]
    f.write("Spliting images into train and val\n")
    f.write("Split Ratio: "+str(splitRatio)+"\n")
    trainsplitLen = round(numImages * (splitRatio))
    valsplitLen = numImages - trainsplitLen
    f.write("Train, Val Length: "+str(trainsplitLen)+", "+str(valsplitLen)+"\n")
    organiseImages(metadata, trainsplitLen, valsplitLen, f)

    print("Training Validation Dataset Creation Completed")
    f.write("Training Validation Dataset Creation Completed\n")