from torchvision import transforms
import torchstain
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import os
import sys
import logging
import argparse

from utils import createDir, readConfig


def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='none', help='Path to the config file.')
    return parser.parse_args()


def machenkoNormal(img, target):
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    t_to_transform = T(img)
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(T(target))
    norm, H, E = normalizer.normalize(I=t_to_transform, stains=True)
    return norm.numpy().astype(np.uint8)


def reinhardNormal(img, target):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    t_to_transform = T(img)
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
    normalizer.fit(T(target))
    norm = normalizer.normalize(I=t_to_transform)
    return norm.numpy().astype(np.uint8)

if __name__=='__main__':

    '''
    To Normalize the images in the dataset. The images in MoNuSeg dataset are HE stained images. 
    It is recommended to normalize the images before training.
    for more information - https://www.leicabiosystems.com/en-kr/knowledge-pathway/he-staining-overview-a-guide-to-best-practices/

    '''

    # read config
    args = arg_init()
    config = args.config

    if config == 'none':
        print("Please provide the path to the config file")
        exit()

    config = readConfig(config)
    
    pathDataset = config["to_be_aug"]
    targetPath = config["targetImagePath"]
    newPath = 'Dataset/MonuSegData/Test/TissueImagesNormalized/'
    normalMethod = config["normalization"]

    dirs = ["Training", "Test"]
    for dir_ in dirs:
        path = pathDataset + dir_ + "/TissueImages/"
        newPath = pathDataset + dir_ + "/TissueImagesNormalized/"
        createDir([newPath])

        print("Normalizing images in ", path)
        for imgPath in tqdm(natsorted(os.listdir(path))):
            # continue if image is not png or tif
            if not imgPath.endswith('.png') and not imgPath.endswith('.tif'):
                continue
            img = cv2.imread(path+imgPath)
            target = cv2.imread(targetPath)
            if normalMethod == "machenko":
                norm = machenkoNormal(img, target)
            elif normalMethod == "reinhard":
                norm = reinhardNormal(img, target)
            else:   
                print("No Normalization method specified. Exiting...")    
                sys.exit(0)        
            cv2.imwrite(newPath+imgPath, norm)
        
    
 