from torchvision import transforms
import torchstain
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import os

from auxilary.utils import createDir


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
    path = 'Dataset/MonuSegData/Test/TissueImages/'
    targetPath = 'Dataset/MonuSegData/Training/TissueImages/TCGA-A7-A13F-01Z-00-DX1.png'
    newPath = 'Dataset/MonuSegData/Test/TissueImagesNormalized/'

    createDir([newPath])

    for imgPath in tqdm(natsorted(os.listdir(path))):
        # continue if image is not png or tif
        
        if imgPath.endswith('.png') or imgPath.endswith('.tif'):
            img = cv2.imread(path+imgPath)
            target = cv2.imread(targetPath)
            norm = reinhardNormal(img, target)
            cv2.imwrite(newPath+imgPath, norm)
        
    
 