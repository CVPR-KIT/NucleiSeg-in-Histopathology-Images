import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from auxilary.utils import *
import shutil
from tqdm import tqdm


path = "Dataset/MonuSegData/split2/train/"

i = 0
for imgName in tqdm(os.listdir(path)):
    if imgName.endswith("_mask.png"):
        continue
    #rename files
    os.rename(path+imgName, path+str(i)+".png")
    os.rename(path+imgName.split(".")[0]+"_bin_mask.png", path+str(i)+"_label.png")
    i+=1
