import os
from auxilary.utils import readConfig, createDir, readJson
import shutil
from tqdm import tqdm
import argparse
import random
from natsort import natsorted


def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='none', help='Path to the config file.')
    return parser.parse_args()

def makeVal(config):
    
    log = config["log"]
    splitRatio = config["splitRatio"]
    trainDir = config["trainDataset"]
    valDir = config["valDataset"]
    augDir = config["augmented_dir"]

    f = open(log+'augmentationLog.txt', 'r')
    fStr = f.read()
    f.close()

    fStr = fStr.strip().split("\n")
    numImages = len(fStr) // 2
    valLen = round(numImages * (1-splitRatio))

    """ print(fStr)
    print(numImages)
    print(valLen) """

    createDir([trainDir, valDir])

    print("Making Validation Files")
    counter = 1
    #cwd = os.getcwd()
    for i in range(numImages-valLen, numImages):
        shutil.move(f"{augDir}{i}.png", f"{valDir}{counter}.png")
        shutil.move(f"{augDir}{i}_label.png", f"{valDir}{counter}_label.png")
        #shutil.move(f"{augDir}{i}_label_b.png", f"{valDir}{counter}_label_b.png")
        counter += 1
        #subprocess.run(f"mv data/augmentated/{fStr[i]} data/val/{fStr[i]}")
    #subprocess.run(f"mv data/augmentated/ data/train/")
    os.rename(augDir, trainDir)
    print("Validation Dataset Creation Completed")

def mergeVal(config):
    trainDir = config["trainDataset"]
    valDir = config["valDataset"]
    count = len(os.listdir(trainDir))//2
    
    for i in tqdm(range(1, len(os.listdir(valDir))//2+1)):
        #print(f"{valDir}{i}.png", f"{trainDir}{count+i-1}.png")
        shutil.move(f"{valDir}{i}.png", f"{trainDir}{count+i-1}.png")
        shutil.move(f"{valDir}{i}_label.png", f"{trainDir}{count+i-1}_label.png")

def makeVal2(config):
    splitRatio = config["splitRatio"]
    trainDir = config["trainDataset"]
    valDir = config["valDataset"]

    numImages = len(os.listdir(trainDir))//2
    valLen = round(numImages * (1-splitRatio))
    print(f"valLen: {valLen}")

    # pick random images
    random.seed(42)
    valImages = random.sample(range(numImages), valLen)
    
    for i in tqdm(valImages):
        shutil.move(f"{trainDir}{i}.png", f"{valDir}{i}.png")
        shutil.move(f"{trainDir}{i}_label.png", f"{valDir}{i}_label.png")

def organiseDataset(config):
    trainDir = config["trainDataset"]
    valDir = config["valDataset"]

    trainImages = os.listdir(trainDir)
    valImages = os.listdir(valDir)

    i = 0
    for imagePath in tqdm(natsorted(trainImages)):
        #print(f"{trainDir}{imagePath}", f"{trainDir}{i}.png")
        #print(f"{trainDir}{imagePath.split('.')[0]}_label.png", f"{trainDir}{i}_label.png")
        if not imagePath.endswith("_label.png"):
            os.rename(f"{trainDir}{imagePath}", f"{trainDir}{i}.png")
            os.rename(f"{trainDir}{imagePath.split('.')[0]}_label.png", f"{trainDir}{i}_label.png")
            i+=1

    print(len(trainImages))

def makeTest3(config):
    testOutDir = config["testDataset"]

    createDir([testOutDir])
    metadataTest = readJson(config["to_be_aug"]+"metadataTest.json")

    for imageidx, imagePath in tqdm(enumerate(metadataTest["images"])):
        new_imagePath = os.path.join(testOutDir, str(imageidx)+".png")
        new_labelPath = os.path.join(testOutDir, str(imageidx)+"_label.png")
        shutil.copy(imagePath, new_imagePath)
        shutil.copy(metadataTest["labels"][imageidx], new_labelPath)    
        
    print("Test Dataset Creation Completed")


if __name__ == '__main__':
    args = arg_init()

    if args.config == "none":
        print("Please provide a config file")
        exit(0)
    config = readConfig(args.config)
    
    #makeVal(config)
    #makeTest(config)
    mergeVal(config)
    #makeVal2(config)
    #organiseDataset(config)