import os
from auxilary.utils import readConfig, createDir, readJson
import shutil
from tqdm import tqdm

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


def makeTest(config):
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
    config = readConfig()
    makeVal(config)
    #makeTest(config)