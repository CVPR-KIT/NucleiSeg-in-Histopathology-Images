import cv2
import os
import numpy
from utils import readConfig
from natsort import natsorted
import logging
import json


def getData(mode = "Training/"):
    config = readConfig()
    dataDir = config["to_be_aug"]
    logging.debug(f"Data Directory: {dataDir}")
    typeDir = os.path.join(dataDir, mode)
    labelDir = os.path.join(typeDir, "GroundTruth/")
    logging.debug(f"Label Directory: {labelDir}")
    imageDir = os.path.join(typeDir, "TissueImages/")
    logging.debug(f"Image Directory: {imageDir}")


    return (imageDir, labelDir, dataDir)



def makeMeta(mode = "Training/", metaName = "metadata.json"):
    imageDir, labelDir, dataDir = getData(mode)
    
    imagePaths = natsorted(os.listdir(imageDir))
    logging.debug(f"Len of imagePaths: {len(imagePaths)}")
    

    # make a dict for image information
    metadata = {}

    # run once
    runOnce = True 

    images = []
    labels = []    
    
    for imagePath in imagePaths:
        # skip if imagepath ends with .png or .tif
        if not imagePath.endswith(".png") and not imagePath.endswith(".tif"):
            continue

        
        labelPath = os.path.join(labelDir, imagePath.split(".")[0]+"_bin_mask.png")
        
        image = cv2.imread(os.path.join(imageDir, imagePath))
        label = cv2.imread(labelPath)

        if image.shape != label.shape:
            logging.debug(f"Image Path: {imagePath}")

        images.append(os.path.join(imageDir, imagePath))
        labels.append(labelPath)
         
        if runOnce:
            logging.debug(f"Image Shape: {image.shape}")
            logging.debug(f"Label Shape: {label.shape}")
            metadata["imageSize"] = image.shape
            metadata["labelSize"] = label.shape

            logging.debug(f"unique values in image: {numpy.unique(image)}")
            logging.debug(f"unique values in label: {numpy.unique(label)}")
            runOnce = False

    metadata["numImages"] = len(images)
    metadata["numLabels"] = len(labels)
    metadata["images"] = images
    metadata["labels"] = labels

    # save metadata
    f = open(dataDir+metaName, "w")
    f.write(json.dumps(metadata, indent=4))
    f.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    makeMeta("Training/", "metadata.json")
    makeMeta("Test/", "metadataTest.json")