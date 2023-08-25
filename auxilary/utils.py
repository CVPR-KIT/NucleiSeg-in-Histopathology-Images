import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import os
import sys
from torchsummary import summary
import torch




# Create directory
def createDir(dirs):
    '''
    Create a directory if it does not exist
    dirs: a list of directories to create
    '''
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('Directory %s already exists' %dir)

# Recolor the result image
def result_recolor(gray_img):
    img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), np.uint8)
    config = readConfig()

    colormap = [config["class1"],config["class2"],config["class3"],config["class4"]] # Purple, Orange, Yellow, Black
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            #print(gray_img[i,j])
            img[i,j,:] = colormap[gray_img[i,j]]
    return img

# get an array of pixel values from string
def getPixel(pixelStr):
    pixelStr = pixelStr.split(",")
    pixel = []
    for i in pixelStr:
        pixel.append(int(i))
    return pixel

# Show image
def showImage(img, name = "image"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
# Generate Configuration File
def makeConfigJson(config, path = "config.json"):
    # convert dict config to json
    f = open(path, "w")
    f.write(json.dumps(config, indent=4))
    f.close()

# Read json file
def readJson(path = "config.json"):
    f = open(path, "r")
    config = json.load(f)
    f.close()
    return config

# Read Configuration File
def readConfig(configPath = "config.sys"):
    f = open(configPath, "r")
    exp = ["[", "#"]
    fileLines = f.readlines()
    config = {}
    for line in fileLines:
        if line[0] in exp:
            continue
        line = line.split("#")[0]
        #print(line)
        if len(line) < 2:
            continue
        # for string inputs
        if "\"" in line.split("=")[1]:
            config[line.split("=")[0].strip()] = line.split("=")[1].strip()[1:-1]
            if config[line.split("=")[0].strip()] == "True":
                config[line.split("=")[0].strip()] = True
            elif config[line.split("=")[0].strip()] == "False":
                config[line.split("=")[0].strip()] = False
            if config[line.split("=")[0].strip()] == "None":
                config[line.split("=")[0].strip()] = None
            if config[line.split("=")[0].strip()] == "none":
                config[line.split("=")[0].strip()] = None
            if line.split("=")[0].strip() in ["class1", "class2", "class3", "class4"]:
                config[line.split("=")[0].strip()] = getPixel(config[line.split("=")[0].strip()])
        # int or float inputs       
        else:
            if "." in line.split("=")[1]:
                config[line.split("=")[0].strip()] = float(line.split("=")[1].strip())
            else:
                config[line.split("=")[0].strip()] = int(line.split("=")[1].strip())
    f.close()
    # print config json for sanity check
    #makeConfigJson(config)
    return config

# Save Torch Summary to a file
def saveTorchSummary(model, input_size, path="modelSummary.txt"):
    sys.stdout = open(path, "w")
    #f.write(summary(model, input_size))
    summary(model, input_size, )
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    #f.close()

# Convert Gray image to fixed pixel gray value and removes interpolation
def toGray(grayscale, threshold=(100, 170), value=(0, 128, 255)):
    """
    Input:
        - grayscale : np.ndarray (h, w)
        - threshold : tuple[int, int] : lower & upper threshold
        - value     : tuple[int, int, int] : lower, middle & upper value
    """
    board = np.zeros_like(grayscale, dtype=np.uint8)
    mask1 = grayscale < threshold[0]
    mask2 = grayscale < threshold[1]
    board[mask1] = value[0]
    board[~mask1 & mask2] = value[1]
    board[~mask2] = value[2]
    return board

# to Gray function but for 4 channels
def toGray4C(grayscale, threshold=(30, 60, 180), value=(0, 52, 154, 255)):
    """
    Input:
        - grayscale : np.ndarray (h, w)
        - threshold : tuple[int, int] : lower & upper threshold
        - value     : tuple[int, int, int] : lower, middle & upper value
    """
    board = np.zeros_like(grayscale, dtype=np.uint8)
    mask1 = grayscale < threshold[0]
    mask2 = grayscale < threshold[1]
    mask3 = grayscale < threshold[2]
    board[mask1] = value[0]
    board[~mask1 & mask2] = value[1]
    board[~mask2 & mask3] = value[2]
    board[~mask3] = value[3]
    return board

# Return magnified image
def zoom_center(img, zoom_factor=1.5):

    y_size = img.shape[0]
    x_size = img.shape[1]
    
    # define new boundaries
    x1 = int(0.5*x_size*(1-1/zoom_factor))
    x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
    y1 = int(0.5*y_size*(1-1/zoom_factor))
    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))

    # first crop image then scale
    img_cropped = img[y1:y2,x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)

# Calculate confusion matrix
def calc_confusion_matrix(label, pred,num_classes):
    label = label.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    category = num_classes*label + pred
    confusion_matrix = np.zeros((num_classes,num_classes))
    for i in range(category.shape[0]):
        oneD_array = np.reshape(category[i],(-1))
        bincount = np.bincount(oneD_array,minlength=num_classes**2)
        confusion_matrix += np.reshape(bincount,(num_classes,num_classes))
   
    return confusion_matrix

# Calculate confusion matrix 2
def calc_confusion_matrix2(label, pred, num_classes):
    label = label.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    # Reshape label and pred tensors to match
    #print shapes of label and pred
    #print(label.shape)
    #print(pred.shape)
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    #print(label.shape)
    #print(pred.shape)

    # Calculate the confusion matrix
    cm = confusion_matrix(label, pred, labels=range(num_classes))

    return cm

# Calculate accuracy
def calc_accuracy(confusion_matrix):
    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    return accuracy

# Calculate mIoU

def calc_mIoU(confusion_matrix):
    mIoU = 0
    for row in range(len(confusion_matrix)):
        if (np.sum(confusion_matrix[row,:])+np.sum(confusion_matrix[:,row])) == 0:
            continue
        mIoU += confusion_matrix[row,row]/(np.sum(confusion_matrix[row,:])+np.sum(confusion_matrix[:,row]))
    return mIoU / len(confusion_matrix)

def calc_mIoU(confusion_matrix):
    mIoU = 0
    for row in range(len(confusion_matrix)):
        intersect = confusion_matrix[row,row]
        union = np.sum(confusion_matrix[row,:]) + np.sum(confusion_matrix[:,row]) - intersect
        if union == 0:
            continue
        mIoU += intersect / union
    return mIoU / len(confusion_matrix)


# Normalize image
def normalize_image(image):
    # Assuming the image shape is (height, width, channels)
    # Calculate the mean and standard deviation for each channel
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))

    # Normalize each channel separately
    normalized_image = (image - mean) / (std + 1e-7) # Adding a small constant to avoid division by zero
    
    return normalized_image

# Function to unnormalize an image based on its mean and std
def unnormalize_image(image):
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    unnormalized_image = (image * std) + mean
    return unnormalized_image


if __name__ == "__main__":
    print("Contains functions used in the project - utils.py")
    #print(readConfig())