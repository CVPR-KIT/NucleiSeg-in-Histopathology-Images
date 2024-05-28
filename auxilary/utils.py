import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import os
import sys
from torchsummary import summary
from tqdm import tqdm
from natsort import natsorted
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

    colormap = [config["class1"],config["class2"]] # black, white
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

# convert string array into float array
def getArray(arrayStr):
    arrayStr = arrayStr.split(",")
    array = []
    for i in arrayStr:
        array.append(float(i))
    return array

# Read Metrics File
def readMetrics(path = None):
    if path == None:
        assert False, "Experiment Path not provided"

    f = open(f"{path}metrics.txt", "r")
    fReader = f.read()
    f.close()   
    fReader.split("\n")
    fReader = fReader.split("\n")
    

    # finding confusion matrix, best mIoU and best val accuracy
    confusionMatrix = []
    bestmIoU = None
    bestValAccuracy = None
    for i, line in enumerate(fReader):
        # if line contains Training Losses
        if "Training Losses" in line:
            trainingLoss = fReader[i+1].strip()
            trainingLoss = getArray(trainingLoss[0:-2])
        # if line contains Training Accuraries
        if "Training Accuracies" in line:
            trainingAccuracy = fReader[i+1].strip()
            trainingAccuracy = getArray(trainingAccuracy[0:-2])
        # if line contains Validation Losses
        if "Validation Losses" in line:
            validationLoss = fReader[i+1].strip()
            validationLoss = getArray(validationLoss[0:-2])
        # if line contains Validation Accuracies
        if "Validation Accuracies" in line:
            validationAccuracy = fReader[i+1].strip()
            validationAccuracy = getArray(validationAccuracy[0:-2])

        # if line contains 'confusion matrix'
        if "confusion matrix" in line:
            #print("confusion matrix found")
            l1 = fReader[i+2].strip()
            l2 = fReader[i+3].strip()


            one_1 = l1.split(" ")[0][2:]
            one_2 = l1.split(" ")[-1][:-1]
            two_1 = l2.split(" ")[0][2:]
            if not len(two_1):
                two_1 = l2.split(" ")[1]
                if not len(two_1):
                    two_1 = l2.split(" ")[2]
            two_2 = l2.split(" ")[-1][:-2]

            #print(one_1, one_2, two_1, two_2)

            confusionMatrix.append([float(one_1), float(one_2)])
            confusionMatrix.append([float(two_1), float(two_2)])
            confusionMatrix = np.array(confusionMatrix)

        if "val mIoUs" in line:
            bestmIoU = float(fReader[i+2].strip())
            #print(bestmIoU)
        if "val accuracy" in line:
            bestValAccuracy = float(fReader[i+2].strip())

    #return trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, confusionMatrix, bestmIoU, bestValAccuracy
    return {
        'trainingLoss': trainingLoss,
        'trainingAccuracy': trainingAccuracy,
        'validationLoss': validationLoss,
        'validationAccuracy': validationAccuracy,
        'confusionMatrix': confusionMatrix,
        'bestmIoU': bestmIoU,
        'bestValAccuracy': bestValAccuracy
    }

# Save Torch Summary to a file
def saveTorchSummary(model, input_size, path="modelSummary.txt"):
    sys.stdout = open(path, "w")
    #f.write(summary(model, input_size))
    summary(model, input_size)
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


def calculate_iou(pred, target):
    intersection = torch.logical_and(target, pred)
    union = torch.logical_or(target, pred)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou

def classify_segments(preds, targets, iou_threshold=0.5):
    tp, fp, fn = 0, 0, len(targets)
    for pred in preds:
        matched = False
        for target in targets:
            iou = calculate_iou(pred, target)
            if iou > iou_threshold:
                tp += 1
                matched = True
                break
        if matched:
            fn -= 1
        else:
            fp += 1
    return tp, fp, fn


def calculate_pq(preds, targets, iou_threshold=0.5):
    tp, fp, fn = classify_segments(preds, targets, iou_threshold)
    total_iou = sum(calculate_iou(pred, target) for pred, target in zip(preds, targets) if calculate_iou(pred, target) > iou_threshold)

    sq = total_iou / tp if tp > 0 else 0
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    pq = sq * dq

    return pq



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


def calculate_mAP(confusion_matrix):
    """
    Calculate Mean Average Precision (mAP) from a confusion matrix.
    
    Parameters:
    - confusion_matrix (numpy array): A square matrix where each row and column corresponds to a class.
                                      The value at (i, j) is the number of instances of class i predicted as class j.
                                      
    Returns:
    - mAP (float): Mean Average Precision.
    """
    num_classes = len(confusion_matrix)
    APs = []
    
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = np.sum(confusion_matrix) - TP - FP - FN
        
        # Calculate Precision and Recall
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        
        # Calculate Average Precision (AP) using trapezoidal approximation
        AP = (precision + recall) / 2 if (precision + recall) != 0 else 0
        
        APs.append(AP)
        
    # Calculate Mean Average Precision (mAP)
    mAP = np.mean(APs)
    
    return mAP

# Calculate accuracy
def calc_accuracy(confusion_matrix):
    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    return accuracy

# Calculate mIoU
'''
def calc_mIoU(confusion_matrix):
    mIoU = 0
    for row in range(len(confusion_matrix)):
        if (np.sum(confusion_matrix[row,:])+np.sum(confusion_matrix[:,row])) == 0:
            continue
        mIoU += confusion_matrix[row,row]/(np.sum(confusion_matrix[row,:])+np.sum(confusion_matrix[:,row]))
    return mIoU / len(confusion_matrix)'''
def calc_mIoU(confusion_matrix):
    mIoU = 0
    for row in range(len(confusion_matrix)):
        intersect = confusion_matrix[row,row]
        union = np.sum(confusion_matrix[row,:]) + np.sum(confusion_matrix[:,row]) - intersect
        if union == 0:
            continue
        mIoU += intersect / union
    return mIoU / len(confusion_matrix)



def calc_mIoU2(pred, label, num_classes):
    # Initialize variables
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    
    # Loop through each class
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (label == cls)
        
        # Calculate intersection and union for each class
        intersection[cls] = (pred_inds[target_inds]).sum()
        union[cls] = pred_inds.sum() + target_inds.sum() - intersection[cls]
    
    # Ignore zero union values (to avoid division by zero)
    iou = intersection / (union + 1e-10)
    
    # Calculate mean IoU
    mIoU = iou.mean().item()
    
    return mIoU

#calculate Dice Score
def calc_dice_score(confusion_matrix):
    dice_score = 0
    for row in range(len(confusion_matrix)):
        intersect = confusion_matrix[row,row]
        union = np.sum(confusion_matrix[row,:]) + np.sum(confusion_matrix[:,row])
        if union == 0:
            continue
        dice_score += 2*intersect / union
    return dice_score / len(confusion_matrix)

def calc_dice_score2(confusion_matrix):
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    # Calculating the Dice Coefficient
    dice_coefficient = (2 * TP) / (2 * TP + FP + FN)
    return dice_coefficient

def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_torch(y_true, y_pred):
    smooth = 1e-5
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


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

# Load images from path
def load_images(image_paths):
    images = []
    print(f"loading Images from path: {image_paths}")
    for filename in tqdm(natsorted(os.listdir(image_paths))):
        if filename.endswith("_label.png"):
            continue
        img = cv2.imread(os.path.join(image_paths,filename))
        if img is not None:
            images.append(img)
    return images

# Load sampling Model
def load_sampling_model(modelType):
    if modelType == "small":
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif modelType == "base":
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    elif modelType == "large":
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    elif modelType == "giga":
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    else:
        print("Invalid Sampler model type")
        return None
    dino_model = dino_model.cuda()
    return dino_model

# Lookahead Class
from torch.optim import Optimizer

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        self.state = {}
        
        # Initialize slow weights with fast weights
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state.setdefault(p, {})
                param_state['slow_buffer'] = torch.zeros_like(p.data)
                param_state['slow_buffer'].copy_(p.data)
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter % self.k == 0:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['slow_buffer'].add_((p.data - param_state['slow_buffer']) * self.alpha)
                    p.data.copy_(param_state['slow_buffer'])

        return loss



    def zero_grad(self):
        self.optimizer.zero_grad()

if __name__ == "__main__":
    print("Contains functions used in the project - utils.py")
    #print(readConfig())