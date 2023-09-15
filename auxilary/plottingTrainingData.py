import numpy as np
import matplotlib.pyplot as plt
from utils import readMetrics, calc_dice_score

def printImage(data, head, outPath):

    print(f"[INFO]: Training {head} Plot Generation")
    N = len(data[0] )
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), data[0], label="train_"+head)
    #plt.plot(np.arange(0, N), train_acc, label="train_acc")
    plt.plot(np.arange(0, N), data[1], label="val_"+head)
    #plt.plot(np.arange(0, N), val_acc, label="val_acc")
    title = f"Training {head} on MoNuSeg Dataset - ShortUnet3+"
    #plt.ylim(0.8, 1)
    if head == "Accuracy":
        plt.ylim(top=1.0)
    if head == "Loss":
        plt.ylim(bottom=0.0)
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel(head)
    plt.legend(loc="lower right")
    plt.savefig(f"{outPath}plot_{head}.png")

def unfinished(path, outPath):
    f = open(path, "r")
    fReader = f.read()
    f.close()
    fReader = fReader.split("\n")

    train_loss = []
    train_mIoU = []
    train_acc = []
    val_loss = []
    val_mIoU = []
    val_acc = []

    for line in fReader:
        if line.startswith("train_loss"):
            train_loss.append(float(line.split(":")[1].strip()))
        elif line.startswith("train_mIoU"):
            train_mIoU.append(float(line.split(":")[1].strip()))
        elif line.startswith("train_acc"):
            train_acc.append(float(line.split(":")[1].strip()))
        elif line.startswith("val_loss"):
            val_loss.append(float(line.split(":")[1].strip()))
        elif line.startswith("val_mIoU"):
            val_mIoU.append(float(line.split(":")[1].strip()))
        elif line.startswith("val_acc"):
            val_acc.append(float(line.split(":")[1].strip()))

    # plot graph for training loss and accuracy and validation loss and accuracy
    #train_loss = [0.25006714254182866, 0.1928768632588563, 0.17703613093672643, 0.1655030488867551, 0.15564322062435998]
    #train_acc = [0.8445546920496652, 0.8713883586559589, 0.8805587089244588, 0.8877894058684218, 0.8939133025803372]
    #val_loss = [0.21477208098691422, 0.19198805339295755, 0.17996057449379052, 0.17317383136813488, 0.16608148720429]
    #val_acc = [0.8558756606747405, 0.8638045314185145, 0.8620655721285527, 0.869460581127642, 0.8688149532485089]
    printImage([train_loss, val_loss], "Loss", outPath)
    printImage([train_acc, val_acc], "Accuracy", outPath)
    printImage([train_mIoU, val_mIoU], "mIoU", outPath)

def finished(path, outPath):
    
    results = readMetrics(path) 

    trainingLoss = results["trainingLoss"]
    trainingAccuracy = results["trainingAccuracy"]
    validationLoss = results["validationLoss"]
    validationAccuracy = results["validationAccuracy"]

    #print('trainingLoss', trainingLoss)
    #print('trainingAccuracy', trainingAccuracy)
    #print('validationLoss', validationLoss)
    #print('validationAccuracy', validationAccuracy)


    cm = results["confusionMatrix"]
    dice = calc_dice_score(cm)
    print(f"Dice Score: {dice}")

    print(f"Best Validation Accuracy: {results['bestValAccuracy']}")

    # plot graph for training loss and accuracy and validation loss and accuracy
    printImage([trainingLoss, validationLoss], "Loss", outPath)
    printImage([trainingAccuracy, validationAccuracy], "Accuracy", outPath)


if __name__ == "__main__":
    #unfinished("log/log-09-08.txt", "Outputs/")
    finalPath = "Outputs/experiment_09-14_21.58.03/"
    finished(finalPath, finalPath)
    