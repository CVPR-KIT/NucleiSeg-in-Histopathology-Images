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
    #unfinished("log/log-09-01-c32-wD.txt", "Outputs/experiment_09-01_19.02.25_NoMBP/")
    finalPath = "Outputs/channelTests/experiment_09-07_17.00.22/"
    finished(finalPath, finalPath)
    