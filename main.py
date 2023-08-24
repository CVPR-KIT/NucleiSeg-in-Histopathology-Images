from tqdm import tqdm
import cv2
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from networkModules.modelUnet3p import UNet_3Plus
from networkModules.model import UNet
from networkModules.modelElunet import ELUnet
import numpy as np
import torch.backends.cudnn as cudnn
import random
from dataset import MonuSegDataSet, MonuSegValDataSet, MonuSegTestDataSet

from torch_lr_finder import LRFinder
import argparse
import os
import time
from datetime import datetime
import logging


from auxilary.utils import *
from auxilary.lossFunctions import *


import wandb

'''
Command to run:
python main.py --config config.sys |& tee log/log-08-07.txt
'''

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
spl_losses = ['unet3+loss', 'improvedLoss']

def make_preRunNecessities(config):
    # Create the log directory
    logging.info("PreRun: Creating required directories")
    print("PreRun: Creating required directories")
    currTime = datetime.now().strftime("%m-%d_%H.%M.%S")
    config["expt_dir"] = f"Outputs/experiment_{currTime}/"
    createDir(['Outputs/', config["log"], config["expt_dir"], config["expt_dir"] + "model/", config["expt_dir"] + "step/"])

    # Create the config file
    logging.info("PreRun: Generating experiment config file")
    print("PreRun: generating experiment config file")
    makeConfigJson(config, config["expt_dir"] + "config.json")
    
def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='none', help='Path to the config file.')
    return parser.parse_args()

def calculate_class_weights(targets, num_classes):
        # Calculate class weights based on target labels
        class_counts = torch.bincount(targets.flatten(), minlength=num_classes)
        total_samples = targets.numel()
        class_weights = total_samples / (num_classes * class_counts.float())
        return class_weights

def getSuggestedLR(model, optimizer, criterion, train_data, config):
    # Finding best learning rate
    # Setting logging to critical to stop debug messages
    #logging.basicConfig(level=logging.CRITICAL)

    # Initialize the learning rate finder
    print("Finding the best learning Rate...")
    logging.info("Finding the best learning Rate...")

    #setting std out to file to store lr_finder output
    sys.stdout = open(config["expt_dir"]+"lr_finder_output.txt", "w")
        
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda", )
    # Run the learning rate finder
    lr_finder.range_test(train_data, end_lr=100, num_iter=100)
    # Plot the learning rate finder results
    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    fig.savefig(config["expt_dir"]+"lr_finder_plot.png")

    # Close the sys.stdout file and set it back to normal
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    # get suggested learning rate from lr_finder saved text file
    f = open(config["expt_dir"]+"lr_finder_output.txt", "r")
    fileReader = f.read()
    sLR = fileReader.split("Suggested LR:")[1]
    # strip and to lowercase
    sLR = sLR.strip().lower()
    print("Best LR found:", sLR)
    logging.info("Best LR found: "+sLR)

    # Setting logging back to info
    #logging.basicConfig(level=logging.INFO)
    return float(sLR)

def run_epoch(model, data_loader, criterion, optimizer, epoch,device, mode,config):
    pgbar = tqdm(data_loader)
    configEpochs = config["epochs"]
    pgbar.set_description(f"Epoch {epoch}/{configEpochs}")
    losses = 0

    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    confusion_matrix = np.zeros((config["num_classes"],config["num_classes"]))


    for idx,(images,label) in enumerate(pgbar):
        #print('images infos',images)
        #print('images shape:'+str(images.shape[2])+":"+str(images.shape[3]))
        #print(images.max(),images.min())
        pred = model(images.to(device))

        #####
        # boundary as a class or not
        ####
        gt = label.to(device)
        
        gt = gt.squeeze()
        if mode =='val' or mode == 'test':
            #print("images shape:", images.shape)
            if config["model_type"] == "UNet_3Plus" or config["model_type"] == "EluNet":
                gt = torch.reshape(gt,(1, config["num_classes"], images.shape[2],images.shape[3]))
            else:
                if config['loss'] in spl_losses:
                    gt = torch.reshape(gt,(1, config["num_classes"], images.shape[2],images.shape[3]))
                else:
                    gt = torch.reshape(gt,(1,images.shape[2],images.shape[3]))


        class_weights = calculate_class_weights(gt, config["num_classes"])
        
        #if loss is modJaccard, jaccard, pwxce, improvedLoss use weights
        weightable_losses = ['modJaccard', 'jaccard', 'pwcel', 'improvedLoss', 'ClassRatioLossPlus']
        if config["loss"] in weightable_losses:
            criterion.setWeights(class_weights.to(device))

        
        loss = criterion(pred,gt)

        #metric = MulticlassJaccardIndex(num_classes=3)
        #loss = metric(pred, gt)

        #print("loss on cuda:", loss.is_cuda)
        #print('loss infos :')
        #print(loss)
        #print(loss.max(),loss.min(),loss.mean())

        if mode == 'train':
            optimizer.zero_grad()
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()


            

        losses += loss.item()
        loss = loss.item()
        
        if config["model_type"] == "UNet_3Plus" or config["model_type"] == "EluNet":
            _, rslt = torch.max(pred,1)
            _, gt = torch.max(gt,1)
        else:
            _, rslt = torch.max(pred,1)
            if config["loss"] in spl_losses:
                _, gt = torch.max(gt,1)
            
        
        confusion_matrix += calc_confusion_matrix(gt.to(device), rslt, config["num_classes"])
        if mode == 'val' and idx == 0:
            label = torch.reshape(gt,(1,images.shape[2],images.shape[3]))
            label = label.permute(1,2,0)
            cv2.imwrite(config["expt_dir"] + "step/"+str(epoch)+'.png',label.cpu().detach().numpy()*(255/config["num_classes"]))
            rslt = rslt.permute(1,2,0)
            cv2.imwrite(config["expt_dir"] + "step/"+str(epoch)+'_pred'+'.png',rslt.cpu().detach().numpy()*(255/config["num_classes"]))

        
    #print('\n\n','================ confusion matrix ================','\n\n')
    #print(train_confusion_matrix)
     
    mIoU = calc_mIoU(confusion_matrix)
    accuracy = calc_accuracy(confusion_matrix)   
    loss = losses/(len(data_loader) if len(data_loader) != 0 else 1)
    
    return loss, confusion_matrix, mIoU, accuracy

def initWandb(config):
    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="monunet-segmenation",
    # track hyperparameters and run metadata
    config=config
)

def main():
    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.set_grad_enabled(True)
    start_time = time.time()

    sys.stdout = sys.__stdout__



    # Read configFile
    userConfig = arg_init().config
    if userConfig == 'none':
        print('please input config file')
        exit()
    config = readConfig(userConfig)

    logging.basicConfig(filename=config["log"] + "System.log", filemode='a', 
                        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("\n\n#################### New Run ####################")
    #logging.basicConfig(level=logging.CRITICAL)


    # check if wandblogging is enabled
    wandbFlag = config["wandb"]

    # Initialize wandb
    if wandbFlag:
        initWandb(config)

    learning_rate = config["learning_rate"]
    num_epochs = config["epochs"]
    
    # Make necessary directories
    make_preRunNecessities(config)

    # Make Model Directorylearning_ratemodel2
    if not os.path.exists("model"):
        os.mkdir("model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    logging.info(f"Using {device} device")

    
    # Configuring DataLoaders
    print('configuring data loaders')
    logging.info('configuring data loaders')
    train_dataset = MonuSegDataSet(config["trainDataset"])
    #train_dataset = SteelDataSet('data/data_4-10/train/')
    #train_dataset = SteelDataSet('data/data3/purpleData/purple_400_20/train/')
    train_data = DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True)
    
    val_dataset = MonuSegValDataSet(config["valDataset"])
    #val_dataset = SteelValDataSet('data/data3/purpleData/purple_400_20/val/')
    val_data = DataLoader(val_dataset,batch_size=1,num_workers=4)

    
    # Configuring Model
    if config["model_type"] == "UNet_3Plus":
        print(f'configuring model - UNET 3+')
        logging.info('configuring model - UNET 3+')
        model = UNet_3Plus(config)
    elif config["model_type"] == "EluNet":
        print(f'configuring model - EluNet')
        logging.info('configuring model - EluNet')
        model = ELUnet(config)
    else:
        print(f'configuring model - UNET')
        logging.info('configuring model - UNET')
        model = UNet(config)
    

    # Set model to Device
    model.to(device)
    
    
    # save model config
    print('saving model summary')
    logging.info(f'saving model summary at {config["expt_dir"]+"modelSummary.txt"}')
    if config["input_img_type"] == "rgb":
        saveTorchSummary(model, input_size=(3, 256, 256), path=config["expt_dir"]+"modelSummary.txt")
    else:
        saveTorchSummary(model, input_size=(1, 256, 256), path=config["expt_dir"]+"modelSummary.txt")
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    

    
    # If resume is true, load model and optimizer
    if config["resume"]:
        print(config["resume"])
        checkpoint = torch.load(config["resumeModel"])
        model.load_state_dict(checkpoint['model_state_dict'])
        print('model loaded from checkpoint')
        logging.info('model loaded from checkpoint')

    if config["weight_path"] is not None:
        model.load_state_dict(torch.load(config["weight_path"]))
        

    # Configuring Loss Function
    if config["loss"] == "jaccard":
        criterion = jaccLoss()
    elif config["loss"] == "pwcel":
        criterion = pwcel()
    elif config["loss"] == "dice":
        criterion = diceLoss()
    elif config["loss"] == "modJaccard":
        criterion = modJaccLoss()
    elif config["loss"] == "unet3+loss":
        criterion = unet_3Loss()
    elif config["loss"] == "improvedLoss":
        criterion = unet_3Loss()
    elif config["loss"] == "ClassRatioLoss":
        criterion = ClassRatioLoss()
    elif config["loss"] == "RBAF":
        criterion = RBAF()
    else:
        criterion = FocalLoss(0.25)

    # Configuring Optimizer
    optimizer = torch.optim.Adam(model.parameters() ,lr=1e-7, weight_decay=1e-2)
    if config["resume"]:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  
    # exit
    #sys.exit(1)

    # If learning rate is auto, find best learning rate
    if config["learning_rate"] == "auto":
        learning_rate = getSuggestedLR(model, optimizer, criterion, train_data, config)
        
    
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=1e-2)
          


    # Logging Parameters
    print('logging parameters')
    logging.info('logging parameters')
    logParam = {}
    
    logParam['model'] = config["model_type"]
    logParam["epochs"] = num_epochs
    logParam["learning_rate"] = learning_rate
    logParam["batch_size"] = config["batch_size"]
    logParam["optimizer"] = "Adam"
    logParam["loss"] = config["loss"]
    logParam["activation"] = config["activation"]
    logParam["Kernel_size"] = config["kernel_size"]
    logParam["num_classes"] = config["num_classes"]
    logParam['channels'] = config['channel']
    logParam["dropout"] = config["dropout"]
    logParam["dilation"] = config["dilation"] 
    logging.info(json.dumps(logParam, indent=4))

    
    train_losses = []
    train_accuracies = []
    val_losses = []
    best_val_accuracy = 0
    best_val_cm = None
    best_val_mIoU = None
    val_accuracies = []

    print('starting training')
    logging.info('starting training')
    
    for epoch in range(config["resume_epoch"],num_epochs):
        train_loss ,train_confusion_matrix, train_mIoU,train_accuracy =  run_epoch(model, train_data, criterion, optimizer, epoch, device, 'train',config)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        print('train_loss:',train_loss)
        #logging.info('train_loss:',train_loss)
        print('train_mIoU:',train_mIoU)
        #logging.info('train_mIoU:',train_mIoU)
        print('train_accuracy:',train_accuracy)
        #logging.info('train_accuracy:',train_accuracy)
        
        val_loss, val_confusion_matrix,val_mIoU,val_accuracy = run_epoch(model, val_data, criterion, optimizer, epoch, device, 'val',config)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print('val_loss:',val_loss)
        #logging.info('val_loss:',val_loss)
        print('val_mIoU:',val_mIoU)
        #logging.info('val_mIoU:',val_mIoU)
        print('val_accuracy:',val_accuracy)
        #logging.info('val_accuracy:',val_accuracy)
        
        if wandbFlag:
            wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "train_mIoU": train_mIoU, "val_loss": val_loss, "val_accuracy": val_accuracy, "val_mIoU": val_mIoU})
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_losses[-1]
            # saving model
            print("saving model")
            logging.info(f"saving model at {config['expt_dir']+'model/best_model_'+str(epoch)+'.pth'}")
            torch.save({'model_state_dict' : model.state_dict(),'optimizer_state_dict':optimizer.state_dict()},config["expt_dir"]+'model/best_model_'+str(epoch)+'.pth')
            torch.save({'model_state_dict' : model.state_dict(),'optimizer_state_dict':optimizer.state_dict()},config["expt_dir"]+'model/best_model.pth')
            best_val_cm = val_confusion_matrix
            best_val_mIoU = val_mIoU
    
    #best_path = './new_unet_down_up/best_model_185.pth'
    # Loading model for testing
    print("loading model for testing")
    logging.info(f"loading model for testing from {config['expt_dir']+'model/best_model.pth'}")

    if config["model_type"] == "UNet_3Plus":
        best_model = UNet_3Plus(config)
    elif config["model_type"] == "UNet":
        best_model = UNet(config)
    elif config["model_type"] == "EluNet":
        best_model = ELUnet(config)
    else:
        print(f"Given model type {config['model_type']} is not supported. Proceeding with UNet")
        logging.info(f"Given model type {config['model_type']} is not supported. Proceeding with UNet")
        best_model = UNet(config)

    checkpoint = torch.load(config["expt_dir"]+'model/best_model.pth')
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device)
    
    print("Testing...")
    logging.info("Testing...")
    test_dataset = MonuSegTestDataSet(config["testDataset"])
    test_data = DataLoader(test_dataset,batch_size=1,num_workers=1)
    # make testing directory
    createDir([config["expt_dir"]+"testResults/"])
    pgbar = enumerate(tqdm(test_data))
    #val_confusion_matrix = np.zeros((config.num_classes, config.num_classes))
    for batch_idx, (images,label) in pgbar:        
            pred = model(images.to(device))

            gt = label.to(device)

            
            _, rslt = torch.max(pred,1)

            rslt = rslt.squeeze().type(torch.uint8)
            #cm = calc_confusion_matrix(y, rslt, config)
            #val_confusion_matrix += cm

            # saving images
            logging.info("saving images")
            if config["input_img_type"] == "rgb":
                images = torch.reshape(images,(images.shape[2],images.shape[3],3))
            else:
                images = torch.reshape(images,(images.shape[2],images.shape[3],1))
            images = images.cpu().detach().numpy()
            cv2.imwrite(config["expt_dir"]+"testResults/"+str(batch_idx)+'_img'+'.png',images*255)
                           
            rslt_color = result_recolor(rslt.cpu().detach().numpy())
            cv2.imwrite(config["expt_dir"]+"testResults/"+str(batch_idx)+'_pred_color'+'.png',rslt_color)
    
    best_val_cm = val_confusion_matrix
    best_val_mIoU = calc_mIoU(best_val_cm)
    best_val_accuracy = calc_accuracy(best_val_cm)

    # Saving Training Stats
    print('Saving Training Stats')
    logging.info(f'Saving Training Stats at {config["expt_dir"]+"loss_log.txt"}')
    loss_log_file = open(config["expt_dir"]+'metrics.txt','w')
    loss_log_file.write("===============Training Losses ===========\n")
    for loss in train_losses:
        loss_log_file.write(str(loss)+', ')
    loss_log_file.write("\n\n===============Training Accuracies ===========\n")
    for acc in train_accuracies:
        loss_log_file.write(str(acc)+', ')
    loss_log_file.write('\n\n===============Validation Losses ===========\n')
    for loss in val_losses:
        loss_log_file.write(str(loss)+', ')
    loss_log_file.write('\n\n===============Validation Accuracies ===========\n')
    for acc in val_accuracies:
        loss_log_file.write(str(acc)+', ')
    loss_log_file.write('\n')
    
    

    print('\n\n','================ confusion matrix ================','\n\n')
    loss_log_file.write('\n\n================ confusion matrix ================\n\n')
    print(best_val_cm)
    loss_log_file.write(str(best_val_cm))
    
    print('\n\n','================     val mIoUs    ================','\n\n')
    loss_log_file.write('\n\n================     val mIoUs    ================\n\n')
    print(best_val_mIoU)
    loss_log_file.write(str(best_val_mIoU))

    print('\n\n','================    val accuracy   ================','\n\n')
    loss_log_file.write('\n\n================    val accuracy   ================\n\n')
    print(best_val_accuracy)
    loss_log_file.write(str(best_val_accuracy))
    
    print('\n\n','================       end        ================','\n\n')
    loss_log_file.write('\n\n================       end        ================\n\n')

    end_time = time.time()
    elapsed_time = round((end_time - start_time ) / 3600, 2)
    print('elapsed time : ',elapsed_time,' hours')
    loss_log_file.write('\n\n================       Time Taken        ================\n\n')
    loss_log_file.write(str(elapsed_time)+' hours')
    loss_log_file.close()

    # Logging run time
    print('Logging run time')
    logging.info(f'Experiment took {elapsed_time} hours')

    #generting loss and accuracy plot
    print('generting loss and accuracy plot')
    logging.info(f'generting loss and accuracy plot at {config["expt_dir"]+"loss_plot.png"}')
    N = config["epochs"]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), train_losses, label="train_loss")
    plt.plot(np.arange(0, N), val_losses, label="val_loss")
    plt.plot(np.arange(0, N), train_accuracies, label="train_acc")
    plt.plot(np.arange(0, N), val_accuracies, label="val_acc")
    title = "Training Loss and Accuracy on MoNuSeg Dataset - "+config["model_type"]
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config['expt_dir']+"plot.png")



    
    # Experiment End
    logging.info('Experiment End')
    
if __name__ == '__main__':
    main()