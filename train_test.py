from dataset import MonuSegOnlyTestDataSet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import os
import cv2
from auxilary.utils import *
from networkModules.model import UNet
from networkModules.modelUnet3p import UNet_3Plus
from networkModules.modelElunet import ELUnet
from datetime import datetime
import json

import logging

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='none', help='Path to the image directory. use "all" for all images')
    parser.add_argument('--img_type', type=str, default='other', help='Image Type: other / x3000 / x5000')
    parser.add_argument('--model_type', type=str, default='best', help='Model Type: best / model number')
    parser.add_argument('--expt_dir', type=str, default='none', help='Path to the experiment directory.')
    return parser.parse_args()

def make_preRunNecessities(expt_dir):
    # Read config.json file
    print("PreRun: Reading config file")
    
    # read json file
    config = None
    with open(expt_dir + "config.json") as f:
        config = json.load(f)   
    
    # Create the required directories
    print("PreRun: Creating required directories")
    createDir([config['log'], config['expt_dir']+'inference/', config['expt_dir']+'inference/testData/'])

    return config if config is not None else print("PreRun: Error reading config file")
    
def runInference(data, model, device, config, img_type):
    accList = []
    count= 0

    for i,(images,y) in enumerate(tqdm(data)):
        pred = model(images.to(device))

        #print(pred.shape)

        if not count:
            #torch.onnx.export(model, images.to(device), 'SS_MODEL.onnx', input_names=["Input Image"], output_names=["Predected Labels"])
            count+=1
        #print(int(pred.shape[2]))
        (wid, hit) = (int(pred.shape[2]), int(pred.shape[3]))
            
        #y = y.reshape((1,wid,hit))

        _, rslt = torch.max(pred,1)

        rslt = rslt.squeeze().type(torch.uint8)
        y = y.reshape((wid,hit))
        test_acc = torch.sum(rslt.cpu() == y)

        accList.append(test_acc.item() / (wid*hit))

        if config["input_img_type"] == "rgb":
            images = torch.reshape(images,(wid,hit, 3))
        else:
            images = torch.reshape(images,(wid,hit,1))
            
        images = images.cpu().detach().numpy()
        cv2.imwrite(config['expt_dir']+'inference/'+img_type+'/'+str(i)+'_'+'img.png',images*255)

            
        y = y.squeeze()
        label_color = result_recolor(y)
        cv2.imwrite(config['expt_dir']+'inference/'+img_type+'/'+str(i)+'_'+'label.png',label_color)

        rslt = rslt.squeeze()
        rslt_color = result_recolor(rslt.cpu().detach().numpy())
        cv2.imwrite(config['expt_dir']+'inference/'+img_type+'/'+str(i)+'_'+str(test_acc.item()/(wid*hit))[:5]+'_'+'predict.png',rslt_color)
    return np.average(accList)


'''
    1. load model
    2. load dataset
    3. inference
    4. save result
'''

def main():
    # Load Config
    args = arg_init()

    if args.expt_dir == 'none':
        print("Please specify experiment directory")
        sys.exit(1)
    if args.img_dir == 'none':
        print("Please specify test image directory")
        sys.exit(1)

    
    # run preRun
    config = make_preRunNecessities(args.expt_dir)

    # set logging
    logging.basicConfig(filename=config["log"] + "Test.log", filemode='a', 
                        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Testing Initiated")
    logging.info("PreRun: Creating required directories")


    # Set Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.debug(f"Using {device} device")

    # set weight path
    weight_path = args.expt_dir + "model/best_model.pth" if args.model_type == 'best' else args.expt_dir + "model/best_model_" + args.model_type + ".pth"
    # check if weight path exists
    if not os.path.exists(weight_path):
        print("Please specify valid model type")
        sys.exit(1)

    # log weight path
    logging.info("Weight Path: " + weight_path)
    
    # set model
    if config["model_type"] == "UNet":
        model = UNet(config)
    elif config["model_type"] == "UNet_3Plus":
        model = UNet_3Plus(config)
    elif config["model_type"] == "EluNet":
        model = ELUnet(config)
    else:
        logging.info("Please specify valid model.")
        print("Please specify valid model. Provided Model - ")
        print(config["model_type"])
        sys.exit(1)

    # Start inference
    logging.info("Starting Inference")

    logging.info(f"Loading Model at {weight_path}")
    checkpoint = torch.load(weight_path)

    logging.info("Loading checkpoints")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    logging.info("Starting evaluations")
    model.eval()

    # Write to File for future reference
    f = open(config["log"] + "inferences.csv", "a")

    if args.img_dir == 'all':
        logging.info("Testing on all images")
        paths = [('Dataset/test/', 'testData')]
        for path, img_type in paths:
            # Load Dataset
            logging.info("Loading dataset")
            dataset = MonuSegOnlyTestDataSet(path)
            data = DataLoader(dataset,batch_size=1)
            acc = runInference(data, model, device, config, img_type)
            f.write(f"{args.expt_dir},{img_type},{np.average(acc)} \n")
            print(f"Testing Accuracy -{args.expt_dir}-{img_type}- {np.average(acc)} \n")
    else:
        # Load Dataset
        logging.info("Loading dataset")
        dataset = MonuSegOnlyTestDataSet(args.img_dir)
        data = DataLoader(dataset,batch_size=1)
        acc = runInference(data, model, device, config, args)
        f.write(f"{args.expt_dir},{args.img_type},{np.average(acc)} \n")
        print(f"Testing Accuracy -{args.expt_dir}-{args.img_type}- {np.average(acc)} \n")

    
    f.close()

    
if __name__ == '__main__':
    '''
    run command: python train_test.py --expt_dir <path to experiment directory> --img_dir all
    '''
    main()
