import cv2
from torch.utils.data import Dataset
import random
import numpy as np
import os
import torch
import sys
import matplotlib.pyplot as plt
import logging
from auxilary.utils import result_recolor, toGray, toGray4C, readConfig, normalize_image
class MonuSegDataSet(Dataset):
    def __init__(self, img_dir, config = None):
        self.img_dir = img_dir

        if config is None:
            self.config = readConfig()
        else:
            self.config = config

        self.definedSize = (self.config["finalTileHeight"], self.config["finalTileWidth"])
        self.wid = 512 # default value and is replaced later 
        self.hit = 512 # default value and is replaced later 
        logging.basicConfig(filename=self.config["log"] + "dataloader.log", filemode='w', 
                        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.loggerFlag = True

        self.spl_losses = ['unet3+loss', 'improvedLoss', 'ClassRatioLoss']
        self.spl_models = ['UNet_3Plus', 'EluNet', 'UNet_3PlusShort']
        self.debug = self.config["debug"]
        self.debugDilution = self.config["debugDilution"]
    
        return 
    
    def __len__(self):
        if self.debug:
            return len(os.listdir(self.img_dir))//self.debugDilution
        else:
            return len(os.listdir(self.img_dir))//2

    def __getitem__(self, index):
        try:
            if self.config["input_img_type"] == "rgb":
                image = cv2.imread(os.path.join(self.img_dir,str(index)+'.png'), cv2.IMREAD_COLOR)/255
                # Normalize image
                #image = normalize_image(image)
            else:
                image = cv2.imread(os.path.join(self.img_dir,str(index)+'.png'),cv2.IMREAD_GRAYSCALE)/255
        except TypeError:
            print(os.path.join(self.img_dir,str(index)+'.png'))

        # set the width and height of the image
        self.wid = image.shape[0]
        self.hit = image.shape[1]
        # skip if height and width are not equal to the defined size
        if self.wid != self.definedSize[1] or self.hit != self.definedSize[0]:
            if self.loggerFlag:
                logging.info("Image size is not {}x{}".format(self.definedSize[1], self.definedSize[0]))
                self.loggerFlag = False
            return self.__getitem__(random.randint(0,len(self)-1))
        
        label = cv2.imread(os.path.join(self.img_dir,str(index)+'_label'+'.png'),cv2.IMREAD_GRAYSCALE)
        label[label==255] = 1
        label[label==0] = 0
        
        

        if self.config["input_img_type"] == "rgb":
            #image = np.reshape(image,(3,self.wid,self.hit))
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image,(1,self.wid,self.hit))

        if self.config['model_type'] in self.spl_models:
            # Only for Spl. Models
            # Create one-hot encoded tensors for each class
            class_0 = np.where(label == 0, 1, 0)  # Channel for class 0
            class_1 = np.where(label == 1, 1, 0)  # Channel for class 1

            # Concatenate the class channels to create the output tensor
            label = np.stack([class_0, class_1], axis=0)
        else:
            if self.config['loss'] in self.spl_losses:
                # Create one-hot encoded tensors for each class
                class_0 = np.where(label == 0, 1, 0)  # Channel for class 0
                class_1 = np.where(label == 1, 1, 0)  # Channel for class 1

                # Concatenate the class channels to create the output tensor
                label = np.stack([class_0, class_1], axis=0)
            else:
                label = np.reshape(label,(1,self.wid,self.hit))

        return torch.Tensor(image),torch.LongTensor(label)

class MonuSegValDataSet(Dataset):
    def __init__(self, img_dir, config = None):
        self.img_dir = img_dir

        if config is None:
            self.config = readConfig()
        else:
            self.config = config

        self.definedSize = (self.config["finalTileHeight"], self.config["finalTileWidth"])
        self.wid = 512 # default value and is replaced later 
        self.hit = 512 # default value and is replaced later 
        logging.basicConfig(filename=self.config["log"] + "dataloader.log", filemode='w', 
                        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.spl_losses = ['unet3+loss', 'improvedLoss', 'ClassRatioLoss']
        self.spl_models = ['UNet_3Plus', 'EluNet', 'UNet_3PlusShort']
        self.loggerFlag = True
        self.debug = self.config["debug"]
        self.debugDilution = self.config["debugDilution"]
    
        return 
    
    def __len__(self):
        if self.debug:
            return len(os.listdir(self.img_dir))//self.debugDilution
        else:
            return len(os.listdir(self.img_dir))//2

    def __getitem__(self, index):
        try:
            if self.config["input_img_type"] == "rgb":
                image = cv2.imread(os.path.join(self.img_dir,str(index+1)+'.png'))/255
                # Normalize image
                #image = normalize_image(image)
            else:
                image = cv2.imread(os.path.join(self.img_dir,str(index+1)+'.png'),cv2.IMREAD_GRAYSCALE)/255
        except TypeError:
            print(os.path.join(self.img_dir,str(index+1)+'.png'))

        # set the width and height of the image
        self.wid = image.shape[0]
        self.hit = image.shape[1]
        # skip if height and width are not equal to the defined size
        if self.wid != self.definedSize[1] or self.hit != self.definedSize[0]:
            if self.loggerFlag:
                logging.info("Image size is not {}x{}".format(self.definedSize[1], self.definedSize[0]))
                self.loggerFlag = False
            return self.__getitem__(random.randint(0,len(self)-1))

        label = cv2.imread(os.path.join(self.img_dir,str(index+1)+'_label'+'.png'),cv2.IMREAD_GRAYSCALE)
        if label is None:
            print(os.path.join(self.img_dir,str(index+1)+'_label'+'.png'))

        label[label==255] = 1
        label[label==0] = 0
        
        
        if self.config["input_img_type"] == "rgb":
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image,(1,self.wid,self.hit))

        if self.config['model_type'] in self.spl_models:
            # Only for Spl. Models
            # Create one-hot encoded tensors for each class
            class_0 = np.where(label == 0, 1, 0)  # Channel for class 0
            class_1 = np.where(label == 1, 1, 0)  # Channel for class 1

            # Concatenate the class channels to create the output tensor
            label = np.stack([class_0, class_1], axis=0)
        else:
            if self.config['loss'] in self.spl_losses:
                # Create one-hot encoded tensors for each class
                class_0 = np.where(label == 0, 1, 0)  # Channel for class 0
                class_1 = np.where(label == 1, 1, 0)  # Channel for class 1

                # Concatenate the class channels to create the output tensor
                label = np.stack([class_0, class_1], axis=0)
            else:
                label = np.reshape(label,(1,self.wid,self.hit))

        return torch.Tensor(image),torch.LongTensor(label)


class MonuSegTestDataSet(Dataset):
    def __init__(self, img_dir, config = None):
        self.img_dir = img_dir

        if config is None:
            self.config = readConfig()
        else:
            self.config = config
            
        self.wid = 512 # default value and is replaced later 
        self.hit = 512 # default value and is replaced later 
        logging.basicConfig(filename=self.config["log"] + "dataloader.log", filemode='w', 
                        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.len = 800
        self.spl_losses = ['unet3+loss', 'improvedLoss', 'ClassRatioLoss']
        self.spl_models = ['UNet_3Plus', 'EluNet', 'UNet_3PlusShort']
        return 
    
    def __len__(self):
        return len(os.listdir(self.img_dir))//2

    def __getitem__(self, index):

        img_paths  = sorted(os.listdir(self.img_dir))
        if self.config["input_img_type"] == "rgb":
            image = cv2.imread(os.path.join(self.img_dir,str(index)+img_paths[0][-4:]))/255
            # Normalize image
            #image = normalize_image(image)
        else:
            image = cv2.imread(os.path.join(self.img_dir,str(index)+img_paths[0][-4:]),cv2.IMREAD_GRAYSCALE)/255
        


        self.wid = image.shape[0]
        self.hit = image.shape[1]

        self.wid = self.hit = self.len
        
        label = cv2.imread(os.path.join(self.img_dir,str(index)+'_label'+img_paths[1][-4:]),cv2.IMREAD_GRAYSCALE)
            
        plabel = np.copy(label)
        image=image[0:self.wid,0:self.hit]
        label=label[0:self.wid,0:self.hit]
        plabel = plabel[0:self.wid,0:self.hit]

        #printout
        logging.info("Outputting label")
        cv2.imwrite('out/input_color.png', plabel)
        cv2.imwrite('out/input_label_gray.png', label)
        
        #with np.printoptions(threshold=np.inf):
         #   print(label[label==225])

        label[label==255] = 1
        label[label==0] = 0
        
        
        if self.config["input_img_type"] == "rgb":
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image,(1,self.wid,self.hit))

        if self.config['model_type'] in self.spl_models:
            # Only for Spl. Models
            # Create one-hot encoded tensors for each class
            class_0 = np.where(label == 0, 1, 0)  # Channel for class 0
            class_1 = np.where(label == 1, 1, 0)  # Channel for class 1

            # Concatenate the class channels to create the output tensor
            label = np.stack([class_0, class_1], axis=0)
        else:
            if self.config['loss'] in self.spl_losses:
                # Create one-hot encoded tensors for each class
                class_0 = np.where(label == 0, 1, 0)  # Channel for class 0
                class_1 = np.where(label == 1, 1, 0)  # Channel for class 1

                # Concatenate the class channels to create the output tensor
                label = np.stack([class_0, class_1], axis=0)
            else:
                label = np.reshape(label,(1,self.wid,self.hit))

        return torch.Tensor(image),torch.LongTensor(label)
    


class MonuSegOnlyTestDataSet(Dataset):
    def __init__(self, img_dir, config):
        self.img_dir = img_dir
        self.config = config
        self.wid = 512 # default value and is replaced later 
        self.hit = 512 # default value and is replaced later 
        logging.basicConfig(filename=self.config["log"] + "dataloader.log", filemode='w', 
                        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.len = 800
        return 
    
    def __len__(self):
        return len(os.listdir(self.img_dir))//2

    def __getitem__(self, index):
        img_paths  = sorted(os.listdir(self.img_dir))
        #logging.debug(img_paths)
        #logging.debug(os.path.join(self.img_dir,str(index+1)+img_paths[index][-4:]))
        #print(self.config["input_img_type"])
        if self.config["input_img_type"] == "rgb":
            image = cv2.imread(os.path.join(self.img_dir,str(index)+img_paths[0][-4:]))/255
            # Normalize image
            #image = normalize_image(image)
        else:
            image = cv2.imread(os.path.join(self.img_dir,str(index)+img_paths[0][-4:]),cv2.IMREAD_GRAYSCALE)/255
        

        label = cv2.imread(os.path.join(self.img_dir,str(index)+'_label'+img_paths[1][-4:]),cv2.IMREAD_GRAYSCALE)

        self.wid = image.shape[0]
        self.hit = image.shape[1]

        #self.wid = self.hit = self.len


        
        image=image[0:self.wid,0:self.hit]
        label=label[0:self.wid,0:self.hit]
        label[label==0] = 0
        label[label==255] = 1
        #plabel = plabel[0:512,0:512]

        #plt.imshow(label*2)
        

        if self.config["input_img_type"] == "rgb":
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image,(1,self.wid,self.hit))

        label = np.reshape(label,(1,self.wid,self.hit))

        return torch.Tensor(image),torch.LongTensor(label)
