import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from  networkModules.conv_modules import *


class UNet(nn.Module):
    def __init__(self,config,in_ch=1):
        super(UNet,self).__init__()

        #print(config)
        '''if config.activation == 'siren':
            self.apply(self._init_weights)'''
        
        self.apply(self._init_weights2)


        self.anti_type = config["anti_type"]
        self.attention = config["attention"]
        self.ch = config["channel"]
        self.depth = config["depth"]
        self.activation = config["activation"]

        self.dropout = nn.Dropout2d(p=config["dropout"])


        # parameter for loss function
        # self.xi = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        self.contract1  = BigConv2times(in_ch,self.ch,config)
        
        if self.anti_type == 'down' or self.anti_type=='down_up':
            self.pool1  = nn.MaxPool2d(kernel_size=2, stride=1)
            self.blur1  = BlurPool(self.ch)
        else:
            self.pool1  = nn.MaxPool2d(kernel_size=2)
            
        self.contract2  = BigConv2times(self.ch,self.ch*2,config)
        
        if self.anti_type=='down'  or self.anti_type=='down_up':
            self.pool2 = nn.MaxPool2d(kernel_size=2,stride=1)
            self.blur2 = BlurPool(self.ch*2)
        else:
            self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.contract3  = BigConv2times(self.ch*2,self.ch*4,config)

        if self.anti_type=='down' or self.anti_type=='down_up':
            self.pool3 = nn.MaxPool2d(kernel_size=2,stride=1)
            self.blur3 = BlurPool(self.ch*4)
        else:
            self.pool3 = nn.MaxPool2d(kernel_size=2)
            
        self.contract4  = BigConv2times(self.ch*4, self.ch*8,config)

        if self.anti_type=='down' or self.anti_type=='down_up':
            self.pool4 = nn.MaxPool2d(kernel_size=2,stride=1)
            self.blur4 = BlurPool(self.ch*8)
        else:
            self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.contract5  = BigConv2times(self.ch*8,self.ch*16,config)

        # [B,1024,32,32]
        self.mhsa_h = nn.MultiheadAttention(32,4,batch_first=True)
        self.mhsa_w = nn.MultiheadAttention(32,4,batch_first=True)

        '''
        self.decv1      = nn.Conv2d(1024,2048,3,1,1)
        self.pixel1     = nn.PixelShuffle(2)
        self.upsample1  = Conv2times(1024,512)

        self.decv2      = nn.Conv2d(512,1024,3,1,1)
        self.pixel2     = nn.PixelShuffle(2)
        self.upsample2  = Conv2times(512,256)

        self.decv3      = nn.Conv2d(256,512,3,1,1)
        self.pixel3     = nn.PixelShuffle(2)
        self.upsample3  = Conv2times(256,128)

        self.decv4      = nn.Conv2d(128,256,3,1,1)
        self.pixel4     = nn.PixelShuffle(2)
        self.upsample5  = Conv2times(128,64)

       
        '''
        if self.anti_type=='up' or self.anti_type=='down_up':
            self.decv1      = nn.Conv2d(self.ch*16,self.ch*32,3,1,1)
            self.pixel1     = nn.PixelShuffle(2)
        else:
            self.decv1      = nn.ConvTranspose2d(self.ch*16,self.ch*8,2,2)
            
        self.upsample1  = Conv2times(self.ch*16,self.ch*8,config)

        if self.anti_type=='up' or self.anti_type=='down_up':
            self.decv2      = nn.Conv2d(self.ch*8,self.ch*16,3,1,1)
            self.pixel2     = nn.PixelShuffle(2)
        else:
            self.decv2      = nn.ConvTranspose2d(self.ch*8,self.ch*4,2,2)

        self.upsample2  = Conv2times(self.ch*8,self.ch*4,config)

        if self.anti_type=='up' or self.anti_type=='down_up':
            self.decv3      = nn.Conv2d(self.ch*4,self.ch*8,3,1,1)
            self.pixel3     = nn.PixelShuffle(2)
        else:
            self.decv3      = nn.ConvTranspose2d(self.ch*4,self.ch*2,2,2)
            
        self.upsample3  = Conv2times(self.ch*4,self.ch*2,config)

        if self.anti_type=='up' or self.anti_type=='down_up':
            self.decv4      = nn.Conv2d(self.ch*2,self.ch*4,3,1,1)
            self.pixel4     = nn.PixelShuffle(2)
        else:
            self.decv4      = nn.ConvTranspose2d(self.ch*2,self.ch,2,2)
        
        self.upsample5  = Conv2times(self.ch*2,self.ch,config)

        self.decv5      = nn.Conv2d(self.ch,config["num_classes"],1)
        self.sigmoid    = nn.Sigmoid()
        self.softmax    = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()
        self.swish = nn.SiLU()

        
        return
    
    def _init_weights(self,module):
        if isinstance(module,nn.Conv2d):
            self.weight.data.uniform_(-np.sqrt(6/(self.in_channels*self.kernel_size*self.kernel_size)),np.sqrt(6/(self.in_channels*self.kernel_size*self.kernel_size)))
            self.bias.data.uniform_(-np.sqrt(6/(self.in_channels*self.kernel_size*self.kernel_size)),np.sqrt(6/(self.in_channels*self.kernel_size*self.kernel_size)))
        return
    

    def _init_weights2(self,module):
        if isinstance(module,nn.Conv2d):
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_parameter('bias', None)
        if self.bias is not None:
          nn.init.constant_(self.bias, 0)
        elif isinstance(self, nn.BatchNorm2d):
            nn.init.normal(self.weight, 1.0, 0.02)
            nn.init.constant_(self.bias, 0)
        elif isinstance(self, nn.Linear):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.constant_(self.bias.data, 0)
        elif isinstance(module, nn.Conv2d) and module is self.sigmoid:
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def masked_concat(self,left,right,ratio:float):
        left = random_masking(left,ratio,left.shape[2]//16)
        x = torch.cat((left,right),1)
        return x

    def forward(self, x):
        y1 = self.contract1(x)
        y2 = self.pool1(y1)
        
        if self.anti_type=='down' or self.anti_type=='down_up':
            y2 = self.blur1(y2)
            
        y3 = self.contract2(y2)
        y4 = self.pool2(y3)
        
        if self.anti_type=='down' or self.anti_type=='down_up':
            y4 = self.blur2(y4)
            
        y5 = self.contract3(y4)
        y6 = self.pool3(y5)
        
        if self.anti_type=='down' or self.anti_type=='down_up':
            y6 = self.blur3(y6)
            
        y7 = self.contract4(y6)
        
        if self.depth == 5:

            y8 = self.pool4(y7)
            if self.anti_type=='down' or self.anti_type=='down_up':
                y8 = self.blur4(y8)
            y9 = self.contract5(y8)

            # Adding dropout layer
            y9 = self.dropout(y9)

            #[B,1024,32,32]
            if self.attention:        
                at = torch.reshape(y9,(y9.shape[0]*y9.shape[1],y9.shape[2],y9.shape[3]))
                at,_ = self.mhsa_h(at,at,at)
                at = torch.reshape(at,(y9.shape[0],y9.shape[1],y9.shape[2],y9.shape[3]))
                at = at+y9
                at = torch.reshape(at,(y9.shape[0]*y9.shape[1],y9.shape[2],y9.shape[3]))
                at = torch.permute(at,(0,2,1))
                at,_ = self.mhsa_w(at,at,at)
                at = torch.permute(at,(0,2,1))
                y9 = torch.reshape(at,(y9.shape[0],y9.shape[1],y9.shape[2],y9.shape[3]))

            y10 = self.decv1(y9)

            if self.anti_type=='up' or self.anti_type=='down_up':
                y10 = self.pixel1(y10)

            y11 = self.upsample1(self.masked_concat(y7,y10,0.7))

        if self.depth == 4:
            y12 = self.decv2(y7)
        else:
            y12 = self.decv2(y11)
        
        if self.anti_type=='up' or self.anti_type=='down_up':
            y12 = self.pixel2(y12)
            
        y13 = self.upsample2(self.masked_concat(y5,y12,0.7))
        y14 = self.decv3(y13)
        
        if self.anti_type=='up' or self.anti_type=='down_up':
            y14 = self.pixel3(y14)
            
        y15 = self.upsample3(self.masked_concat(y3,y14,0.7))
        y16 = self.decv4(y15)
        
        if self.anti_type=='up' or self.anti_type=='down_up':
            y16= self.pixel4(y16)
            
        y17 = self.upsample5(self.masked_concat(y1,y16,0.7))
        y18 = self.decv5(y17)
        
        y18 = self.sigmoid(y18)
        #y18 = self.softmax(y18)
        return y18



        

        