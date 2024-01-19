# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkModules.conv_modules_unet3p import unetConv2, BlurPool2D, MaxBlurPool2d, MultiScaleAttentionBlock, EigenDecomposition, TopKFeatures, DropBlock
from init_weights import init_weights
from networkModules.DCA.dual_cross_attention import DCA
from guided_filter_pytorch.guided_filter import ConvGuidedFilter, FastGuidedFilter

'''
    UNet 3+
'''
class UNet_3PlusShort(nn.Module):

    def __init__(self, config, in_channels=1, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3PlusShort, self).__init__()
        self.is_deconv = is_deconv
        if config["input_img_type"] == "rgb":
            in_channels = 3    
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.kernel_size = config["kernel_size"]
        self.ch = config["channel"]
        n_classes = config["num_classes"]
        self.dropout = nn.Dropout2d(p=config["dropout"])
        self.useMaxBPool = config["use_maxblurpool"]

        self.img_size = config["finalTileHeight"]

        #DCA flag
        try:
            self.DCAFlag = config["DCAFlag"]
        except:
            self.DCAFlag = False
        self.patch_size = 8
        self.patch = self.img_size // self.patch_size
        spatial_att=True,
        channel_att=True,
        spatial_head_dim=[4, 4, 4, 4],
        channel_head_dim=[1, 1, 1, 1],

        # Guided filter
        if config["guidedFilter"]:
            self.guildedFilterFlag = True
        else:
            self.guildedFilterFlag = False
            
        #self.guided_filter_module = ConvGuidedFilter(radius=2)
        self.guided_filter_module = FastGuidedFilter(r=2, eps=1e-2)

        # Drop out or drop block
        self.dropoutFlag = False
        self.dropblockFlag = False
        try:
            self.dropBlock = DropBlock(block_size=config["dropBlockSize"], keep_prob=config["dropBlockProb"])
        except:
            pass
        
        supportedActivations = ["relu", "GLU"]
        if config["activation"] not in supportedActivations:
            raise Exception("Activation function not supported. Supported activations: {}".format(supportedActivations))
        self.activation = config["activation"]
        
        try:
            self.eigen_decompositionFlag = config["eigen_decomposition"]
        except:
            self.eigen_decompositionFlag = False

        try:
            self.top_k_featuresFlag = config["top_k_features"]
        except:
            self.top_k_featuresFlag = False

        try:
            self.multiScaleAttention = config["multiScaleAttention"]
            self.basicAttention  = True
        except:
            self.multiScaleAttention = False


        # self.ch, original paper uses channel size of 64, while we use 16
        # uses relu activation by default
        #filters = [self.ch, self.ch * 2, self.ch * 4, self.ch * 8]
        filters = [self.ch, self.ch * 2, self.ch * 4, self.ch * 4]
        #print(f"Filters: {filters}")
        #filters = [64, 128, 256, 512, 1024]
        if self.eigen_decompositionFlag:
            self.eigen_decomposition = EigenDecomposition()
        if self.top_k_featuresFlag:
            self.top_k_features = TopKFeatures(k=self.top_k_featuresFlag)  



        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, ks=self.kernel_size, act=self.activation)
        if self.useMaxBPool:
            self.maxpool1 = MaxBlurPool2d(kernel_size=2)
        else:
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, ks=self.kernel_size, act=self.activation)
        if self.useMaxBPool:
            self.maxpool2 = MaxBlurPool2d(kernel_size=2)
        else:
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, ks=self.kernel_size, act=self.activation)
        if self.useMaxBPool:
            self.maxpool3 = MaxBlurPool2d(kernel_size=2)    
        else:
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, ks=self.kernel_size, act=self.activation)
        if self.useMaxBPool:
            self.maxpool4 = MaxBlurPool2d(kernel_size=2)
        else:
            self.maxpool4 = nn.MaxPool2d(kernel_size=2)


        ## Attention
        if self.multiScaleAttention:
            self.multi_scale_attention1 = MultiScaleAttentionBlock(filters[0])
            self.multi_scale_attention2 = MultiScaleAttentionBlock(filters[1])
            self.multi_scale_attention3 = MultiScaleAttentionBlock(filters[2])
            self.multi_scale_attention4 = MultiScaleAttentionBlock(filters[3])

        if self.DCAFlag:
            self.DCA = DCA(n=1,                                            
                                features = filters,                                                                                                              
                                strides=[self.patch_size, self.patch_size // 2, self.patch_size // 4, self.patch_size // 8],
                                patch=self.patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                                            )  



        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 4
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, self.kernel_size, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, self.kernel_size, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, self.kernel_size, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, self.kernel_size, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, self.kernel_size, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)


        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, self.kernel_size, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)


        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)


        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, self.kernel_size, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def setdropoutFlag(self, flag):
        self.dropoutFlag = flag

    def setdropblockFlag(self, flag):
        self.dropblockFlag = flag

    def forward(self, input):

        inputs, label = input

        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        if self.multiScaleAttention:
            h1 = self.multi_scale_attention1(h1)

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        if self.multiScaleAttention:
            h2 = self.multi_scale_attention2(h2)

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        if self.multiScaleAttention:
            h3 = self.multi_scale_attention3(h3)

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        if self.multiScaleAttention:
            h4 = self.multi_scale_attention4(h4)

        #h5 = self.maxpool4(h4)
        #hd5 = self.conv5(h5)  # h5->20*20*1024

        if self.eigen_decompositionFlag:
            h4 = self.eigen_decomposition(h4)
        if self.top_k_featuresFlag:
            h4 = self.top_k_features(h4)

        if self.DCAFlag:
            h1, h2, h3, h4 = self.DCA([h1, h2, h3, h4])


        # dropout
        if self.dropoutFlag:
            h4 = self.dropout(h4)

        if self.dropblockFlag:
            h4 = self.dropBlock(h4)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        #hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))

        ## Attention
        if self.multiScaleAttention and not self.basicAttention:
            h1_PT_hd4_attn = self.multi_scale_attention1(h1_PT_hd4)
            h2_PT_hd4_attn = self.multi_scale_attention2(h2_PT_hd4)
            h3_PT_hd4_attn = self.multi_scale_attention3(h3_PT_hd4)
            hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
                torch.cat((h1_PT_hd4_attn, h2_PT_hd4_attn, h3_PT_hd4_attn, h4_Cat_hd4), 1)))) # hd4->40*40*UpChannels
            hd4 = self.multi_scale_attention4(hd4)
        else:
            hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
                torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        #hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        
        ## Attention
        if self.multiScaleAttention and not self.basicAttention:
            h1_PT_hd3_attn = self.multi_scale_attention1(h1_PT_hd3)
            h2_PT_hd3_attn = self.multi_scale_attention2(h2_PT_hd3)
            hd4_UT_hd3_attn = self.multi_scale_attention4(hd4_UT_hd3)
            hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
                torch.cat((h1_PT_hd3_attn, h2_PT_hd3_attn, h3_Cat_hd3, hd4_UT_hd3_attn), 1)))) # hd3->80*80*UpChannels
            h3 = self.multi_scale_attention3(h3)
        else:
            hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
                torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        #hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        
        
        ## Attention
        if self.multiScaleAttention and not self.basicAttention:
            h1_PT_hd2_attn = self.multi_scale_attention1(h1_PT_hd2)
            hd3_UT_hd2_attn = self.multi_scale_attention3(hd3_UT_hd2)
            hd4_UT_hd2_attn = self.multi_scale_attention4(hd4_UT_hd2)
            hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
                torch.cat((h1_PT_hd2_attn, h2_Cat_hd2, hd3_UT_hd2_attn, hd4_UT_hd2_attn), 1)))) # hd2->160*160*UpChannels
            hd2 = self.multi_scale_attention2(hd2)
        else:
            hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
                torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2), 1)))) # hd2->160*160*UpChannels


        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        #hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        
        ## Attention
        if self.multiScaleAttention and not self.basicAttention:
            hd2_UT_hd1_attn = self.multi_scale_attention2(hd2_UT_hd1)
            hd3_UT_hd1_attn = self.multi_scale_attention3(hd3_UT_hd1)
            hd4_UT_hd1_attn = self.multi_scale_attention4(hd4_UT_hd1)
            hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
                torch.cat((h1_Cat_hd1, hd2_UT_hd1_attn, hd3_UT_hd1_attn, hd4_UT_hd1_attn), 1)))) # hd1->320*320*UpChannels
            hd1 = self.multi_scale_attention1(hd1)
        else:
            hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
                torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1), 1)))) # hd1->320*320*UpChannels


        d1 = self.outconv1(hd1)  # d1->320*320*n_classes

        # Guided filter
        if self.guildedFilterFlag:
            d1 = self.guided_filter_module(label, d1, label)
        return torch.sigmoid(d1)