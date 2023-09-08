import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """ [(Conv2d) => (BN) => (ReLu)] * 2 """
    
    def __init__(self,in_channels,out_channels, ks) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,ks,padding="same",stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels,out_channels,ks,padding="same",stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()      
            )

    def forward(self,x):
        return self.double_conv(x)

class DownSample(nn.Module):
    """ MaxPool => DoubleConv """
    def __init__(self,in_channels,out_channels, ks, useMaxBPool) -> None:
        super().__init__()
        if useMaxBPool:
            self.down_sample = nn.Sequential(
                MaxBlurPool2d(kernel_size=2),
                DoubleConv(in_channels,out_channels, ks)
            )
        else:
            self.down_sample = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels,out_channels, ks)
            )
    def forward(self,x):
        x  = self.down_sample(x)
        return x

class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels,c:int, ks) -> None:
        """ UpSample input tensor by a factor of `c`
                - the value of base 2 log c defines the number of upsample 
                layers that will be applied
        """
        super().__init__()
        n = 0 if c == 0 else int(math.log(c,2))

        self.upsample = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels,in_channels,2,2) for i in range(n)]
        )
        self.conv_3 = nn.Conv2d(in_channels,out_channels,ks,padding="same",stride=1)

    def forward(self,x):
        for layer in self.upsample:
            x = layer(x)
        return self.conv_3(x)        

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, ks):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks,padding=1)
    def forward(self, x):
        return self.conv(x)
    

class MaxBlurPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(MaxBlurPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Max pooling
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # Blurring kernel
        blur_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        blur_kernel = blur_kernel.view(1, 1, 3, 3)
        blur_kernel = blur_kernel / blur_kernel.sum()
        blur_kernel = blur_kernel.repeat(x.size(1), 1, 1, 1).to(x.device)

        # Applying the blur using the 'depthwise' convolution
        x = F.conv2d(x, blur_kernel, groups=x.size(1), padding=1)

        return x