import torch
import torch.nn as nn
import torch.nn.functional as F
from init_weights import init_weights
from torch import linalg as LA


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1, act='relu'):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                if act == 'relu':
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                if act == 'GLU':
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size*2, ks, s, p),
                                         nn.BatchNorm2d(out_size*2),
                                         nn.GLU(dim=1), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                if act == 'relu':
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                if act == 'GLU':
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size*2, ks, s, p),
                                         nn.GLU(dim=1), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class BlurPool2D(nn.Module):
    def __init__(self, kernel_size, stride):
        super(BlurPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.weight = self._get_weights(kernel_size)

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        return F.conv2d(x, self.weight, stride=self.stride, groups=x.size(1))

    def _get_weights(self, kernel_size):
        # create a 1D kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size))
        # normalize the kernel to make it a mean filter
        kernel /= kernel_size ** 2
        # expand dimensions to match the input tensor
        kernel = kernel.expand((3, -1, -1, -1))
        return kernel
    
class MaxBlurPool2D_0(nn.Module):
    def __init__(self, kernel_size, stride, in_channels):
        super(MaxBlurPool2D_0, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        #print(f"inside maxblurpool2d: c={in_channels}, k={kernel_size}")
        self.weight = self._get_weights(kernel_size, in_channels)


    def forward(self, x):
        x = self.maxpool(x)  # Apply max-pooling
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        weight = self.weight.to(x.device)
        return F.conv2d(x, weight, stride=1, groups=x.size(1))  # Apply blur

    def _get_weights(self, kernel_size, num_channels):
        # create a 1D kernel
        kernel = torch.ones((num_channels, 1, kernel_size, kernel_size))
        # normalize the kernel to make it a mean filter
        kernel /= kernel_size ** 2
        return kernel

class MaxBlurPool2d_old(nn.Module):
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
    
class MaxBlurPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(MaxBlurPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize blur kernel
        self.blur_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        self.blur_kernel = self.blur_kernel.view(1, 1, 3, 3) / 16.0

    def forward(self, x):
        # Max pooling
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        # Move blur kernel to the same device as the input
        blur_kernel = self.blur_kernel.to(x.device).repeat(x.size(1), 1, 1, 1)
        
        # Apply blurring
        x = F.conv2d(x, blur_kernel, groups=x.size(1), padding=1)
        
        return x



class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, num_channels):
        super(MultiScaleAttentionBlock, self).__init__()
        self.num_channels = num_channels

        # Spatial attention
        self.spatial_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        
        # Channel attention
        self.channel_conv1 = nn.Conv2d(num_channels, num_channels//4, kernel_size=1)
        self.channel_conv2 = nn.Conv2d(num_channels//4, num_channels, kernel_size=1)
        
        # Edge attention
        self.sobel_filter = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        sobel_kernel = sobel_kernel.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, 3, 3]
        self.sobel_filter.weight = nn.Parameter(sobel_kernel)

        self.sobel_filter.requires_grad = False

    def forward(self, x):
        # Spatial Attention
        spatial_attn_map = torch.sigmoid(self.spatial_conv(x))

        # Channel Attention
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)
        channel_attn_map = self.channel_conv2(F.relu(self.channel_conv1(avg_pool)))
        channel_attn_map = torch.sigmoid(channel_attn_map)

        # Edge Attention
        edge_attn_maps = []
        for i in range(self.num_channels):
            single_channel = x[:, i:i+1, :, :]
            
            '''print("Shape of sobel_filter.weight:", self.sobel_filter.weight.shape)
            print("Shape of single_channel:", single_channel.shape)

            '''
            with torch.no_grad():
                edge_attn_map = F.conv2d(single_channel, self.sobel_filter.weight, stride=(1, 1), padding=1)

            edge_attn_map = torch.sigmoid(edge_attn_map)
            edge_attn_maps.append(edge_attn_map)
        edge_attn_map = torch.cat(edge_attn_maps, dim=1)
        
        # Multi-Scale Attention
        multi_scale_attn_map = spatial_attn_map * channel_attn_map * edge_attn_map
        return x * multi_scale_attn_map
    
class EigenDecomposition(nn.Module):
    def forward(self, x):
        cov_matrix = torch.matmul(x.transpose(-1, -2), x)
        eigenvalues, eigenvectors = LA.eigh(cov_matrix)
        enhanced_x = torch.matmul(x, eigenvectors)
        return enhanced_x

class TopKFeatures(nn.Module):
    def __init__(self, k):
        super(TopKFeatures, self).__init__()
        self.k = k

    def forward(self, x):
        values, _ = torch.topk(x, self.k, dim=1)
        return values