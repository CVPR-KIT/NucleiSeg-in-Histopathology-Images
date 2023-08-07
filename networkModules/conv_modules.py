import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
import math

class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class Siren(nn.Module):
    def __init__(self):
        super().__init__()
        self.sin = torch.sin
        return
    
    def forward(self,x):
        return self.sin(x)

class SirenConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        return
    
    def forward(self, x):
        return super.forward(x)

# Calculate the padding size for a given kernel size
def getPadSize(inChannel,outChannel,kernel_size, dilation, stride):
    #pad = (outChannel-1)*stride - inChannel + dilation*(kernel_size-1) + 1
    #pad = (((outChannel - 1) * stride) - inChannel + (dilation * (kernel_size - 1)) + 1) // 2
    pad = (kernel_size*dilation -1) // 2 
    return pad

# print model information
def printStat(inChannel,outChannel,kernel_size, dilation, stride):
    pad = getPadSize(inChannel,outChannel,kernel_size, dilation, stride)
    print(f"in_c = {inChannel}, out_c = {outChannel} kernel_size = {kernel_size}, dilation = {dilation}, stride = {stride}, padding = {pad}")


class Conv2times(nn.Module):
    def __init__(self,inChannel,outChannel,config):
        super(Conv2times,self).__init__()
        pad = getPadSize(inChannel,outChannel,config["kernel_size"],config["dilation"],1)
        #printStat(inChannel,outChannel,config.kernel_size,config.dilation,1)
        self.kernel_size = config["kernel_size"]
        self.conv1 = nn.Conv2d(inChannel,outChannel,self.kernel_size,1,padding=pad,padding_mode='reflect', bias=True, dilation=config["dilation"])
        self.norm1 = nn.BatchNorm2d(outChannel)
        if(config["activation"] == 'relu'):
            self.actv1 = nn.ReLU()
        elif(config["activation"]  == 'siren'):
            self.actv1 = Siren()
        elif config["activation"]  == 'leakyrelu':
            self.actv1 = nn.LeakyReLU()
        elif config["activation"]  == 'swish':
            self.actv1 = nn.SiLU()
        pad = getPadSize(outChannel,outChannel,config["kernel_size"],config["dilation"],1)
        #printStat(inChannel,outChannel,config.kernel_size,config.dilation,1)
        self.conv2 = nn.Conv2d(outChannel,outChannel,self.kernel_size,1,padding=pad,padding_mode='reflect', bias=True, dilation=config["dilation"])
        self.norm2 = nn.BatchNorm2d(outChannel)
        if(config["activation"]  == 'relu'):
            self.actv2 = nn.ReLU()
        elif(config["activation"]  == 'siren'):
            self.actv2 = Siren()
        elif config["activation"]  == 'leakyrelu':
            self.actv2 = nn.LeakyReLU()
        elif config["activation"]  == 'swish':
            self.actv2 = nn.SiLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.actv1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return self.actv2(y)

class BigConv2times(nn.Module):
    def __init__(self,inChannel,outChannel,config):
        super(BigConv2times,self).__init__()
        pad = getPadSize(inChannel,outChannel,config["kernel_size"],config["dilation"],1)
        self.kernel_size = config["kernel_size"]
        #printStat(inChannel,outChannel,config.kernel_size,config.dilation,1)
        self.conv1 = nn.Conv2d(inChannel,outChannel,self.kernel_size,1,pad,padding_mode='reflect', bias=True, dilation=config["dilation"])#,dilation=2)
        self.norm1 = nn.BatchNorm2d(outChannel)
        if(config["activation"]  == 'relu'):
            self.actv1 = nn.ReLU()
        elif(config["activation"]  == 'siren'):
            self.actv1 = Siren()
        elif config["activation"]  == 'leakyrelu':
            self.actv1 = nn.LeakyReLU()
        elif config["activation"]  == 'swish':
            self.actv1 = nn.SiLU()
        pad = getPadSize(outChannel,outChannel,config["kernel_size"],config["dilation"],1)
        #printStat(inChannel,outChannel,config.kernel_size,config.dilation,1)
        self.conv2 = nn.Conv2d(outChannel,outChannel,self.kernel_size,1,pad,padding_mode='reflect', bias=True, dilation=config["dilation"])#,dilation=2)
        self.norm2 = nn.BatchNorm2d(outChannel)
        if(config["activation"]  == 'relu'):
            self.actv2 = nn.ReLU()
        elif(config["activation"]  == 'siren'):
            self.actv2 = Siren()
        elif config["activation"]  == 'leakyrelu':
            self.actv2 = nn.LeakyReLU()
        elif config["activation"]  == 'swish':
            self.actv2 = nn.SiLU()
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.actv1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return self.actv2(y)

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)        
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute2D(channel)

    def forward(self, x):
        b, c, h, w = x.size()
        # pe = self.positional_encoding_2d(c, h, w)
        pe = self.pe(x)
        x = x + pe
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) /
                         math.sqrt(c))  #[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x

def random_masking( x, mask_ratio, mask_width):
    N,C,W,H = x.shape
    
    num_masks = W//mask_width

    mask = torch.rand((N,C,num_masks,num_masks),device=x.device)
    mask[mask<mask_ratio] = 0
    mask[mask>mask_ratio] = 1
    mask = F.interpolate(mask,(x.shape[2:]))
    

    return mask * x
    