import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
# 论文：EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation, CVPR2024
# 论文地址：https://arxiv.org/pdf/2405.06880

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
                channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

# Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, reso,kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.share_channels = 8
        self.reso = reso
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                    padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down,from_up):
        # print("from_up1:",from_up.shape)
        from_up = self.up_dwc(from_up)
        # print("from_up2:",from_up.shape)
        # print("from_down:",from_down.shape)
        from_up = channel_shuffle(from_up, self.in_channels)
        from_up = self.pwc(from_up)
        # print(x.shape.shape)
        x = torch.cat((from_up,from_down),1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
    
    # def forward(self, x):
    #     x = self.up_dwc(x)
    #     x = channel_shuffle(x, self.in_channels)
    #     x = self.pwc(x)
    #     return x
    
    def weight_init(self):
        # Initialize the up_dwc part
        for m in self.up_dwc:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He initialization for Conv2d
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # Initialize bias to 0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # BatchNorm weights to 1
                init.constant_(m.bias, 0)  # BatchNorm bias to 0

        # Initialize the pwc part
        for m in self.pwc:
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)  # Xavier initialization for Conv2d (1x1 conv)
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # Initialize bias to 0

        # Initialize PReLU if exists (in the activation layer)
        for m in self.up_dwc:
            if isinstance(m, nn.PReLU):
                init.constant_(m.weight, 0.25)  # Initialize PReLU weights


if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64) #B C H W

    block = EUCB(in_channels=32, out_channels=64)
    block.weight_init()

    print(input.size())

    output = block(input)
    print(output.size())