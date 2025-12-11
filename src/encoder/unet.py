import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from src.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
from torch_scatter import scatter_mean, scatter_max, scatter_add, scatter_softmax
hidden_size = 32 
depth = 4

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups),
        nn.BatchNorm2d(out_channels)
    )

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.Sequential(
            nn.ConvTranspose2d(  #ConvTranspose2d:转置卷积（反卷积），一种上采样的操作
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
        )
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=groups,
            stride=1,
            bias=bias),
        nn.BatchNorm2d(out_channels)
    )
    
def conv1x12(in_channels, out_channels, groups=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=groups,
            stride=1),
        # nn.BatchNorm3d(out_channels)
    )

class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    
    def __init__(self, in_channels, out_channels, reso, pooling):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.share_channels = 8 # 共享的通道数，用于特征分组
        self.pooling = pooling
        self.reso = reso
        self.padding = 0.1
        self.c_dim = 32
        self.sample_mode = 'bilinear'
        
        # 定义卷积层，包含两个 3x3 卷积操作和 ReLU 激活函数
        self.conv1 = nn.Sequential(
            conv3x3(self.in_channels, self.out_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.out_channels, self.out_channels),
        )
        # 定义最大池化层，池化核为 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Keys的全连接层
        self.fc_k = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )

        # Values的全连接层，包含多层线性层和 ReLU 激活
        self.fc_v = nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels,2*out_channels),
                        nn.ReLU(),
                        nn.Linear(2*out_channels, out_channels)
                    )
        # Queries的 1x1 卷积层，用于降低特征维度
        self.conv_q = conv1x12(out_channels, out_channels)

        # 位置编码的全连接层，增强位置特征
        self.fc_pos = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(inplace=True), nn.Linear(2, out_channels))

        # 权重特征的全连接层，进一步处理特征
        self.fc_w = nn.Sequential(nn.Linear(out_channels, out_channels // self.share_channels),
                            nn.BatchNorm1d(out_channels // self.share_channels), nn.ReLU(inplace=True))

        # 权重卷积，用于卷积操作后进一步处理
        self.weight_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # 最后的 1x1 卷积，用于特征合并
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)


       
    def generate_plane_features(self, p, c, k, q, channel, reso_plane, plane='xy'):
        
        b, n, _ = p.shape

        # 对输入点坐标进行归一化，将其转换到 0 到 1 的范围内
        p_nor = normalize_coordinate(p.clone(), padding=self.padding, plane=plane) 
        index = coordinate2index(p_nor, reso_plane)
        
        pos_enc = p_nor - (p_nor * reso_plane).floor() / reso_plane #将点云从全局坐标系转换到点所在网格的局部坐标系，用于捕捉局部信息
        for i, layer in enumerate(self.fc_pos): # 通过多个全连接层对位置编码进一步处理，增强位置特征
            pos_enc = layer(pos_enc.transpose(1, 2).contiguous()).transpose(1, 2) if i == 1 else layer(pos_enc)
        
        # 为每个批次的点生成索引，用于后续操作
        batch_index = torch.arange(p.size(0), device=p.device).repeat_interleave(p.size(1))
        # batch_index(batch_size * num_points,)
        
        # 计算注意力权重 w，使用键特征 k 减去查询特征 q 并加上位置编码 pos_enc
        w = k - q.reshape(-1, channel, reso_plane**2)[batch_index, :, index.reshape(-1)].reshape(k.shape[0], k.shape[1], channel) + pos_enc
        w = w.reshape(k.shape[0]*k.shape[1], -1)
        w = self.fc_w(w)
        w = w.reshape(k.shape[0], k.shape[1], -1)
        
        # 对权重进行 softmax 归一化处理
        weights_soft = scatter_softmax(w.transpose(1, 2), index).transpose(1, 2).reshape(b*n, -1)
        
        value = (c + pos_enc).reshape(b*n, -1) # 这里的c对应v
        # 将其按共享通道数进行分组，并根据计算的权重 weights_soft 对特征进行加权计算，最终得到每个点的特征
        value_grid = (value.reshape(b*n, self.share_channels, self.out_channels // self.share_channels) * weights_soft.unsqueeze(-2)).view(b*n, -1)
        value_grid = value_grid.reshape(b, n, -1).transpose(1, 2)
        

        fea_plane = c.new_zeros(p.size(0), channel, reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T

        # 将每个点的特征值根据索引添加到平面特征中
        fea_plane = scatter_add(value_grid, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), channel, reso_plane, reso_plane) # sparce matrix (B x 512 x reso x reso)

        return fea_plane


    def sample_plane_feature(self, p, c, plane='xz'):
        # 根据给定的采样坐标，使用双线性插值的方式来获得点特征
        # c:平面特征
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)

        return c

    def forward(self, p, x, plane_type, c=None):
        #p:点云
        #x:fea,平面特征
        #c:点特征

        b, n, _ = c.shape
        
        if self.pooling:
            x['xy'] = self.pool(x['xy'])
            x['xz'] = self.pool(x['xz'])
            x['yz'] = self.pool(x['yz'])
            
        x['xy'] = F.relu(self.conv1(x['xy']))
        x['xz'] = F.relu(self.conv1(x['xz']))
        x['yz'] = F.relu(self.conv1(x['yz']))

 
        x_down = {}
        x_down['xy'] = x['xy']
        x_down['xz'] = x['xz']
        x_down['yz'] = x['yz']

        channel = x['xy'].shape[1]
        reso_plane = x['xy'].shape[2]
        
        c = c.reshape(b*n, -1)
        # 计算Keys和Values
        k = self.fc_k(c)
        v = self.fc_v(c)
        c = c.reshape(b, n, -1)
        k = k.reshape(b, n, -1)
        v = v.reshape(b, n, -1)
        
        q = {}
        # 计算Queries
        q['xy'] = self.conv_q(x['xy'])
        q['xz'] = self.conv_q(x['xz'])
        q['yz'] = self.conv_q(x['yz'])
        
        if 'xy' in plane_type:
            x['xy'] = self.generate_plane_features(p, v, k, q['xy'], channel, reso_plane, plane='xy')
            x['xy'] = self.weight_conv(x['xy'])
            x['xy'] = x['xy'] + self.conv1x1(x_down['xy'])
        if 'xz' in plane_type:
            x['xz'] = self.generate_plane_features(p, v, k, q['xz'], channel, reso_plane, plane='xz')
            x['xz'] = self.weight_conv(x['xz'])
            x['xz'] = x['xz'] + self.conv1x1(x_down['xz'])
        if 'yz' in plane_type:
            x['yz'] = self.generate_plane_features(p, v, k, q['yz'], channel, reso_plane, plane='yz')
            x['yz'] = self.weight_conv(x['yz'])
            x['yz'] = x['yz'] + self.conv1x1(x_down['yz'])

        c = v
        
        if 'xy' in plane_type:
            c += self.sample_plane_feature(p, x['xy'], 'xy').transpose(1, 2)
        if 'xz' in plane_type:
            c += self.sample_plane_feature(p, x['xz'], 'xz').transpose(1, 2)
        if 'yz' in plane_type:
            c += self.sample_plane_feature(p, x['yz'], 'yz').transpose(1, 2)
        
        before_pool = {}
        before_pool['xy'] = x['xy']
        before_pool['xz'] = x['xz']
        before_pool['yz'] = x['yz']

        return x, before_pool, c


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
    def __init__(self, in_channels, out_channels, reso,kernel_size=3, stride=1, activation='relu',merge_mode='concat'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.share_channels = 8
        self.merge_mode = merge_mode
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

        # # 根据分辨率，选择上采样方式。如果是最后一级，则不进行上采样，只进行 1x1 卷积
        # if reso != 64:
        #     self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        # else:
        #     self.upconv_noup = conv1x1(self.in_channels, self.out_channels)
        self.upconv_noup = conv1x1(self.in_channels, self.out_channels)

        self.fc_k = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )

        self.fc_v = nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels,2*out_channels),
                        nn.ReLU(),
                        nn.Linear(2*out_channels, out_channels)
                    )
        
        self.conv_q = conv1x12(out_channels, out_channels)
        self.fc_pos = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(inplace=True), nn.Linear(2, out_channels))
        self.fc_w = nn.Sequential(nn.Linear(out_channels, out_channels // self.share_channels),
                            nn.BatchNorm1d(out_channels // self.share_channels), nn.ReLU(inplace=True))
        
        # 如果采用 concat 的融合模式，定义卷积层来减少通道数
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else: # 否则，直接使用两个 3x3 卷积层
            self.conv1 = nn.Sequential(
                conv3x3(self.in_channels, self.out_channels),
                nn.ReLU(inplace=True),
                conv3x3(self.out_channels, self.out_channels),
            )
        self.weight_conv = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels
            ),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels
            )
        )
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 最终的 1x1 卷积层，用于特征的最终处理
        self.conv_final = conv1x1(out_channels, out_channels)
        self.padding = 0.1
        self.c_dim = 32 
        self.sample_mode = 'bilinear'

    def generate_plane_features(self, p, c, k, q, channel, reso_plane, plane='xy'):
        
        b, n, _ = p.shape

        p_nor = normalize_coordinate(p.clone(), padding=self.padding, plane=plane) # normalize to the range of (0, 1)
        index = coordinate2index(p_nor, reso_plane)
        
        pos_enc = p_nor - (p_nor * reso_plane).floor() / reso_plane
        for i, layer in enumerate(self.fc_pos):
            pos_enc = layer(pos_enc.transpose(1, 2).contiguous()).transpose(1, 2) if i == 1 else layer(pos_enc)
        
        batch_index = torch.arange(p.size(0), device=p.device).repeat_interleave(p.size(1))
        
        w = k - q.reshape(-1, channel, reso_plane**2)[batch_index, :, index.reshape(-1)].reshape(k.shape[0], k.shape[1], channel) + pos_enc
        w = w.reshape(k.shape[0]*k.shape[1], -1)
        w = self.fc_w(w)
        w = w.reshape(k.shape[0], k.shape[1], -1)
        
        weights_soft = scatter_softmax(w.transpose(1, 2), index).transpose(1, 2).reshape(b*n, -1)
        
        value = (c + pos_enc).reshape(b*n, -1)
        value_grid = (value.reshape(b*n, self.share_channels, self.out_channels // self.share_channels) * weights_soft.unsqueeze(-2)).view(b*n, -1)
        value_grid = value_grid.reshape(b, n, -1).transpose(1, 2)
        
        
        fea_plane = c.new_zeros(p.size(0), channel, reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T

        fea_plane = scatter_add(value_grid, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), channel, reso_plane, reso_plane) # sparce matrix (B x 512 x reso x reso)

        return fea_plane


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c
    
    def forward(self, p, from_down, from_up, plane_type, c_last, i):
        
        b, n, _ = c_last.shape
        
        # 根据当前深度选择是否执行上采样或不进行上采样
        if i == 2:
            from_up['xy'] = self.upconv_noup(from_up['xy'])
            from_up['xz'] = self.upconv_noup(from_up['xz'])
            from_up['yz'] = self.upconv_noup(from_up['yz'])
        else:
            from_up['xy'] = self.pwc(channel_shuffle(self.up_dwc(from_up['xy']),self.in_channels))
            from_up['xz'] = self.pwc(channel_shuffle(self.up_dwc(from_up['xz']),self.in_channels))
            from_up['yz'] = self.pwc(channel_shuffle(self.up_dwc(from_up['yz']),self.in_channels))
            # from_up['xy'] = self.upconv(from_up['xy'])
            # from_up['xz'] = self.upconv(from_up['xz'])
            # from_up['yz'] = self.upconv(from_up['yz'])

        # 融合上采样的特征和下采样的特征
        x = {}
        if self.merge_mode == 'concat':
            x['xy'] = torch.cat((from_up['xy'], from_down['xy']), 1)
            x['xz'] = torch.cat((from_up['xz'], from_down['xz']), 1)
            x['yz'] = torch.cat((from_up['yz'], from_down['yz']), 1)
        else:
            x['xy'] = from_up['xy'] + from_down['xy']
            x['xz'] = from_up['xz'] + from_down['xz']
            x['yz'] = from_up['yz'] + from_down['yz']
        
        x_up = {}
        if 'xy' in plane_type:
            x['xy'] = F.relu(self.conv1(x['xy']))
            x_up['xy'] = x['xy']
        if 'xz' in plane_type:
            x['xz'] = F.relu(self.conv1(x['xz']))
            x_up['xz'] = x['xz']
        if 'yz' in plane_type:
            x['yz'] = F.relu(self.conv1(x['yz']))
            x_up['yz'] = x['yz']
                
        channel = x['xy'].shape[1]
        reso_plane = x['xy'].shape[2]

        # 如果是倒数第二层，直接返回当前特征
        if i == depth - 2:
            return x, c_last

        c = c_last

        c = c.reshape(b*n, -1)
        k = self.fc_k(c)
        v = self.fc_v(c)
        c = c.reshape(b, n, -1)
        k = k.reshape(b, n, -1)
        v = v.reshape(b, n, -1)
        q = {}
        q['xy'] = self.conv_q(x['xy'])
        q['xz'] = self.conv_q(x['xz'])
        q['yz'] = self.conv_q(x['yz'])
        
        if 'xy' in plane_type:
            x['xy'] = self.generate_plane_features(p, v, k, q['xy'], channel, reso_plane, plane='xy')
            x['xy'] = self.weight_conv(x['xy'])
            x['xy'] = x['xy'] + self.conv1x1(x_up['xy'])
            x['xy'] = self.conv_final(x['xy'])
            
        if 'xz' in plane_type:
            x['xz'] = self.generate_plane_features(p, v, k, q['xz'], channel, reso_plane, plane='xz')
            x['xz'] = self.weight_conv(x['xz'])
            x['xz'] = x['xz'] + self.conv1x1(x_up['xz'])
            x['xz'] = self.conv_final(x['xz'])
            
        if 'yz' in plane_type:
            x['yz'] = self.generate_plane_features(p, v, k, q['yz'], channel, reso_plane, plane='yz')
            x['yz'] = self.weight_conv(x['yz'])
            x['yz'] = x['yz'] + self.conv1x1(x_up['yz'])
            x['yz'] = self.conv_final(x['yz'])
            
        c = v
        if 'xy' in plane_type:
            c += self.sample_plane_feature(p, x['xy'], 'xy').transpose(1, 2)
        if 'xz' in plane_type:
            c += self.sample_plane_feature(p, x['xz'], 'xz').transpose(1, 2)
        if 'yz' in plane_type:
            c += self.sample_plane_feature(p, x['yz'], 'yz').transpose(1, 2)
        
        return x, c

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


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, reso,
                 merge_mode='concat', up_mode='transpose'):

        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.share_channels = 8
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.reso = reso
        
        # 根据分辨率，选择上采样方式。如果是最后一级，则不进行上采样，只进行 1x1 卷积
        if reso != 64:
            self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        else:
            self.upconv_noup = conv1x1(self.in_channels, self.out_channels)

        self.in_channels = in_channels

        self.fc_k = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )

        self.fc_v = nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels,2*out_channels),
                        nn.ReLU(),
                        nn.Linear(2*out_channels, out_channels)
                    )
        
        self.conv_q = conv1x12(out_channels, out_channels)
        self.fc_pos = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(inplace=True), nn.Linear(2, out_channels))
        self.fc_w = nn.Sequential(nn.Linear(out_channels, out_channels // self.share_channels),
                            nn.BatchNorm1d(out_channels // self.share_channels), nn.ReLU(inplace=True))
        
        # 如果采用 concat 的融合模式，定义卷积层来减少通道数
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else: # 否则，直接使用两个 3x3 卷积层
            self.conv1 = nn.Sequential(
                conv3x3(self.in_channels, self.out_channels),
                nn.ReLU(inplace=True),
                conv3x3(self.out_channels, self.out_channels),
            )
        self.weight_conv = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels
            ),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels
            )
        )
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 最终的 1x1 卷积层，用于特征的最终处理
        self.conv_final = conv1x1(out_channels, out_channels)
        self.padding = 0.1
        self.c_dim = 32 
        self.sample_mode = 'bilinear'
        
    
    def generate_plane_features(self, p, c, k, q, channel, reso_plane, plane='xy'):
        
        b, n, _ = p.shape

        p_nor = normalize_coordinate(p.clone(), padding=self.padding, plane=plane) # normalize to the range of (0, 1)
        index = coordinate2index(p_nor, reso_plane)
        
        pos_enc = p_nor - (p_nor * reso_plane).floor() / reso_plane
        for i, layer in enumerate(self.fc_pos):
            pos_enc = layer(pos_enc.transpose(1, 2).contiguous()).transpose(1, 2) if i == 1 else layer(pos_enc)
        
        batch_index = torch.arange(p.size(0), device=p.device).repeat_interleave(p.size(1))
        
        w = k - q.reshape(-1, channel, reso_plane**2)[batch_index, :, index.reshape(-1)].reshape(k.shape[0], k.shape[1], channel) + pos_enc
        w = w.reshape(k.shape[0]*k.shape[1], -1)
        w = self.fc_w(w)
        w = w.reshape(k.shape[0], k.shape[1], -1)
        
        weights_soft = scatter_softmax(w.transpose(1, 2), index).transpose(1, 2).reshape(b*n, -1)
        
        value = (c + pos_enc).reshape(b*n, -1)
        value_grid = (value.reshape(b*n, self.share_channels, self.out_channels // self.share_channels) * weights_soft.unsqueeze(-2)).view(b*n, -1)
        value_grid = value_grid.reshape(b, n, -1).transpose(1, 2)
        
        
        fea_plane = c.new_zeros(p.size(0), channel, reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T

        fea_plane = scatter_add(value_grid, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), channel, reso_plane, reso_plane) # sparce matrix (B x 512 x reso x reso)

        return fea_plane


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c
    
    def forward(self, p, from_down, from_up, plane_type, c_last, i):
        
        b, n, _ = c_last.shape
        
        # 根据当前深度选择是否执行上采样或不进行上采样
        if i == 2:
            from_up['xy'] = self.upconv_noup(from_up['xy'])
            from_up['xz'] = self.upconv_noup(from_up['xz'])
            from_up['yz'] = self.upconv_noup(from_up['yz'])
        else:
            from_up['xy'] = self.upconv(from_up['xy'])
            from_up['xz'] = self.upconv(from_up['xz'])
            from_up['yz'] = self.upconv(from_up['yz'])

        # 融合上采样的特征和下采样的特征
        x = {}
        if self.merge_mode == 'concat':
            x['xy'] = torch.cat((from_up['xy'], from_down['xy']), 1)
            x['xz'] = torch.cat((from_up['xz'], from_down['xz']), 1)
            x['yz'] = torch.cat((from_up['yz'], from_down['yz']), 1)
        else:
            x['xy'] = from_up['xy'] + from_down['xy']
            x['xz'] = from_up['xz'] + from_down['xz']
            x['yz'] = from_up['yz'] + from_down['yz']
        
        x_up = {}
        if 'xy' in plane_type:
            x['xy'] = F.relu(self.conv1(x['xy']))
            x_up['xy'] = x['xy']
        if 'xz' in plane_type:
            x['xz'] = F.relu(self.conv1(x['xz']))
            x_up['xz'] = x['xz']
        if 'yz' in plane_type:
            x['yz'] = F.relu(self.conv1(x['yz']))
            x_up['yz'] = x['yz']
                
        channel = x['xy'].shape[1]
        reso_plane = x['xy'].shape[2]

        # 如果是倒数第二层，直接返回当前特征
        x_temp = x
        # if i == depth - 2:
        #     return x, c_last

        c = c_last

        c = c.reshape(b*n, -1)
        k = self.fc_k(c)
        v = self.fc_v(c)
        c = c.reshape(b, n, -1)
        k = k.reshape(b, n, -1)
        v = v.reshape(b, n, -1)
        q = {}
        q['xy'] = self.conv_q(x['xy'])
        q['xz'] = self.conv_q(x['xz'])
        q['yz'] = self.conv_q(x['yz'])
        
        if 'xy' in plane_type:
            x['xy'] = self.generate_plane_features(p, v, k, q['xy'], channel, reso_plane, plane='xy')
            x['xy'] = self.weight_conv(x['xy'])
            x['xy'] = x['xy'] + self.conv1x1(x_up['xy'])
            x['xy'] = self.conv_final(x['xy'])
            
        if 'xz' in plane_type:
            x['xz'] = self.generate_plane_features(p, v, k, q['xz'], channel, reso_plane, plane='xz')
            x['xz'] = self.weight_conv(x['xz'])
            x['xz'] = x['xz'] + self.conv1x1(x_up['xz'])
            x['xz'] = self.conv_final(x['xz'])
            
        if 'yz' in plane_type:
            x['yz'] = self.generate_plane_features(p, v, k, q['yz'], channel, reso_plane, plane='yz')
            x['yz'] = self.weight_conv(x['yz'])
            x['yz'] = x['yz'] + self.conv1x1(x_up['yz'])
            x['yz'] = self.conv_final(x['yz'])
            
        c = v
        if 'xy' in plane_type:
            c += self.sample_plane_feature(p, x['xy'], 'xy').transpose(1, 2)
        if 'xz' in plane_type:
            c += self.sample_plane_feature(p, x['xz'], 'xz').transpose(1, 2)
        if 'yz' in plane_type:
            c += self.sample_plane_feature(p, x['yz'], 'yz').transpose(1, 2)

        if i == depth - 2:
            return x_temp,c
        
        return x, c


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat', **kwargs):
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        resos = [64, 64, 32, 16]
        
        for i in range(depth): # 0 1 2 3 
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            if i == 0 or i == 1:
                pooling = False
            else:
                pooling = True
            down_conv = DownConv(ins, outs, resos[i], pooling)
            self.down_convs.append(down_conv)

        for i in range(depth-1): # 0 1 2 
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, resos[-i-1], up_mode=up_mode,
                merge_mode=merge_mode)
            # up_conv = EUCB(ins, outs, resos[-i-1],merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m,EUCB):
            m.weight_init()

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, p, x, plane_type, c):
        #x:平面特征
        encoder_outs = []
        x_out = []
        for i, module in enumerate(self.down_convs):
            x, before_pool, c = module(p, x, plane_type,  c)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x, c = module(p, before_pool, x, plane_type, c, i)
            x_ = {}
            if 'xy' in plane_type:
                x_['xy'] = x['xy']
            if 'xz' in plane_type:
                x_['xz'] = x['xz']
            if 'yz' in plane_type:
                x_['yz'] = x['yz']
            x_out.append(x_)
                
        return x_out,p,c
