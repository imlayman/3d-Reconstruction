import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from src.common import normalize_coordinate, normalize_3d_coordinate,map2local
from torch import distributions as dist
from src.attention import *

from einops import rearrange, repeat
from pytorch3d.ops import knn_gather, knn_points
from xformers.ops.fmha import memory_efficient_attention,AttentionBias
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from src.modules import ResidualMLP, activation
from src.rotary import PointRotaryEmbedding
from typing import Optional


class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128, num_iterations=2,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.num_iterations = num_iterations
        
        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])
        
        self.fc_multi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
            ),
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            ),
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
    

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_planes, logits):
        c = 0
        for i in range(3): #3是因为encoder的输出是3个尺度的3平面特征
            c_plane = c_planes[i]
            c_temp = 0
            if self.c_dim != 0:
                plane_type = list(c_plane.keys())
                if 'grid' in plane_type:
                    c_temp += self.sample_grid_feature(p, c_plane['grid'])
                if 'xz' in plane_type:
                    c_temp += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
                if 'xy' in plane_type:
                    c_temp += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
                if 'yz' in plane_type:
                    c_temp += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
                c_temp = c_temp.transpose(1, 2)
                # 将每个平面采样到的特征通过 fc_multi 进行融合并累加到 c 中
                c += self.fc_multi[i](c_temp)

        p = p.float()
        # 使用 fc_p 将输入点坐标映射到隐藏特征空间，生成隐藏特征
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                # 将累加的特征 c 通过 fc_c[i] 映射并与当前特征 net 相加
                net = net + self.fc_c[i](c)
            # 结果传递给 ResnetBlockFC 进行进一步处理
            net = self.blocks[i](net)
        # fc_out 线性层输出一个标量值 out，表示输入点的占用概率
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        if logits:
            out = dist.Bernoulli(logits=out)
        return out

class PatchLocalDecoder(nn.Module):
    
    ''' Decoder adapted for crop training.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        local_coord (bool): whether to use local coordinate
        unit_size (float): defined voxel unit size for local system
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='bilinear', local_coord=False, pos_encoding='linear', unit_size=0.1, padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        #self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None

        if pos_encoding == 'sin_cos':
            self.fc_p = nn.Linear(60, hidden_size)
        else:
            self.fc_p = nn.Linear(dim, hidden_size)
    
    def sample_feature(self, xy, c, fea_type='2d'):
        if fea_type == '2d':
            xy = xy[:, :, None].float()
            vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        else:
            xy = xy[:, :, None, None].float()
            vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        p_n = p['p_n']
        p = p['p']

        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_feature(p_n['grid'], c_plane['grid'], fea_type='3d')
            if 'xz' in plane_type:
                c += self.sample_feature(p_n['xz'], c_plane['xz'])
            if 'xy' in plane_type:
                c += self.sample_feature(p_n['xy'], c_plane['xy'])
            if 'yz' in plane_type:
                c += self.sample_feature(p_n['yz'], c_plane['yz'])
            c = c.transpose(1, 2)

        p = p.float()
        if self.map2local:
            p = self.map2local(p)
        
        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out
    
class DecoderTransformer(nn.Module):
    def __init__(self, dim, num_heads, act_fn="relu"):
        super().__init__()
        self.num_heads = num_heads

        self.pe = PointRotaryEmbedding(dim)
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            activation(act_fn),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, xyz_seq, x_seq):
        """
        - xyz_seq: b m k+1 3
        - x_seq: b m k+1 c
        """
        b, n_seq, seq_len = x_seq.shape[:3]
        shortcut = x_seq

        q = self.to_q(self.norm(x_seq))
        k = self.to_k(x_seq)
        v = self.to_v(x_seq)

        q, k = self.pe(xyz_seq, q, k)  # b m k+1 c
        q, k, v = [
            rearrange(_, "b m k (h c) -> 1 (b m k) h c", h=self.num_heads)
            .contiguous()
            for _ in (q, k, v)]
        attn_bias = BlockDiagonalMask.from_seqlens([seq_len] * b * n_seq, [seq_len] * b * n_seq)
        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        out = rearrange(out, "1 (b m k) h c -> b m k (h c)", b=b, m=n_seq)  # b m k+1 c
        out = shortcut[..., 0, :] + self.to_out(out[..., 0, :])  # b m c
        out = out + self.ffn(out)
        return out


class ULTODecoder2d(nn.Module):
    def __init__(
        self,
        dim,
        c_dim,
        dim_out=1,
        n_blocks=5,
        padding=0.1,
        act_fn="relu",
        num_neighbors=32,
        head_dim=32,
        hidden_size = 256,
        sample_mode='bilinear',
    ):
        super().__init__()
        self.padding = padding
        self.num_neighbors = num_neighbors
        num_heads = (c_dim * 2) // head_dim
        self.sample_mode = sample_mode

        self.to_c_query = nn.Linear(c_dim, c_dim * 2, bias=False)
        self.transformers = nn.ModuleList(
            [DecoderTransformer(c_dim * 2, num_heads=num_heads, act_fn=act_fn) for _ in range(n_blocks)]
        )
        self.fc_out = nn.Linear(c_dim * 2, dim_out, bias=False)

        self.fc_multi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
            ),
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            ),
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        ])

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def forward(self, c_planes, query, p, c_point,logits):
        """
        - c_planes: triplane特征 [3 x (batch, dim, res, res)]
        - query: 查询点坐标 [batch, num_queries, 3]
        - p: 输入点云坐标 [batch, num_points, 3]
        - c_point: point特征 [batch, num_points, dim]
        """
        # 1. 获取查询点的grid特征
        c_query = 0
        for i in range(3): #3是因为encoder的输出是3个尺度的3平面特征
            c_plane = c_planes[i]
            c_temp = 0
            plane_type = list(c_plane.keys())
            if 'grid' in plane_type:
                c_temp += self.sample_grid_feature(query, c_plane['grid'])
            if 'xz' in plane_type:
                c_temp += self.sample_plane_feature(query, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c_temp += self.sample_plane_feature(query, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c_temp += self.sample_plane_feature(query, c_plane['yz'], plane='yz')
            c_temp = c_temp.transpose(1, 2)
            # 将每个平面采样到的特征通过 fc_multi 进行融合并累加到 c 中
            c_query += self.fc_multi[i](c_temp)
        c_query = self.to_c_query(c_query)  # b m 2c

        # 2. 获取输入点的grid特征并与point特征concat
        c2 = 0
        for i in range(3): #3是因为encoder的输出是3个尺度的3平面特征
            c_plane = c_planes[i]
            c_temp = 0
            plane_type = list(c_plane.keys())
            if 'grid' in plane_type:
                c_temp += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c_temp += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c_temp += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c_temp += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c_temp = c_temp.transpose(1, 2)
            # 将每个平面采样到的特征通过 fc_multi 进行融合并累加到 c 中
            c2 += self.fc_multi[i](c_temp)
        c_point = torch.cat([c_point, c2], -1)  # b n 2c

        # print(c_point.shape)

        # 3. 构建KNN邻域
        knn = knn_points(query, p, K=self.num_neighbors, return_nn=True, return_sorted=False)
        c_knn = knn_gather(c_point, knn.idx)

        # 4. 构建特征序列
        xyz_seq = torch.cat([query.unsqueeze(2), knn.knn], 2)  # b m k+1 3

         # 5. 通过Transformer块处理
        for transformer in self.transformers:
            c_seq = torch.cat([c_query.unsqueeze(2), c_knn], 2)  # b m k+1 2c
            c_query = transformer(xyz_seq, c_seq)  # b m 2c

        # 6. 预测occupancy
        out = self.fc_out(c_query)  # b m 1
        out = out.squeeze(-1)
        if logits:
            out = dist.Bernoulli(logits=out)
        return out
    
if __name__ == "__main__":
    model = ULTODecoder2d(
        dim=3,
        c_dim=32,
        dim_out=1,
        n_blocks=5,
        padding=0.1,
        act_fn="relu",
        num_neighbors=32,
        head_dim=32,
        hidden_size=32,
    ).cuda()
    c_dim = [128,64,32]
    batch_size = 2
    c_planes = [{} for _ in range(3)]
    for i in range(3):
        c_planes[i]['xy'] = torch.rand(batch_size, c_dim[i], 64, 64,device='cuda') # xy平面特征
        c_planes[i]['xz'] = torch.rand(batch_size, c_dim[i], 64, 64,device='cuda') # xz平面特征
        c_planes[i]['yz'] = torch.rand(batch_size, c_dim[i], 64, 64,device='cuda') # yz平面特征
    # c_planes = [torch.rand(2, 32, 64, 64, device="cuda") for _ in range(3)]
    query = torch.rand(batch_size, 2048, 3, device="cuda")
    p = torch.rand(batch_size, 3000, 3, device="cuda")
    c_point = torch.rand(batch_size, 3000, 32, device="cuda")
    out = model(c_planes, query, p, c_point,True)
    print(out.shape)