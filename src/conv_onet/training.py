import os
from tqdm import trange
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    compute_iou, make_3d_grid, add_key,
)
from src.utils import visualize as vis
from src.training import BaseTrainer
import numpy as np

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    ''' 

    def __init__(self, model, optimizer, scheduler, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, m=0.4):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train() # 设置模型为训练模式
        self.optimizer.zero_grad() # 清除梯度
        loss = self.compute_loss(data, m) # 计算损失 
        loss.backward() # 反向传播
        self.optimizer.step() # 更新参数

        return loss.item()
    
    def schedule(self):
        self.scheduler.step()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval() # 进入评估模式

        device = self.device 
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        
        batch_size = points.size(0)

        kwargs = {}
        
        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            if isinstance(self.model, nn.DataParallel):
                # 推测占用概率
                p_out = self.model.module(points, inputs, sample=self.eval_sample, **kwargs)
            else:
                p_out = self.model(points, inputs, sample=self.eval_sample, **kwargs)

        # # Compute iou
        # with torch.no_grad():
        #     if isinstance(self.model, nn.DataParallel):
        #         # 推测占用概率
        #         p_out = self.model.module(points_iou, inputs, sample=self.eval_sample, **kwargs)
        #     else:
        #         p_out = self.model(points_iou, inputs, sample=self.eval_sample, **kwargs)
        
        # 根据阈值判断占用情况
        # occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_np = (occ >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

        # 计算 IOU（交并比）
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # 如果存在体素数据，则计算体素的 IOU
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def compute_loss(self, data, m):
        ''' 基于二元交叉熵损失

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device) # 采样点
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device) # 原始点云

        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        p_r = self.model(p, inputs, logits=False)
        # 通过 Bernoulli 分布 将 logits 封装成二分类问题
        p_r = dist.Bernoulli(logits=p_r)
        # 通过参数 m 对 logits 进行加权调整，主要目的是平衡占用点和非占用点的损失
        logits = (p_r.logits - m * (occ * 2 - 1))
        # logits = p_r.logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss_i.sum(-1).mean()

        return loss
