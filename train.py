import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
import torch.nn as nn ###


# 解析命令行参数
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.') # 添加配置文件路径参数
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.') # 选项：禁用 CUDA
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.') # 设置超时退出

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml') # 加载配置文件
is_cuda = (torch.cuda.is_available() and not args.no_cuda) # 检查 CUDA 是否可用且未禁用
device = torch.device("cuda" if is_cuda else "cpu")  # 设置设备为 CUDA 或 CPU
# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir'] # 输出目录
batch_size = cfg['training']['batch_size'] # 批次大小
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
exit_after = args.exit_after

# 决定模型选择的度量标准及其优化方向
model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1 # 最大化度量
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1 # 最小化度量
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# 创建输出目录（如不存在）
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml')) # 将配置文件复制到输出目录

# 数据集初始化
train_dataset = config.get_dataset('train', cfg, return_idx=True) # 加载训练数据集
train_boundary_dataset = config.get_dataset('train_boundary', cfg, return_idx=True) # 加载边界训练数据集
val_dataset = config.get_dataset('val', cfg, return_idx=True) # 加载验证数据集

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

train_boundary_loader = torch.utils.data.DataLoader(
    train_boundary_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
model_counter = defaultdict(int) # 模型计数器
data_vis_list = [] # 存储可视化数据的列表


# 模型初始化
model = config.get_model(cfg, device=device, dataset=train_dataset)
### use 2 gpu
model.to(device)
print(model)
print('output path: ', cfg['training']['out_dir'])

# 初始化生成器
generator = config.get_generator(model, cfg, device=device)

# 设置学习率
learning_rate =  1e-4 # 5e-5
# Intialize training
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 使用 Adam 优化器
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9) 备选：使用 SGD 优化器
# 初始化学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500, eta_min=learning_rate * 0.0001)
# 初始化训练器
trainer = config.get_trainer(model, optimizer, scheduler, cfg, device=device)

# 初始化检查点管理
checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer, scheduler=scheduler)
try:
    load_dict = checkpoint_io.load('model.pt') # 尝试加载现有检查点
except FileExistsError:
    load_dict = dict() # 如果没有检查点，从头开始
epoch_it = load_dict.get('epoch_it', 0)  # 当前训练轮次
it = load_dict.get('it', 0) # 当前迭代次数
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf) # 最佳验证指标

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs')) # 初始化 TensorBoard 日志记录

# Shorthands
print_every = cfg['training']['print_every']  # 每隔多少次迭代打印日志
checkpoint_every = cfg['training']['checkpoint_every']  # 每隔多少轮次保存检查点
validate_every = cfg['training']['validate_every']  # 每隔多少轮次进行验证
visualize_every = cfg['training']['visualize_every']  # 每隔多少轮次进行可视化
stage_1_epoch = cfg['training']['stage_1_epoch']  # 阶段1的训练轮次
stage_2_epoch = cfg['training']['stage_2_epoch']  # 阶段2的训练轮次

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)

# 阶段1训练循环
while epoch_it <= stage_1_epoch:
    epoch_it += 1
    for batch in train_loader:
        it += 1
        loss = trainer.train_step(batch, m=0) # 执行训练步骤并获取损失
        logger.add_scalar('train/loss', loss, it) # 将损失写入 TensorBoard

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                    % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))
            
    # 保存检查点
    if (checkpoint_every > 0 and (epoch_it % checkpoint_every) == 0):
        print('Saving checkpoint')
        checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                        loss_val_best=metric_val_best)

    # 备份模型
    if (backup_every > 0 and (epoch_it % backup_every) == 0):
        print('Backup checkpoint')
        checkpoint_io.save('model_%d.pt' % epoch_it, epoch_it=epoch_it, it=it,
                        loss_val_best=metric_val_best)
        
    # 验证模型
    if validate_every > 0 and (epoch_it % validate_every) == 0:
        eval_dict = trainer.evaluate(val_loader) # 验证模型
        metric_val = eval_dict[model_selection_metric] # 获取验证指标
        print('Validation metric (%s): %.4f'
            % (model_selection_metric, metric_val))

        for k, v in eval_dict.items():
            logger.add_scalar('val/%s' % k, v, it) # 将验证指标写入 TensorBoard

        if model_selection_sign * (metric_val - metric_val_best) > 0:
            metric_val_best = metric_val # 更新最佳指标
            print('New best model (loss %.4f)' % metric_val_best)
            checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                            loss_val_best=metric_val_best)


    trainer.schedule()
