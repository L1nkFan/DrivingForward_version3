"""
支持分布式训练 (DDP) 的 Trainer
继承自基础Trainer，添加DDP特定功能
"""

import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch import Tensor

from utils import Logger
from .trainer import DrivingForwardTrainer

from lpips import LPIPS
from jaxtyping import Float, UInt8
from skimage.metrics import structural_similarity
from einops import reduce

from PIL import Image
from pathlib import Path
from einops import rearrange, repeat
from typing import Union
import numpy as np


class DrivingForwardTrainerDDP(DrivingForwardTrainer):
    """
    支持分布式数据并行 (DDP) 的 Trainer
    
    主要修改:
    1. 每个epoch开始时更新DistributedSampler
    2. 添加进程同步点
    3. 只在rank 0进行验证和日志记录
    """
    
    def __init__(self, cfg, rank, use_tb=True):
        super().__init__(cfg, rank, use_tb)
        self.ddp_enable = True
    
    def learn(self, model):
        """
        设置训练过程 (DDP版本)
        
        主要修改:
        - 每个epoch开始时更新sampler的epoch
        - 添加更多的同步点
        """
        train_dataloader = model.train_dataloader()
        
        # 获取sampler以便更新epoch
        train_sampler = model.get_train_sampler() if hasattr(model, 'get_train_sampler') else None
        
        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)
        
        self.step = 0
        start_time = time.time()
        
        for self.epoch in range(self.num_epochs):
            # 更新sampler的epoch (重要: 确保每个epoch数据打乱)
            if train_sampler is not None:
                train_sampler.set_epoch(self.epoch)
            
            # 同步所有进程
            if self.ddp_enable:
                dist.barrier()
            
            # 训练一个epoch
            self.train(model, train_dataloader, start_time)
            
            # 同步所有进程
            if self.ddp_enable:
                dist.barrier()
            
            # 只在rank 0保存模型
            if self.rank == 0:
                model.save_model(self.epoch)
                print('-' * 110)
            
            # 同步所有进程
            if self.ddp_enable:
                dist.barrier()
                
        if self.rank == 0:
            self.logger.close_tb()
    
    def train(self, model, data_loader, start_time):
        """
        训练模型 (DDP版本)
        
        主要修改:
        - 只在rank 0显示进度条
        - 添加梯度同步
        """
        model.set_train()
        
        # 只在rank 0显示进度条
        if self.rank == 0:
            pbar = tqdm(total=len(data_loader), 
                       desc=f'Epoch {self.epoch}/{self.num_epochs}', 
                       mininterval=1)
        else:
            pbar = None
        
        for batch_idx, inputs in enumerate(data_loader):
            before_op_time = time.time()
            
            # 清零梯度
            model.optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            outputs, losses = model.process_batch(inputs, self.rank)
            
            # 反向传播
            losses['total_loss'].backward()
            
            # 梯度裁剪 (可选，防止梯度爆炸)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器步进
            model.optimizer.step()
            
            # 只在rank 0记录日志
            if self.rank == 0:
                self.logger.update(
                    'train',
                    self.epoch,
                    self.world_size,
                    batch_idx,
                    self.step,
                    start_time,
                    before_op_time,
                    inputs,
                    outputs,
                    losses
                )
                
                if self.logger.is_checkpoint(self.step):
                    self.validate(model)
                
                pbar.update(1)
            
            self.step += 1
        
        if pbar is not None:
            pbar.close()
        
        # 学习率调度器步进
        model.lr_scheduler.step()
        
        # 同步所有进程
        if self.ddp_enable:
            dist.barrier()
