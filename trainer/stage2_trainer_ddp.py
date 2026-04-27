"""
第二阶段训练器 - 分布式数据并行 (DDP) 版本
支持多GPU训练
基于修复后的单卡版本实现
"""

import os
import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger
from lpips import LPIPS
from skimage.metrics import structural_similarity
import numpy as np

from .stage2_trainer import Stage2Trainer


class Stage2TrainerDDP(Stage2Trainer):
    """
    第二阶段训练器 (DDP版本)
    继承自单GPU版本，添加分布式训练支持
    """
    
    def __init__(self, cfg, rank=0, use_tb=True):
        super().__init__(cfg, rank, use_tb)
        self.ddp_enable = True
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    def setup_distributed_model(self, model):
        """
        为DDP设置模型
        
        Args:
            model: Stage2Model实例
            
        Returns:
            model: 配置好的模型
        """
        # 将模型移动到当前GPU
        model.cuda(self.rank)
        
        # 冻结网络不需要DDP包装，因为它们不参与梯度计算
        # 只需要包装ResFlowNet
        res_flow_net = model.stage2_modules['res_flow_net']
        
        # 使用DDP包装ResFlowNet
        res_flow_net_ddp = DDP(
            res_flow_net,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False,  # 如果所有参数都参与计算，设为False以提高效率
            broadcast_buffers=True
        )
        
        # 替换原始模块
        model.stage2_modules['res_flow_net'] = res_flow_net_ddp
        
        return model
    
    def reduce_dict(self, loss_dict):
        """
        在所有GPU间聚合损失字典
        
        Args:
            loss_dict: 损失字典
            
        Returns:
            reduced_dict: 聚合后的损失字典
        """
        if not dist.is_initialized():
            return loss_dict
        
        reduced_dict = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                # 创建tensor用于收集所有GPU的值
                world_size = dist.get_world_size()
                gathered_values = [torch.zeros_like(value) for _ in range(world_size)]
                dist.all_gather(gathered_values, value)
                reduced_dict[key] = torch.stack(gathered_values).mean()
            else:
                reduced_dict[key] = value
        
        return reduced_dict
    
    def learn(self, model):
        """
        主训练循环 (DDP版本)
        
        Args:
            model: Stage2Model实例
        """
        # 设置分布式模型
        model = self.setup_distributed_model(model)
        
        # 获取数据加载器
        train_dataloader = model.train_dataloader()
        train_sampler = model.get_train_sampler() if hasattr(model, 'get_train_sampler') else None
        
        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)
        
        self.step = 0
        start_time = time.time()
        
        for self.epoch in range(self.num_epochs):
            # 更新sampler的epoch
            if train_sampler is not None:
                train_sampler.set_epoch(self.epoch)
            
            # 同步所有进程
            if self.ddp_enable:
                dist.barrier()
            
            # 训练一个epoch
            self.train_epoch(model, train_dataloader, start_time)
            
            # 同步所有进程
            if self.ddp_enable:
                dist.barrier()
            
            # 只在rank 0保存模型
            if self.rank == 0:
                # 获取原始模型 (解除DDP包装)
                original_res_flow_net = model.stage2_modules['res_flow_net'].module
                # 临时替换以便保存
                model.stage2_modules['res_flow_net'] = original_res_flow_net
                model.save_model(self.epoch)
                # 恢复DDP包装
                model.stage2_modules['res_flow_net'] = DDP(
                    original_res_flow_net,
                    device_ids=[self.rank],
                    output_device=self.rank,
                    find_unused_parameters=False,
                    broadcast_buffers=True
                )
                print('-' * 110)
            
            # 同步所有进程
            if self.ddp_enable:
                dist.barrier()
        
        if self.rank == 0:
            self.logger.close_tb()
    
    def train_epoch(self, model, data_loader, start_time):
        """
        训练一个epoch (DDP版本)
        
        Args:
            model: Stage2Model实例
            data_loader: 训练数据加载器
            start_time: 开始时间
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
            
            # 数据移动到设备
            inputs = model.to_device(inputs)
            
            # 清零梯度
            model.optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            outputs, losses = model(inputs)
            
            # 反向传播
            losses['total_loss'].backward()
            
            # 梯度裁剪 (可选)
            # torch.nn.utils.clip_grad_norm_(model.stage2_modules['res_flow_net'].parameters(), max_norm=1.0)
            
            # 优化器步进
            model.optimizer.step()
            
            # 聚合损失 (所有GPU)
            if self.ddp_enable:
                losses = self.reduce_dict(losses)
            
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
                
                # 定期验证 (使用独立的验证频率)
                val_frequency = getattr(self, 'val_frequency', 1000)
                if self.step % val_frequency == 0:
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
    
    @torch.no_grad()
    def validate(self, model):
        """
        验证模型 (DDP版本)
        
        Args:
            model: Stage2Model实例
        """
        val_dataloader = model.val_dataloader()
        val_iter = iter(val_dataloader)
        
        model.set_val()
        
        # 获取一个batch进行验证
        try:
            inputs = next(val_iter)
        except StopIteration:
            val_iter = iter(val_dataloader)
            inputs = next(val_iter)
        
        # 数据移动到设备
        inputs = model.to_device(inputs)
        
        # 前向传播
        outputs, losses = model(inputs)
        
        # 计算重建指标
        metrics = self.compute_reconstruction_metrics(inputs, outputs)
        
        # 聚合指标 (所有GPU)
        if self.ddp_enable:
            metrics_tensor = torch.tensor([
                metrics['psnr'], metrics['ssim'], metrics['lpips']
            ], device=self.rank)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
            metrics['psnr'] = metrics_tensor[0].item()
            metrics['ssim'] = metrics_tensor[1].item()
            metrics['lpips'] = metrics_tensor[2].item()
        
        # 只在rank 0打印和记录
        if self.rank == 0:
            print('\nValidation Results:')
            print(f"  PSNR: {metrics['psnr']:.4f}")
            print(f"  SSIM: {metrics['ssim']:.4f}")
            print(f"  LPIPS: {metrics['lpips']:.4f}")
            
            # 记录到tensorboard
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    self.logger.log_scalar(f'val/{key}', value.item(), self.step)
            for key, value in metrics.items():
                self.logger.log_scalar(f'val/{key}', value, self.step)
        
        # 恢复训练模式
        model.set_train()
    
    @torch.no_grad()
    def evaluate(self, model, eval_dataloader):
        """
        评估模型 (DDP版本)
        
        Args:
            model: Stage2Model实例
            eval_dataloader: 评估数据加载器
        """
        model.set_val()
        
        all_metrics = defaultdict(list)
        
        # 只在rank 0显示进度条
        if self.rank == 0:
            pbar = tqdm(total=len(eval_dataloader), desc='Evaluating')
        else:
            pbar = None
        
        for batch_idx, inputs in enumerate(eval_dataloader):
            # 数据移动到设备
            inputs = model.to_device(inputs)
            
            # 前向传播
            outputs, _ = model(inputs)
            
            # 计算指标
            metrics = self.compute_reconstruction_metrics(inputs, outputs)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    'PSNR': np.mean(all_metrics['psnr']),
                    'SSIM': np.mean(all_metrics['ssim']),
                    'LPIPS': np.mean(all_metrics['lpips'])
                })
        
        if pbar is not None:
            pbar.close()
        
        # 计算平均指标
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        # 聚合所有GPU的指标
        if self.ddp_enable:
            metrics_tensor = torch.tensor([
                avg_metrics['psnr'], avg_metrics['ssim'], avg_metrics['lpips']
            ], device=self.rank)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
            avg_metrics['psnr'] = metrics_tensor[0].item()
            avg_metrics['ssim'] = metrics_tensor[1].item()
            avg_metrics['lpips'] = metrics_tensor[2].item()
        
        # 只在rank 0打印结果
        if self.rank == 0:
            print('\nEvaluation Results:')
            for key, value in avg_metrics.items():
                print(f"  {key.upper()}: {value:.4f}")
        
        return avg_metrics


def setup_distributed(rank, world_size, backend='nccl'):
    """
    初始化分布式训练环境
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        backend: 分布式后端
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # 同步所有进程
    dist.barrier()
    
    if rank == 0:
        print(f"Distributed training initialized:")
        print(f"  - Backend: {backend}")
        print(f"  - World Size: {world_size}")
        print(f"  - Rank: {rank}")


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
