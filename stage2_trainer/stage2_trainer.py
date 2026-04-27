"""
第二阶段训练器 - 单GPU版本
"""

import os
import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger
from lpips import LPIPS
from skimage.metrics import structural_similarity
import numpy as np


class Stage2Trainer:
    """
    第二阶段训练器 (单GPU版本)
    """
    
    def __init__(self, cfg, rank=0, use_tb=True):
        self.cfg = cfg
        self.rank = rank
        self.read_config(cfg)
        
        # 初始化日志记录器
        if rank == 0:
            self.logger = Logger(cfg, use_tb)
        
        # 初始化评估指标
        self.lpips = LPIPS(net="vgg").cuda(rank)
        
    def read_config(self, cfg):
        """读取配置"""
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)
        
        # 读取验证频率配置
        validation_cfg = cfg.get('validation', {})
        self.val_frequency = validation_cfg.get('val_frequency', 1000)
    
    def learn(self, model):
        """
        主训练循环
        
        Args:
            model: Stage2Model实例
        """
        train_dataloader = model.train_dataloader()
        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)
        
        self.step = 0
        start_time = time.time()
        
        for self.epoch in range(self.num_epochs):
            # 训练一个epoch
            self.train_epoch(model, train_dataloader, start_time)
            
            # 保存模型
            if self.rank == 0:
                model.save_model(self.epoch)
                print('-' * 110)
        
        if self.rank == 0:
            self.logger.close_tb()
    
    def train_epoch(self, model, data_loader, start_time):
        """
        训练一个epoch
        
        Args:
            model: Stage2Model实例
            data_loader: 训练数据加载器
            start_time: 开始时间
        """
        model.set_train()
        criterion = getattr(model, 'criterion', None)
        if criterion is None and hasattr(model, 'model'):
            criterion = getattr(model.model, 'criterion', None)
        if criterion is not None and hasattr(criterion, 'set_training_progress'):
            criterion.set_training_progress(self.epoch, self.num_epochs, self.step, len(data_loader))
        
        # 进度条
        pbar = tqdm(total=len(data_loader), 
                   desc=f'Epoch {self.epoch}/{self.num_epochs}', 
                   mininterval=1)
        
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
            
            # 记录日志
            if self.rank == 0:
                self.logger.update(
                    'train',
                    self.epoch,
                    1,  # world_size=1 for single GPU
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
        
        pbar.close()
        
        # 学习率调度器步进
        model.lr_scheduler.step()
    
    @torch.no_grad()
    def validate(self, model):
        """
        验证模型
        
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
        
        print('\nValidation Results:')
        print(f"  PSNR: {metrics['psnr']:.4f}")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  LPIPS: {metrics['lpips']:.4f}")
        
        # 记录到tensorboard
        if self.rank == 0:
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    self.logger.log_scalar(f'val/{key}', value.item(), self.step)
            for key, value in metrics.items():
                self.logger.log_scalar(f'val/{key}', value, self.step)
        
        # 恢复训练模式
        model.set_train()
    
    def compute_reconstruction_metrics(self, inputs, outputs):
        """
        计算重建指标 (PSNR, SSIM, LPIPS)
        
        Args:
            inputs: 输入数据
            outputs: 模型输出
            
        Returns:
            metrics: 指标字典
        """
        # 获取渲染图像和GT
        rendered = outputs.get('rendered_I_t', None)
        gt = inputs[('color', 0, 0)]
        
        if rendered is None:
            return {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
        
        # 转换为numpy并计算指标
        batch_size, num_cams, C, H, W = rendered.shape
        
        psnr_list = []
        ssim_list = []
        lpips_list = []
        
        for b in range(batch_size):
            for cam in range(num_cams):
                pred_img = rendered[b, cam].detach().cpu().numpy()
                gt_img = gt[b, cam].detach().cpu().numpy()
                
                # 确保范围在[0, 1]
                pred_img = np.clip(pred_img, 0, 1)
                gt_img = np.clip(gt_img, 0, 1)
                
                # PSNR
                mse = np.mean((pred_img - gt_img) ** 2)
                psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
                psnr_list.append(psnr)
                
                # SSIM
                pred_chw = pred_img.transpose(1, 2, 0)
                gt_chw = gt_img.transpose(1, 2, 0)
                ssim = structural_similarity(
                    pred_chw, gt_chw, 
                    multichannel=True, 
                    channel_axis=2,
                    data_range=1.0
                )
                ssim_list.append(ssim)
                
                # LPIPS
                pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).cuda(self.rank)
                gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).cuda(self.rank)
                lpips_val = self.lpips(pred_tensor, gt_tensor, normalize=True).item()
                lpips_list.append(lpips_val)
        
        metrics = {
            'psnr': np.mean(psnr_list),
            'ssim': np.mean(ssim_list),
            'lpips': np.mean(lpips_list)
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, model, eval_dataloader):
        """
        评估模型
        
        Args:
            model: Stage2Model实例
            eval_dataloader: 评估数据加载器
        """
        model.set_val()
        
        all_metrics = defaultdict(list)
        
        pbar = tqdm(total=len(eval_dataloader), desc='Evaluating')
        
        for batch_idx, inputs in enumerate(eval_dataloader):
            # 数据移动到设备
            inputs = model.to_device(inputs)
            
            # 前向传播
            outputs, _ = model(inputs)
            
            # 计算指标
            metrics = self.compute_reconstruction_metrics(inputs, outputs)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            pbar.update(1)
            pbar.set_postfix({
                'PSNR': np.mean(all_metrics['psnr']),
                'SSIM': np.mean(all_metrics['ssim']),
                'LPIPS': np.mean(all_metrics['lpips'])
            })
        
        pbar.close()
        
        # 计算平均指标
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        print('\nEvaluation Results:')
        for key, value in avg_metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
        
        return avg_metrics
