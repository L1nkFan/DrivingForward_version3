"""
第二阶段模型工厂
根据配置自动选择合适的模型和损失函数
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2_trainer import Stage2Model, Stage2ModelMultiMode
from stage2_modules import Stage2Loss, Stage2LossMultiMode


def build_stage2_model(cfg, rank=0):
    """
    根据配置构建第二阶段模型
    
    Args:
        cfg: 配置字典
        rank: GPU rank
        
    Returns:
        model: 第二阶段模型实例
    """
    # 检查是否启用了多模式
    model_cfg = cfg.get('model', {})
    temporal = model_cfg.get('temporal', False)
    spatio = model_cfg.get('spatio', False)
    spatio_temporal = model_cfg.get('spatio_temporal', False)
    
    # 如果启用了任意一种模式，使用多模式版本
    if temporal or spatio or spatio_temporal:
        print(f"[Rank {rank}] Using Stage2ModelMultiMode")
        print(f"[Rank {rank}] Training modes - Temporal: {temporal}, Spatio: {spatio}, Spatio-Temporal: {spatio_temporal}")
        return Stage2ModelMultiMode(cfg, rank)
    else:
        print(f"[Rank {rank}] Using Stage2Model (default spatio-temporal mode)")
        return Stage2Model(cfg, rank)


def build_stage2_criterion(cfg, rank=0):
    """
    根据配置构建第二阶段损失函数
    
    Args:
        cfg: 配置字典
        rank: GPU rank
        
    Returns:
        criterion: 第二阶段损失函数实例
    """
    # 获取损失权重
    loss_cfg = cfg.get('loss', {})
    lambda_warp = loss_cfg.get('lambda_warp', 0.02)
    lambda_consist = loss_cfg.get('lambda_consist', 1e-5)
    lambda_render = loss_cfg.get('lambda_render', 0.01)
    enable_spatial_consistency = loss_cfg.get('enable_spatial_consistency', False)
    enable_render_lpips = loss_cfg.get('enable_render_lpips', False)
    
    # 检查是否启用了多模式
    model_cfg = cfg.get('model', {})
    temporal = model_cfg.get('temporal', False)
    spatio = model_cfg.get('spatio', False)
    spatio_temporal = model_cfg.get('spatio_temporal', False)
    
    # 如果启用了任意一种模式，使用多模式版本
    if temporal or spatio or spatio_temporal:
        print(f"[Rank {rank}] Using Stage2LossMultiMode")
        return Stage2LossMultiMode(
            lambda_warp=lambda_warp,
            lambda_consist=lambda_consist,
            lambda_render=lambda_render,
            enable_spatial_consistency=enable_spatial_consistency,
            enable_render_lpips=enable_render_lpips,
            rank=rank
        )
    else:
        print(f"[Rank {rank}] Using Stage2Loss (default)")
        return Stage2Loss(
            lambda_warp=lambda_warp,
            lambda_consist=lambda_consist,
            lambda_render=lambda_render,
            rank=rank
        )


def get_training_mode(cfg):
    """
    获取当前训练模式
    
    Args:
        cfg: 配置字典
        
    Returns:
        mode: 训练模式字符串 ('temporal', 'spatio', 'spatio_temporal', 或 'unknown')
    """
    model_cfg = cfg.get('model', {})
    
    temporal = model_cfg.get('temporal', False)
    spatio = model_cfg.get('spatio', False)
    spatio_temporal = model_cfg.get('spatio_temporal', False)
    
    if temporal and not (spatio or spatio_temporal):
        return 'temporal'
    elif spatio and not (temporal or spatio_temporal):
        return 'spatio'
    elif spatio_temporal or (temporal and spatio):
        return 'spatio_temporal'
    else:
        return 'spatio_temporal'  # 默认


def print_training_mode_info(cfg, rank=0):
    """
    打印训练模式信息
    
    Args:
        cfg: 配置字典
        rank: GPU rank
    """
    mode = get_training_mode(cfg)
    
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Stage 2 Training Mode: {mode.upper()}")
    print(f"{'='*60}")
    
    if mode == 'temporal':
        print("  - Only temporal dimension learning")
        print("  - Computes optical flow between consecutive frames")
        print("  - Uses single camera per iteration")
        print("  - Loss: L_warp + L_consist + L_render")
    elif mode == 'spatio':
        print("  - Only spatial dimension learning")
        print("  - Computes optical flow between adjacent cameras")
        print("  - Uses multi-camera setup at current time step")
        print("  - Loss: L_warp_spatial + L_consist_spatial")
    elif mode == 'spatio_temporal':
        print("  - Joint spatio-temporal learning")
        print("  - Computes both temporal and spatial optical flows")
        print("  - Uses multi-camera + multi-frame setup")
        print("  - Loss: L_warp_temporal + L_warp_spatial + L_consist + L_render")
    
    print(f"{'='*60}\n")


# 便捷函数
def is_temporal_mode(cfg):
    """检查是否为Temporal模式"""
    model_cfg = cfg.get('model', {})
    return model_cfg.get('temporal', False) and not model_cfg.get('spatio', False)


def is_spatio_mode(cfg):
    """检查是否为Spatio模式"""
    model_cfg = cfg.get('model', {})
    return model_cfg.get('spatio', False) and not model_cfg.get('temporal', False)


def is_spatio_temporal_mode(cfg):
    """检查是否为Spatio-Temporal模式"""
    model_cfg = cfg.get('model', {})
    return model_cfg.get('spatio_temporal', False) or (
        model_cfg.get('temporal', False) and model_cfg.get('spatio', False)
    )
