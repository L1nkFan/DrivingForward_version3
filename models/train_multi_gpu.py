"""
多GPU分布式训练脚本
基于 PyTorch DDP (DistributedDataParallel) 实现

使用方法:
1. 单机多卡: python -m torch.distributed.launch --nproc_per_node=N train_multi_gpu.py --config_file=configs/nuscenes/main_ddp.yaml
2. 使用 torchrun: torchrun --standalone --nproc_per_node=N train_multi_gpu.py --config_file=configs/nuscenes/main_ddp.yaml

环境变量:
- CUDA_VISIBLE_DEVICES: 指定使用的GPU (例如: 0,1,2,3)
- MASTER_ADDR: 主节点地址 (默认: localhost)
- MASTER_PORT: 主节点端口 (默认: 29500)
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 设置确定性行为
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)

import utils
from models import DrivingForwardModelDDP
from trainer import DrivingForwardTrainerDDP


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multi-GPU Distributed Training Script')
    parser.add_argument('--config_file', default='./configs/nuscenes/main_ddp.yaml', 
                        type=str, help='config yaml file for multi-GPU training')
    parser.add_argument('--novel_view_mode', default='MF', type=str, 
                        help='novel view mode: MF (Multi-Frame) or SF (Single-Frame)')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training (automatically set by torchrun)')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of processes for distributed training (overrides config if > 0)')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend: nccl (recommended for GPU), gloo')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    args = parser.parse_args()
    return args


def setup_distributed(rank, world_size, backend='nccl', dist_url='env://'):
    """
    初始化分布式训练环境
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数 (GPU数)
        backend: 分布式后端 (nccl推荐用于GPU)
        dist_url: 分布式初始化URL
    """
    # 设置环境变量
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    # 初始化进程组
    if dist_url == 'env://':
        # 使用环境变量初始化 (torchrun会自动设置)
        dist.init_process_group(backend=backend)
    else:
        dist.init_process_group(backend=backend, init_method=dist_url,
                               world_size=world_size, rank=rank)
    
    # 设置当前设备
    torch.cuda.set_device(rank)
    
    # 同步所有进程
    dist.barrier()
    
    if rank == 0:
        print(f"Distributed training initialized:")
        print(f"  - Backend: {backend}")
        print(f"  - World Size: {dist.get_world_size()}")
        print(f"  - Rank: {rank}")
        print(f"  - Local Rank: {rank}")
        print(f"  - Device: cuda:{rank}")


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_model_for_ddp(model, rank):
    """
    将模型包装为DistributedDataParallel
    
    Args:
        model: 原始模型
        rank: 当前进程的rank
    
    Returns:
        DDP包装后的模型
    """
    # 将模型移动到当前GPU
    for model_name, m in model.models.items():
        m.cuda(rank)
    
    # 使用DDP包装每个子模型
    for model_name in model.models.keys():
        model.models[model_name] = DDP(
            model.models[model_name],
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,  # 如果某些前向传播中不是所有参数都使用，需要设为True
            broadcast_buffers=True
        )
    
    # 设置DDP标志
    model.ddp_enable = True
    model.world_size = dist.get_world_size()
    model.rank = rank
    
    return model


def adjust_hyperparameters_for_ddp(cfg, world_size):
    """
    根据GPU数量调整超参数
    
    多GPU训练时的一些最佳实践:
    1. 学习率: 通常与batch size成正比，线性缩放
       lr_ddp = lr_single * world_size (如果batch size也相应增加)
    2. Batch size: 每个GPU的batch size可以保持不变，总batch size = per_gpu_batch * world_size
    3. 学习率调度: epoch数可能需要调整，因为每个epoch看到的样本数增加了
    
    Args:
        cfg: 配置字典
        world_size: GPU数量
    """
    # 原始配置
    original_batch_size = cfg['training']['batch_size']
    original_lr = cfg['training']['learning_rate']
    original_num_epochs = cfg['training']['num_epochs']
    
    # 线性缩放学习率 (如果batch size相应增加)
    # 注意: 这里假设总batch size = per_gpu_batch * world_size
    # 如果保持per_gpu_batch不变，则不需要调整学习率
    scale_lr = cfg.get('ddp', {}).get('scale_lr', False)
    if scale_lr:
        cfg['training']['learning_rate'] = original_lr * world_size
        if current_rank == 0:
            print(f"  - Learning rate scaled: {original_lr} -> {cfg['training']['learning_rate']}")
    
    # 可选: 调整scheduler step size
    # 如果使用更大的effective batch size，可能需要更少的steps
    # cfg['training']['scheduler_step_size'] = max(1, cfg['training']['scheduler_step_size'] // world_size)
    
    # 更新DDP配置
    cfg['ddp']['world_size'] = world_size
    cfg['ddp']['gpus'] = list(range(world_size))
    cfg['ddp']['ddp_enable'] = True
    
    return cfg


def create_distributed_dataloader(dataset, batch_size, num_workers, shuffle=True, rank=0, world_size=1):
    """
    创建支持分布式训练的数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 每个GPU的batch size
        num_workers: 数据加载的工作进程数
        shuffle: 是否打乱数据
        rank: 当前进程的rank
        world_size: 总进程数
    
    Returns:
        DataLoader with DistributedSampler
    """
    # 创建DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=0
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    return dataloader, sampler


def train_worker(rank, world_size, args, cfg):
    """
    每个GPU上运行的训练工作进程
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        args: 命令行参数
        cfg: 配置字典
    """
    global current_rank
    current_rank = rank
    
    # 初始化分布式环境
    setup_distributed(rank, world_size, args.dist_backend, args.dist_url)
    
    # 调整超参数
    cfg = adjust_hyperparameters_for_ddp(cfg, world_size)
    
    # 创建DDP模型 (内部已包含DDP设置)
    model = DrivingForwardModelDDP(cfg, rank)
    
    # 创建DDP训练器
    trainer = DrivingForwardTrainerDDP(cfg, rank)
    trainer.ddp_enable = True
    trainer.world_size = world_size
    
    # 开始训练
    try:
        trainer.learn(model)
    except Exception as e:
        print(f"Rank {rank}: Training failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    cfg = utils.get_config(args.config_file, mode='train', novel_view_mode=args.novel_view_mode)
    
    # 确定world_size
    if args.world_size > 0:
        world_size = args.world_size
    elif 'ddp' in cfg and 'world_size' in cfg['ddp']:
        world_size = cfg['ddp']['world_size']
    else:
        # 自动检测可用GPU数量
        world_size = torch.cuda.device_count()
    
    # 检查GPU可用性
    if world_size == 0:
        raise RuntimeError("No CUDA devices available for training")
    
    print(f"Starting distributed training with {world_size} GPUs")
    print(f"Config file: {args.config_file}")
    print(f"Novel view mode: {args.novel_view_mode}")
    
    # 检查是否使用torchrun或torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 使用torchrun启动，环境变量已设置
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        print(f"Detected torchrun environment:")
        print(f"  - RANK: {rank}")
        print(f"  - LOCAL_RANK: {local_rank}")
        print(f"  - WORLD_SIZE: {world_size}")
        
        train_worker(local_rank, world_size, args, cfg)
    else:
        # 使用mp.spawn启动多进程
        print(f"Using mp.spawn to start {world_size} processes")
        mp.spawn(
            train_worker,
            args=(world_size, args, cfg),
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    main()
