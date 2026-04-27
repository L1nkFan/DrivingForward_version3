import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

# Deterministic setup
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_config
from stage2_trainer import Stage2Model, Stage2Trainer, Stage2TrainerDDP
from stage2_trainer.stage2_trainer_ddp import setup_distributed, cleanup_distributed
from dataset import construct_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Stage2 training script (single or multi-GPU)')
    parser.add_argument(
        '--config_file',
        default='./configs/nuscenes/phase2_training.yaml',
        type=str,
        help='config yaml file for stage2 training',
    )
    parser.add_argument(
        '--novel_view_mode',
        default='MF',
        type=str,
        help='MF or SF',
    )
    parser.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='local rank for distributed training (set by torchrun)',
    )
    parser.add_argument(
        '--world_size',
        default=-1,
        type=int,
        help='number of processes for distributed training',
    )
    parser.add_argument(
        '--dist_backend',
        default='nccl',
        type=str,
        help='distributed backend',
    )
    parser.add_argument(
        '--dist_url',
        default='env://',
        type=str,
        help='url used to set up distributed training',
    )
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        help='path to res_flow_net checkpoint to resume/evaluate',
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='only run evaluation',
    )
    return parser.parse_args()


def prepare_dataloaders(cfg, rank, world_size):
    train_augmentation = {
        'image_shape': (int(cfg['training']['height']), int(cfg['training']['width'])),
        'jittering': (0.2, 0.2, 0.2, 0.05),
        'crop_train_borders': (),
        'crop_eval_borders': (),
    }

    val_augmentation = {
        'image_shape': (int(cfg['training']['height']), int(cfg['training']['width'])),
        'jittering': (0.0, 0.0, 0.0, 0.0),
        'crop_train_borders': (),
        'crop_eval_borders': (),
    }

    train_dataset = construct_dataset(cfg, 'train', **train_augmentation)
    val_dataset = construct_dataset(cfg, 'val', **val_augmentation)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
    ) if world_size > 1 else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        sampler=train_sampler,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
        shuffle=(train_sampler is None),
    )

    # Keep complete evaluation set for final metrics.
    eval_batch_size = cfg.get('eval', {}).get('eval_batch_size', cfg['training']['batch_size'])
    eval_num_workers = cfg.get('eval', {}).get('eval_num_workers', 0)
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if world_size > 1 else None

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        sampler=val_sampler,
        shuffle=False if val_sampler is None else None,
        num_workers=eval_num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, val_dataloader, train_sampler


class Stage2ModelWithData(Stage2Model):
    def __init__(self, cfg, rank, train_dataloader, val_dataloader, train_sampler=None):
        super().__init__(cfg, rank)
        self._dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        self._train_sampler = train_sampler

    def get_train_sampler(self):
        return self._train_sampler


def main_worker(rank, world_size, args, cfg):
    torch.cuda.set_device(rank)

    if world_size > 1:
        setup_distributed(rank, world_size, backend=args.dist_backend)

    train_dataloader, val_dataloader, train_sampler = prepare_dataloaders(cfg, rank, world_size)
    model = Stage2ModelWithData(cfg, rank, train_dataloader, val_dataloader, train_sampler)

    if args.resume and os.path.exists(args.resume):
        model.load_res_flow_net(args.resume)
        if rank == 0:
            print(f'Resumed from {args.resume}')

    if world_size > 1:
        trainer = Stage2TrainerDDP(cfg, rank, use_tb=(rank == 0))
    else:
        trainer = Stage2Trainer(cfg, rank, use_tb=True)

    if args.eval_only:
        trainer.evaluate(model, model.val_dataloader())
    else:
        trainer.learn(model)

    if world_size > 1:
        cleanup_distributed()


def main():
    args = parse_args()
    cfg = get_config(args.config_file)

    if args.world_size > 0:
        world_size = args.world_size
    elif 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1

    if args.local_rank >= 0:
        rank = args.local_rank
    elif 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    else:
        rank = 0

    if rank == 0:
        print('=' * 80)
        print('Stage2 Training for DrivingForward MF Mode')
        print('=' * 80)
        print(f'Config file: {args.config_file}')
        print(f'World size: {world_size}')
        print(f'Rank: {rank}')
        print(f'Backend: {args.dist_backend}')
        print('=' * 80)

    if world_size > 1:
        mp.spawn(
            main_worker,
            args=(world_size, args, cfg),
            nprocs=world_size,
            join=True,
        )
    else:
        main_worker(rank, world_size, args, cfg)


if __name__ == '__main__':
    main()
