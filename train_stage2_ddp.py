import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, DistributedSampler

# Deterministic setup
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.misc import get_config
from stage2_trainer import Stage2TrainerDDP
from stage2_trainer.stage2_trainer_ddp import setup_distributed, cleanup_distributed
from stage2_trainer.model_factory import build_stage2_model
from dataset import construct_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Stage2 DDP training/evaluation script')
    parser.add_argument(
        '--config_file',
        default='./configs/nuscenes/phase2_training.yaml',
        type=str,
        help='config yaml file for stage2 training',
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
    parser.add_argument(
        '--detect_anomaly',
        action='store_true',
        help='enable torch autograd anomaly detection',
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
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        sampler=train_sampler,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
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


class Stage2ModelWithData:
    def __init__(self, cfg, rank, train_dataloader, val_dataloader, train_sampler=None):
        self.model = build_stage2_model(cfg, rank)
        self._dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        self._train_sampler = train_sampler

        self.rank = self.model.rank
        self.stage2_modules = self.model.stage2_modules
        self.optimizer = self.model.optimizer
        self.lr_scheduler = self.model.lr_scheduler

    def get_train_sampler(self):
        return self._train_sampler

    def train_dataloader(self):
        return self._dataloaders['train']

    def val_dataloader(self):
        return self._dataloaders['val']

    def to_device(self, inputs):
        return self.model.to_device(inputs)

    def set_train(self):
        self.model.set_train()

    def set_val(self):
        self.model.set_val()

    def save_model(self, epoch):
        self.model.save_model(epoch)

    def load_res_flow_net(self, checkpoint_path):
        self.model.load_res_flow_net(checkpoint_path)

    def forward(self, inputs):
        return self.model(inputs)

    def __call__(self, inputs):
        return self.model(inputs)


def main_worker(rank, world_size, args, cfg):
    torch.cuda.set_device(rank)

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        if rank == 0:
            print('[DEBUG] torch.autograd anomaly detection enabled')

    setup_distributed(rank, world_size, backend=args.dist_backend)

    train_dataloader, val_dataloader, train_sampler = prepare_dataloaders(cfg, rank, world_size)
    model = Stage2ModelWithData(cfg, rank, train_dataloader, val_dataloader, train_sampler)

    if rank == 0:
        model_cfg = cfg.get('model', {})
        temporal = model_cfg.get('temporal', False)
        spatio = model_cfg.get('spatio', False)
        spatio_temporal = model_cfg.get('spatio_temporal', False)

        print('\n' + '=' * 80)
        print('Training Mode Configuration:')
        if temporal and not (spatio or spatio_temporal):
            print('  Mode: Temporal')
        elif spatio and not (temporal or spatio_temporal):
            print('  Mode: Spatio')
        elif spatio_temporal or (temporal and spatio):
            print('  Mode: Spatio-Temporal')
        else:
            print('  Mode: Default Spatio-Temporal')
        print('=' * 80 + '\n')

    if args.resume and os.path.exists(args.resume):
        model.load_res_flow_net(args.resume)
        if rank == 0:
            print(f'Resumed from {args.resume}')

    trainer = Stage2TrainerDDP(cfg, rank, use_tb=(rank == 0))

    if args.eval_only:
        trainer.evaluate(model, model.val_dataloader())
    else:
        trainer.learn(model)

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
    elif 'LOCAL_RANK' in os.environ:
        rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0

    if rank == 0:
        print('=' * 80)
        print('Stage2 DDP Training for DrivingForward MF Mode')
        print('=' * 80)
        print(f'Config file: {args.config_file}')
        print(f'World size: {world_size}')
        print(f'Rank: {rank}')
        print(f'Backend: {args.dist_backend}')
        print('=' * 80)
        print('Stage2 Configuration:')
        for section, params in cfg.items():
            print(f'  [{section}]')
            for key, value in params.items():
                print(f'    {key}: {value}')
        print('=' * 80)

    main_worker(rank, world_size, args, cfg)


if __name__ == '__main__':
    main()


