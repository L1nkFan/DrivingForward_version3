"""Stage2 DDP trainer and distributed helpers."""

import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .stage2_trainer import Stage2Trainer


class Stage2TrainerDDP(Stage2Trainer):
    """Distributed trainer for Stage2 residual flow optimization."""

    def __init__(self, cfg, rank=0, use_tb=True):
        super().__init__(cfg, rank, use_tb)
        self.ddp_enable = True
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def setup_distributed_model(self, model):
        """Wrap the trainable Stage2 module with DDP."""
        actual_model = model.model
        actual_model.cuda(self.rank)

        res_flow_net = actual_model.stage2_modules['res_flow_net']
        res_flow_net_ddp = DDP(
            res_flow_net,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False,
            broadcast_buffers=True,
        )
        actual_model.stage2_modules['res_flow_net'] = res_flow_net_ddp
        return model

    def reduce_dict(self, loss_dict):
        """Average tensor losses across all distributed workers."""
        if not dist.is_initialized():
            return loss_dict

        world_size = dist.get_world_size()
        reduced_dict = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                gathered_values = [torch.zeros_like(value) for _ in range(world_size)]
                dist.all_gather(gathered_values, value)
                reduced_dict[key] = torch.stack(gathered_values).mean()
            else:
                reduced_dict[key] = value
        return reduced_dict

    def learn(self, model):
        """Main distributed training loop."""
        model = self.setup_distributed_model(model)

        train_dataloader = model.train_dataloader()
        train_sampler = model.get_train_sampler() if hasattr(model, 'get_train_sampler') else None

        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)

        self.step = 0
        start_time = time.time()

        for self.epoch in range(self.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(self.epoch)

            if self.ddp_enable:
                dist.barrier()

            self.train_epoch(model, train_dataloader, start_time)

            if self.ddp_enable:
                dist.barrier()

            if self.rank == 0:
                # Keep DDP wrapper intact across ranks; only rank-0 performs checkpoint I/O.
                model.save_model(self.epoch)
                print('-' * 110)

            if self.ddp_enable:
                dist.barrier()

        if self.rank == 0:
            self.logger.close_tb()

    def train_epoch(self, model, data_loader, start_time):
        """Train one epoch in DDP mode."""
        model.set_train()

        criterion = getattr(model, 'criterion', None)
        if criterion is None and hasattr(model, 'model'):
            criterion = getattr(model.model, 'criterion', None)
        if criterion is not None and hasattr(criterion, 'set_training_progress'):
            criterion.set_training_progress(self.epoch, self.num_epochs, self.step, len(data_loader))

        if self.rank == 0:
            pbar = tqdm(
                total=len(data_loader),
                desc=f'Epoch {self.epoch}/{self.num_epochs}',
                mininterval=1,
            )
        else:
            pbar = None

        for batch_idx, inputs in enumerate(data_loader):
            before_op_time = time.time()
            inputs = model.to_device(inputs)

            model.optimizer.zero_grad(set_to_none=True)
            outputs, losses = model(inputs)
            losses['total_loss'].backward()
            model.optimizer.step()

            if self.ddp_enable:
                losses = self.reduce_dict(losses)

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
                    losses,
                )
                pbar.update(1)

            # In DDP all ranks must enter validation together; rank-0-only validation can deadlock NCCL.
            val_frequency = getattr(self, 'val_frequency', 1000)
            run_validation = (self.step > 0) and (self.step % val_frequency == 0)
            if run_validation:
                if self.ddp_enable:
                    dist.barrier()
                self.validate(model)
                if self.ddp_enable:
                    dist.barrier()

            self.step += 1

        if pbar is not None:
            pbar.close()

        model.lr_scheduler.step()

        if self.ddp_enable:
            dist.barrier()

    @torch.no_grad()
    def validate(self, model):
        """Run one validation step and log reconstruction metrics."""
        val_dataloader = model.val_dataloader()
        val_iter = iter(val_dataloader)

        model.set_val()

        try:
            inputs = next(val_iter)
        except StopIteration:
            val_iter = iter(val_dataloader)
            inputs = next(val_iter)

        inputs = model.to_device(inputs)
        outputs, losses = model(inputs)
        metrics = self.compute_reconstruction_metrics(inputs, outputs)

        if self.ddp_enable:
            metrics_tensor = torch.tensor(
                [metrics['psnr'], metrics['ssim'], metrics['lpips']],
                device=self.rank,
            )
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
            metrics['psnr'] = metrics_tensor[0].item()
            metrics['ssim'] = metrics_tensor[1].item()
            metrics['lpips'] = metrics_tensor[2].item()

        if self.rank == 0:
            print('\nValidation Results:')
            print(f"  PSNR: {metrics['psnr']:.4f}")
            print(f"  SSIM: {metrics['ssim']:.4f}")
            print(f"  LPIPS: {metrics['lpips']:.4f}")

            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    self.logger.log_scalar(f'val/{key}', value.item(), self.step)
            for key, value in metrics.items():
                self.logger.log_scalar(f'val/{key}', value, self.step)

        model.set_train()

    @torch.no_grad()
    def evaluate(self, model, eval_dataloader):
        """Evaluate on an entire dataloader in DDP mode."""
        model.set_val()

        all_metrics = defaultdict(list)

        if self.rank == 0:
            pbar = tqdm(total=len(eval_dataloader), desc='Evaluating')
        else:
            pbar = None

        for inputs in eval_dataloader:
            inputs = model.to_device(inputs)
            outputs, _ = model(inputs)
            metrics = self.compute_reconstruction_metrics(inputs, outputs)

            for key, value in metrics.items():
                all_metrics[key].append(value)

            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        'PSNR': np.mean(all_metrics['psnr']),
                        'SSIM': np.mean(all_metrics['ssim']),
                        'LPIPS': np.mean(all_metrics['lpips']),
                    }
                )

        if pbar is not None:
            pbar.close()

        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

        if self.ddp_enable:
            metrics_tensor = torch.tensor(
                [avg_metrics['psnr'], avg_metrics['ssim'], avg_metrics['lpips']],
                device=self.rank,
            )
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
            avg_metrics['psnr'] = metrics_tensor[0].item()
            avg_metrics['ssim'] = metrics_tensor[1].item()
            avg_metrics['lpips'] = metrics_tensor[2].item()

        if self.rank == 0:
            print('\nEvaluation Results:')
            for key, value in avg_metrics.items():
                print(f"  {key.upper()}: {value:.4f}")

        return avg_metrics


def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize torch distributed process group."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dist.barrier()

    if rank == 0:
        print('Distributed training initialized:')
        print(f'  - Backend: {backend}')
        print(f'  - World Size: {world_size}')
        print(f'  - Rank: {rank}')


def cleanup_distributed():
    """Destroy torch distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
