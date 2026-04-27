"""
支持分布式训练 (DDP) 的 DrivingForwardModel
继承自基础模型，添加分布式数据加载器支持
"""

import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

torch.manual_seed(0)

from dataset import construct_dataset
from network import *

from .base_model import BaseModel
from .geometry import Pose, ViewRendering
from .losses import MultiCamLoss, SingleCamLoss
from .gaussian import GaussianNetwork, depth2pc, pts2render, focal2fov, getProjectionMatrix, getWorld2View2, rotate_sh

_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename', 'token']


class DrivingForwardModelDDP(BaseModel):
    """
    支持分布式数据并行 (DDP) 的 DrivingForwardModel
    
    主要修改:
    1. 使用DistributedSampler实现数据分片
    2. 支持DDP模型包装
    3. 保持与单GPU版本的功能一致性
    """
    
    def __init__(self, cfg, rank):
        super(DrivingForwardModelDDP, self).__init__(cfg)
        self.rank = rank
        self.read_config(cfg)
        
        # 设置DDP相关属性
        self.ddp_enable = cfg.get('ddp', {}).get('ddp_enable', False)
        self.world_size = cfg.get('ddp', {}).get('world_size', 1)
        
        # 准备数据集 (使用DDP版本)
        self.prepare_dataset_ddp(cfg, rank)
        
        # 准备模型
        self.models = self.prepare_model(cfg, rank)
        
        # 初始化损失函数
        self.losses = self.init_losses(cfg, rank)
        
        # 初始化几何模块
        self.view_rendering, self.pose = self.init_geometry(cfg, rank)
        
        # 设置优化器
        self.set_optimizer()
        
        # 加载预训练权重 (仅在rank 0)
        if self.pretrain and rank == 0:
            self.load_weights()
            # 同步其他进程
            if self.ddp_enable:
                dist.barrier()
        elif self.ddp_enable:
            # 非rank 0进程等待rank 0加载权重
            dist.barrier()
        
        # 相机映射
        self.left_cam_dict = {2: 0, 0: 1, 4: 2, 1: 3, 5: 4, 3: 5}
        self.right_cam_dict = {0: 2, 1: 0, 2: 4, 3: 1, 4: 5, 5: 3}
    
    def read_config(self, cfg):
        """读取配置"""
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)
        self.pretrain = cfg.get('load', {}).get('pretrain', False)
        self.load_weights_dir = cfg.get('load', {}).get('load_weights_dir', './weight/weights_MF')
        self.models_to_load = cfg.get('load', {}).get('models_to_load', [])
    
    def init_geometry(self, cfg, rank):
        """初始化几何模块"""
        view_rendering = ViewRendering(cfg, rank)
        pose = Pose(cfg)
        return view_rendering, pose
    
    def init_losses(self, cfg, rank):
        """初始化损失函数"""
        if self.spatio_temporal or self.spatio:
            loss_model = MultiCamLoss(cfg, rank)
        else:
            loss_model = SingleCamLoss(cfg, rank)
        return loss_model
    
    def prepare_model(self, cfg, rank):
        """准备模型"""
        models = {}
        models['pose_net'] = self.set_posenet(cfg)
        models['depth_net'] = self.set_depthnet(cfg)
        if self.gaussian:
            models['gs_net'] = self.set_gaussiannet(cfg)
        
        # 使用DDP包装模型
        if self.ddp_enable:
            for model_name in models.keys():
                models[model_name] = DDP(
                    models[model_name],
                    device_ids=[rank],
                    output_device=rank,
                    find_unused_parameters=True,
                    broadcast_buffers=True
                )
        
        return models
    
    def set_posenet(self, cfg):
        """设置位姿网络"""
        return PoseNetwork(cfg).cuda(self.rank)
    
    def set_depthnet(self, cfg):
        """设置深度网络"""
        return DepthNetwork(cfg).cuda(self.rank)
    
    def set_gaussiannet(self, cfg):
        """设置高斯网络"""
        cross_view_heads = getattr(self, 'cross_view_num_heads', 4)
        return GaussianNetwork(
            rgb_dim=3,
            depth_dim=1,
            num_cams=self.num_cams,
            cross_view_num_heads=cross_view_heads,
            enable_cross_view_fusion=getattr(self, 'enable_cross_view_fusion', True),
        ).cuda(self.rank)
    
    def prepare_dataset_ddp(self, cfg, rank):
        """
        准备支持DDP的数据集和数据加载器
        
        Args:
            cfg: 配置字典
            rank: 当前进程的rank
        """
        if rank == 0:
            print('### Preparing Datasets for Distributed Training')
        
        if self.mode == 'train':
            self.set_train_dataloader_ddp(cfg, rank)
            if rank == 0:
                self.set_val_dataloader(cfg)
                
        if self.mode == 'eval':
            self.set_eval_dataloader(cfg)
    
    def set_train_dataloader_ddp(self, cfg, rank):
        """
        设置支持DDP的训练数据加载器
        
        使用DistributedSampler确保每个GPU处理不同的数据子集
        """
        # 数据增强和图像尺寸调整
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)), 
            'jittering': (0.2, 0.2, 0.2, 0.05),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # 构建训练数据集
        train_dataset = construct_dataset(cfg, 'train', **_augmentation)

        # 创建DistributedSampler
        # shuffle=True表示每个epoch数据顺序会打乱
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=rank,
            shuffle=True,
            seed=0
        )

        # 数据加载器配置
        dataloader_opts = {
            'batch_size': self.batch_size,
            'sampler': train_sampler,  # 使用DistributedSampler替代shuffle
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True,
            'persistent_workers': self.num_workers > 0  # 保持worker进程存活以提高性能
        }

        self._dataloaders['train'] = DataLoader(train_dataset, **dataloader_opts)
        
        # 保存sampler引用以便在每个epoch开始时shuffle
        self.train_sampler = train_sampler
        
        # 计算总步数
        # 注意: 在DDP中，每个进程处理 world_size 分之一的数据
        num_train_samples_per_gpu = len(train_dataset) // self.world_size
        self.num_total_steps = num_train_samples_per_gpu // self.batch_size * self.num_epochs
        
        if rank == 0:
            print(f"  - Total training samples: {len(train_dataset)}")
            print(f"  - Samples per GPU: {num_train_samples_per_gpu}")
            print(f"  - Batch size per GPU: {self.batch_size}")
            print(f"  - Effective batch size: {self.batch_size * self.world_size}")
            print(f"  - Total steps per epoch: {num_train_samples_per_gpu // self.batch_size}")
    
    def set_val_dataloader(self, cfg):
        """
        设置验证数据加载器 (仅在rank 0上运行)
        
        验证不需要DDP，只在主进程进行
        """
        # 图像尺寸调整
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # 构建验证数据集
        val_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['val'] = DataLoader(val_dataset, **dataloader_opts)
    
    def set_eval_dataloader(self, cfg):
        """
        设置评估数据加载器
        """
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        dataloader_opts = {
            'batch_size': self.eval_batch_size,
            'shuffle': False,
            'num_workers': self.eval_num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        eval_dataset = construct_dataset(cfg, 'eval', **_augmentation)
        self._dataloaders['eval'] = DataLoader(eval_dataset, **dataloader_opts)
    
    def get_train_sampler(self):
        """获取训练sampler以便在每个epoch更新"""
        return getattr(self, 'train_sampler', None)
    
    def set_optimizer(self):
        """设置优化器"""
        vit_params = []
        other_params = []
        for v in self.models.values():
            # 获取实际模型 (如果是DDP包装)
            actual_model = v.module if hasattr(v, 'module') else v
            
            if hasattr(actual_model, "vit_parameters"):
                vit_param_ids = {id(p) for p in actual_model.vit_parameters()}
                for p in v.parameters():
                    if id(p) in vit_param_ids:
                        vit_params.append(p)
                    else:
                        other_params.append(p)
            else:
                other_params += list(v.parameters())

        param_groups = [{"params": other_params, "lr": self.learning_rate}]
        vit_lr = getattr(self, "vit_learning_rate", 1e-5)
        if len(vit_params) > 0:
            param_groups.append({"params": vit_params, "lr": vit_lr})

        self.optimizer = optim.Adam(param_groups)

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            self.scheduler_step_size,
            0.1
        )
    
    def process_batch(self, inputs, rank):
        """
        处理一个batch的数据
        """
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(rank) for k in range(len(inputs[key]))]
                if 'ego_pose' in key:
                    inputs[key] = [ipt[k].float().to(rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(rank)

        outputs = self.estimate(inputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses
    
    def estimate(self, inputs):
        """
        估计网络输出
        """
        # 预计算外参矩阵的逆
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])
        
        # 初始化输出字典
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        pose_pred = self.predict_pose(inputs)
        depth_feats = self.predict_depth(inputs)

        for cam in range(self.num_cams):
            if self.mode != 'train':
                outputs[('cam', cam)].update({('cam_T_cam', 0, 1): inputs[('cam_T_cam', 0, 1)][:, cam, ...]})
                outputs[('cam', cam)].update({('cam_T_cam', 0, -1): inputs[('cam_T_cam', 0, -1)][:, cam, ...]})
            elif self.mode == 'train':
                outputs[('cam', cam)].update(pose_pred[('cam', cam)])
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])
            
        self.compute_depth_maps(inputs, outputs)
        return outputs
    
    def predict_pose(self, inputs):
        """预测位姿"""
        net = self.models['pose_net']
        # 如果是DDP模型，使用module访问
        if hasattr(net, 'module'):
            net = net.module
        pose = self.pose.compute_pose(net, inputs)
        return pose
    
    def predict_depth(self, inputs):
        """预测深度"""
        net = self.models['depth_net']
        if hasattr(net, 'module'):
            net = net.module
        depth_feats = net(inputs)
        return depth_feats
    
    def compute_depth_maps(self, inputs, outputs):
        """计算深度图"""
        source_scale = 0
        for cam in range(self.num_cams):
            ref_K = inputs[('K', source_scale)][:, cam, ...]
            for scale in self.scales:
                disp = outputs[('cam', cam)][('disp', scale)]
                outputs[('cam', cam)][('depth', 0, scale)] = self.to_depth(disp, ref_K)
                if self.novel_view_mode == 'MF':
                    disp_last = outputs[('cam', cam)][('disp', -1, scale)]
                    outputs[('cam', cam)][('depth', -1, scale)] = self.to_depth(disp_last, ref_K)
                    disp_next = outputs[('cam', cam)][('disp', 1, scale)]
                    outputs[('cam', cam)][('depth', 1, scale)] = self.to_depth(disp_next, ref_K)
    
    def to_depth(self, disp_in, K_in):
        """将视差转换为深度"""
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        disp_range = max_disp - min_disp

        disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1 / disp
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2) / self.focal_length_scale
    
    def get_gaussian_data(self, inputs, outputs, cam):
        """获取高斯数据"""
        bs, _, height, width = inputs[('color', 0, 0)][:, cam, ...].shape
        zfar = self.max_depth
        znear = 0.01

        if self.novel_view_mode == 'MF':
            for frame_id in self.frame_ids:
                if frame_id == 0:
                    outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = inputs['extrinsics_inv'][:, cam, ...]
                    outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = inputs['extrinsics'][:, cam, ...]
                    FovX_list = []
                    FovY_list = []
                    world_view_transform_list = []
                    full_proj_transform_list = []
                    camera_center_list = []
                    for i in range(bs):
                        intr = inputs[('K', 0)][:, cam, ...][i, :]
                        extr = inputs['extrinsics_inv'][:, cam, ...][i, :]
                        FovX = focal2fov(intr[0, 0], width)
                        FovY = focal2fov(intr[1, 1], height)
                        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, K=intr, h=height, w=width).transpose(0, 1).cuda(self.rank)
                        world_view_transform = torch.tensor(extr).transpose(0, 1).cuda(self.rank)
                        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                        camera_center = world_view_transform.inverse()[3, :3]

                        FovX_list.append(FovX)
                        FovY_list.append(FovY)
                        world_view_transform_list.append(world_view_transform.unsqueeze(0))
                        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
                        camera_center_list.append(camera_center.unsqueeze(0))

                    outputs[('cam', cam)][('FovX', frame_id, 0)] = torch.tensor(FovX_list).cuda(self.rank)
                    outputs[('cam', cam)][('FovY', frame_id, 0)] = torch.tensor(FovY_list).cuda(self.rank)
                    outputs[('cam', cam)][('world_view_transform', frame_id, 0)] = torch.cat(world_view_transform_list, dim=0)
                    outputs[('cam', cam)][('full_proj_transform', frame_id, 0)] = torch.cat(full_proj_transform_list, dim=0)
                    outputs[('cam', cam)][('camera_center', frame_id, 0)] = torch.cat(camera_center_list, dim=0)
                else:
                    outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = \
                        torch.matmul(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)], inputs['extrinsics_inv'][:, cam, ...])
                    outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = \
                        torch.matmul(inputs['extrinsics'][:, cam, ...], torch.inverse(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)]))
                outputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(outputs[('cam', cam)][('depth', frame_id, 0)], outputs[('cam', cam)][('e2c_extr', frame_id, 0)], inputs[('K', 0)][:, cam, ...])
                valid = outputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
                outputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)
                # Gaussian maps are computed in a single batched multi-view pass.
        elif self.novel_view_mode == 'SF':
            frame_id = 0
            outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = inputs['extrinsics_inv'][:, cam, ...]
            outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = inputs['extrinsics'][:, cam, ...]
            outputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(outputs[('cam', cam)][('depth', frame_id, 0)], outputs[('cam', cam)][('e2c_extr', frame_id, 0)], inputs[('K', 0)][:, cam, ...])
            valid = outputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
            outputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)
            # Gaussian maps are computed in a single batched multi-view pass.

            # 新视角
            for frame_id in self.frame_ids[1:]:
                outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = \
                    torch.matmul(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)], inputs['extrinsics_inv'][:, cam, ...])
                outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = \
                    torch.matmul(inputs['extrinsics'][:, cam, ...], torch.inverse(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)]))
                
                FovX_list = []
                FovY_list = []
                world_view_transform_list = []
                full_proj_transform_list = []
                camera_center_list = []
                for i in range(bs):
                    intr = inputs[('K', 0)][:, cam, ...][i, :]
                    extr = inputs['extrinsics_inv'][:, cam, ...][i, :]
                    T_i = outputs[('cam', cam)][('cam_T_cam', 0, frame_id)][i, :]
                    FovX = focal2fov(intr[0, 0], width)
                    FovY = focal2fov(intr[1, 1], height)
                    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, K=intr, h=height, w=width).transpose(0, 1).cuda(self.rank)
                    world_view_transform = torch.matmul(T_i, torch.tensor(extr).cuda(self.rank)).transpose(0, 1)
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                    camera_center = world_view_transform.inverse()[3, :3]
                    FovX_list.append(FovX)
                    FovY_list.append(FovY)
                    world_view_transform_list.append(world_view_transform.unsqueeze(0))
                    full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
                    camera_center_list.append(camera_center.unsqueeze(0))
                outputs[('cam', cam)][('FovX', frame_id, 0)] = torch.tensor(FovX_list).cuda(self.rank)
                outputs[('cam', cam)][('FovY', frame_id, 0)] = torch.tensor(FovY_list).cuda(self.rank)
                outputs[('cam', cam)][('world_view_transform', frame_id, 0)] = torch.cat(world_view_transform_list, dim=0)
                outputs[('cam', cam)][('full_proj_transform', frame_id, 0)] = torch.cat(full_proj_transform_list, dim=0)
                outputs[('cam', cam)][('camera_center', frame_id, 0)] = torch.cat(camera_center_list, dim=0)
    
    def compute_gaussian_maps_batch(self, inputs, outputs):
        """Predict Gaussian maps using one multi-view forward pass per frame."""
        gs_net = self.models['gs_net']
        if hasattr(gs_net, 'module'):
            gs_net = gs_net.module

        frame_ids = self.frame_ids if self.novel_view_mode == 'MF' else [0]

        for frame_id in frame_ids:
            imgs = inputs[('color', frame_id, 0)]
            depths = torch.stack(
                [outputs[('cam', cam)][('depth', frame_id, 0)] for cam in range(self.num_cams)],
                dim=1,
            )

            num_img_feat_levels = len(outputs[('cam', 0)][('img_feat', frame_id, 0)])
            img_feats = []
            for lvl in range(num_img_feat_levels):
                lvl_feat = torch.stack(
                    [outputs[('cam', cam)][('img_feat', frame_id, 0)][lvl] for cam in range(self.num_cams)],
                    dim=1,
                )
                img_feats.append(lvl_feat)

            rot_maps, scale_maps, opacity_maps, sh_maps = gs_net(imgs, depths, img_feats)

            if rot_maps.dim() == 4:
                b, n_cam = imgs.shape[:2]
                rot_maps = rot_maps.view(b, n_cam, *rot_maps.shape[1:])
                scale_maps = scale_maps.view(b, n_cam, *scale_maps.shape[1:])
                opacity_maps = opacity_maps.view(b, n_cam, *opacity_maps.shape[1:])
                sh_maps = sh_maps.view(b, n_cam, *sh_maps.shape[1:])

            for cam in range(self.num_cams):
                c2w_rotations = rearrange(
                    outputs[('cam', cam)][('c2e_extr', frame_id, 0)][..., :3, :3],
                    "k i j -> k () () () i j",
                )
                sh_cam = rotate_sh(sh_maps[:, cam, ...], c2w_rotations[..., None, :, :])
                outputs[('cam', cam)][('rot_maps', frame_id, 0)] = rot_maps[:, cam, ...]
                outputs[('cam', cam)][('scale_maps', frame_id, 0)] = scale_maps[:, cam, ...]
                outputs[('cam', cam)][('opacity_maps', frame_id, 0)] = opacity_maps[:, cam, ...]
                outputs[('cam', cam)][('sh_maps', frame_id, 0)] = sh_cam

    def compute_losses(self, inputs, outputs):
        """计算损失"""
        losses = 0
        loss_fn = defaultdict(list)
        loss_mean = defaultdict(float)

        # 计算高斯数据
        if self.gaussian:
            for cam in range(self.num_cams):
                self.get_gaussian_data(inputs, outputs, cam)
            self.compute_gaussian_maps_batch(inputs, outputs)

        # 生成图像并计算每个相机的损失
        for cam in range(self.num_cams):
            self.pred_cam_imgs(inputs, outputs, cam)
            if self.gaussian:
                self.pred_gaussian_imgs(inputs, outputs, cam)
            cam_loss, loss_dict = self.losses(inputs, outputs, cam)
            
            losses += cam_loss
            for k, v in loss_dict.items():
                loss_fn[k].append(v)

        losses /= self.num_cams
        
        for k in loss_fn.keys():
            loss_mean[k] = sum(loss_fn[k]) / float(len(loss_fn[k]))

        loss_mean['total_loss'] = losses
        return loss_mean
    
    def pred_cam_imgs(self, inputs, outputs, cam):
        """预测相机图像"""
        rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)
        self.view_rendering(inputs, outputs, cam, rel_pose_dict)
    
    def pred_gaussian_imgs(self, inputs, outputs, cam):
        """预测高斯图像"""
        if self.novel_view_mode == 'MF':
            outputs[('cam', cam)][('gaussian_color', 0, 0)] = \
                pts2render(inputs=inputs,
                           outputs=outputs,
                           cam_num=self.num_cams,
                           novel_cam=cam,
                           novel_frame_id=0,
                           bg_color=[1.0, 1.0, 1.0],
                           mode=self.novel_view_mode)
        elif self.novel_view_mode == 'SF':
            for novel_frame_id in self.frame_ids[1:]:
                outputs[('cam', cam)][('gaussian_color', novel_frame_id, 0)] = \
                    pts2render(inputs=inputs,
                               outputs=outputs,
                               cam_num=self.num_cams,
                               novel_cam=cam,
                               novel_frame_id=novel_frame_id,
                               bg_color=[1.0, 1.0, 1.0],
                               mode=self.novel_view_mode)
    
    def save_model(self, epoch):
        """
        保存模型 (仅在rank 0)
        
        DDP包装的模型需要使用module.state_dict()
        """
        if self.rank != 0:
            return
            
        curr_model_weights_dir = os.path.join(self.save_weights_root, f'weights_{epoch}')
        os.makedirs(curr_model_weights_dir, exist_ok=True)

        for model_name, model in self.models.items():
            model_file_path = os.path.join(curr_model_weights_dir, f'{model_name}.pth')
            
            # 如果是DDP模型，获取内部的module
            if hasattr(model, 'module'):
                to_save = model.module.state_dict()
            else:
                to_save = model.state_dict()
            
            torch.save(to_save, model_file_path)
        
        # 保存优化器
        optim_file_path = os.path.join(curr_model_weights_dir, 'adam.pth')
        torch.save(self.optimizer.state_dict(), optim_file_path)
        
        print(f"Model saved to {curr_model_weights_dir}")
