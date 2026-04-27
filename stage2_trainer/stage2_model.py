"""
第二阶段模型 - DrivingForward MF模式
整合第一阶段冻结网络和第二阶段可训练模块
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# 导入第一阶段模型组件
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network import PoseNetwork, DepthNetwork
from models.gaussian import GaussianNetwork
from models.geometry import Pose, ViewRendering
from models.gaussian import depth2pc, pts2render, focal2fov, getProjectionMatrix, rotate_sh
from einops import rearrange

# 导入第二阶段模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage2_modules import RigidFlowCalculator, ResFlowNet, DynamicGaussianGenerator, Stage2Loss
from stage2_modules.rigid_flow import batch_warp_image_with_flow


_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename', 'token']


class Stage2Model(nn.Module):
    """
    第二阶段模型
    冻结: PoseNet, DepthNet, GaussianNet
    可训练: ResFlowNet
    """
    
    def __init__(self, cfg, rank=0):
        super(Stage2Model, self).__init__()
        self.cfg = cfg
        self.rank = rank
        self.read_config(cfg)
        
        # 初始化第一阶段冻结网络
        self.models = self.prepare_frozen_models(cfg, rank)
        
        # 初始化第二阶段可训练模块
        self.stage2_modules = self.prepare_stage2_modules(cfg, rank)
        
        # 初始化几何计算模块
        self.view_rendering, self.pose = self.init_geometry(cfg, rank)
        
        # 初始化损失函数
        self.criterion = Stage2Loss(
            lambda_warp=self.lambda_warp,
            lambda_consist=self.lambda_consist,
            lambda_render=self.lambda_render,
            rank=rank
        )
        
        # 加载第一阶段权重并冻结
        self.load_and_freeze_stage1_weights()
        
        # 设置优化器 (仅优化ResFlowNet)
        self.set_optimizer()
        
    def read_config(self, cfg):
        """读取配置"""
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)
    
    def prepare_frozen_models(self, cfg, rank):
        """准备第一阶段冻结网络"""
        models = {}
        models['pose_net'] = PoseNetwork(cfg).cuda(rank)
        models['depth_net'] = DepthNetwork(cfg).cuda(rank)
        if self.gaussian:
            models['gs_net'] = GaussianNetwork(
                rgb_dim=3,
                depth_dim=1,
                num_cams=self.num_cams,
                cross_view_num_heads=getattr(self, 'cross_view_num_heads', 4),
                enable_cross_view_fusion=getattr(self, 'enable_cross_view_fusion', True),
            ).cuda(rank)
        return models
    
    def prepare_stage2_modules(self, cfg, rank):
        """准备第二阶段可训练模块"""
        modules = {}
        # 刚性流计算器
        modules['rigid_flow_calc'] = RigidFlowCalculator(
            height=self.height,
            width=self.width
        ).cuda(rank)
        # 残差流网络
        modules['res_flow_net'] = ResFlowNet(
            num_cams=self.num_cams,
            base_channels=64
        ).cuda(rank)
        # 动态高斯生成器
        modules['dynamic_gaussian'] = DynamicGaussianGenerator(
            height=self.height,
            width=self.width
        ).cuda(rank)
        return modules
    
    def init_geometry(self, cfg, rank):
        """初始化几何计算模块"""
        from models.geometry import ViewRendering, Pose
        view_rendering = ViewRendering(cfg, rank)
        pose = Pose(cfg)
        return view_rendering, pose
    
    def load_and_freeze_stage1_weights(self):
        """加载第一阶段权重并冻结网络"""
        # 加载PoseNet
        pose_net_path = os.path.join(self.stage1_weights_path, 'pose_net.pth')
        if os.path.exists(pose_net_path):
            self.models['pose_net'].load_state_dict(torch.load(pose_net_path, map_location=f'cuda:{self.rank}'))
            print(f"[Rank {self.rank}] Loaded PoseNet from {pose_net_path}")
        else:
            print(f"[Rank {self.rank}] Warning: PoseNet weights not found at {pose_net_path}")
        
        # 加载DepthNet
        depth_net_path = os.path.join(self.stage1_weights_path, 'depth_net.pth')
        if os.path.exists(depth_net_path):
            self.models['depth_net'].load_state_dict(torch.load(depth_net_path, map_location=f'cuda:{self.rank}'))
            print(f"[Rank {self.rank}] Loaded DepthNet from {depth_net_path}")
        else:
            print(f"[Rank {self.rank}] Warning: DepthNet weights not found at {depth_net_path}")
        
        # 加载GaussianNet
        if self.gaussian:
            gs_candidates = ['gs_net.pth', 'gaussian_net.pth']
            gs_net_path = None
            for name in gs_candidates:
                candidate = os.path.join(self.stage1_weights_path, name)
                if os.path.exists(candidate):
                    gs_net_path = candidate
                    break
            if gs_net_path is not None:
                self.models['gs_net'].load_state_dict(torch.load(gs_net_path, map_location=f'cuda:{self.rank}'))
                print(f"[Rank {self.rank}] Loaded GaussianNet from {gs_net_path}")
            else:
                print(f"[Rank {self.rank}] Warning: GaussianNet weights not found in {self.stage1_weights_path} ({gs_candidates})")
        
        # 冻结所有第一阶段网络
        for model_name, model in self.models.items():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            print(f"[Rank {self.rank}] Frozen {model_name}")
    
    def set_optimizer(self):
        """设置优化器 - 仅优化ResFlowNet"""
        # 只收集ResFlowNet的可训练参数
        trainable_params = list(self.stage2_modules['res_flow_net'].parameters())
        
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999)
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.scheduler_step_size,
            gamma=0.1
        )
    
    def to_device(self, inputs):
        """将输入数据移动到设备"""
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(self.rank) for k in range(len(inputs[key]))]
                elif 'ego_pose' in key:
                    inputs[key] = [ipt[k].float().to(self.rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(self.rank)
        return inputs
    
    @torch.no_grad()
    def compute_stage1_outputs(self, inputs):
        """
        计算第一阶段的输出 (冻结网络)
        包括: 位姿、深度、高斯参数
        """
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}
        
        # 预计算外参逆矩阵
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])
        
        # 位姿预测
        pose_pred = self.predict_pose(inputs)
        
        # 深度预测
        depth_feats = self.predict_depth(inputs)
        
        # 整合输出
        for cam in range(self.num_cams):
            if self.mode != 'train':
                outputs[('cam', cam)].update({('cam_T_cam', 0, 1): inputs[('cam_T_cam', 0, 1)][:, cam, ...]})
                outputs[('cam', cam)].update({('cam_T_cam', 0, -1): inputs[('cam_T_cam', 0, -1)][:, cam, ...]})
            else:
                outputs[('cam', cam)].update(pose_pred[('cam', cam)])
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])
        
        # 计算深度图
        self.compute_depth_maps(inputs, outputs)
        
        # 计算高斯数据
        if self.gaussian:
            for cam in range(self.num_cams):
                self.get_gaussian_data(inputs, outputs, cam)
        
        return outputs
    
    def predict_pose(self, inputs):
        """预测位姿"""
        net = self.models['pose_net']
        pose = self.pose.compute_pose(net, inputs)
        return pose
    
    def predict_depth(self, inputs):
        """预测深度"""
        net = self.models['depth_net']
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
        """视差转深度"""
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        disp_range = max_disp - min_disp
        
        disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1 / disp
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2) / self.focal_length_scale
    
    def get_gaussian_data(self, inputs, outputs, cam):
        """计算高斯数据"""
        from models.gaussian.utils import getProjectionMatrix, focal2fov
        
        bs, _, height, width = inputs[('color', 0, 0)][:, cam, ...].shape
        zfar = self.max_depth
        znear = 0.01
        
        if self.novel_view_mode == 'MF':
            for frame_id in [0, -1, 1]:
                outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = inputs['extrinsics_inv'][:, cam, ...] if frame_id == 0 else \
                    torch.matmul(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)], inputs['extrinsics_inv'][:, cam, ...])
                outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = inputs['extrinsics'][:, cam, ...] if frame_id == 0 else \
                    torch.matmul(inputs['extrinsics'][:, cam, ...], torch.inverse(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)]))
                
                outputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(
                    outputs[('cam', cam)][('depth', frame_id, 0)],
                    outputs[('cam', cam)][('e2c_extr', frame_id, 0)],
                    inputs[('K', 0)][:, cam, ...]
                )
                valid = outputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
                outputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)
                
                # 高斯参数预测
                rot_maps, scale_maps, opacity_maps, sh_maps = self.models['gs_net'](
                    inputs[('color', frame_id, 0)][:, cam, ...],
                    outputs[('cam', cam)][('depth', frame_id, 0)],
                    outputs[('cam', cam)][('img_feat', frame_id, 0)]
                )
                
                c2w_rotations = rearrange(
                    outputs[('cam', cam)][('c2e_extr', frame_id, 0)][..., :3, :3],
                    "k i j -> k () () () i j"
                )
                sh_maps = rotate_sh(sh_maps, c2w_rotations[..., None, :, :])
                
                outputs[('cam', cam)][('rot_maps', frame_id, 0)] = rot_maps
                outputs[('cam', cam)][('scale_maps', frame_id, 0)] = scale_maps
                outputs[('cam', cam)][('opacity_maps', frame_id, 0)] = opacity_maps
                outputs[('cam', cam)][('sh_maps', frame_id, 0)] = sh_maps
                
                # 计算相机参数 (用于渲染)
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
                    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, K=intr, h=height, w=width).transpose(0, 1).cuda()
                    world_view_transform = torch.tensor(extr).transpose(0, 1).cuda()
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                    camera_center = world_view_transform.inverse()[3, :3]
                    
                    FovX_list.append(FovX)
                    FovY_list.append(FovY)
                    world_view_transform_list.append(world_view_transform.unsqueeze(0))
                    full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
                    camera_center_list.append(camera_center.unsqueeze(0))
                
                outputs[('cam', cam)][('FovX', frame_id, 0)] = torch.tensor(FovX_list).cuda()
                outputs[('cam', cam)][('FovY', frame_id, 0)] = torch.tensor(FovY_list).cuda()
                outputs[('cam', cam)][('world_view_transform', frame_id, 0)] = torch.cat(world_view_transform_list, dim=0)
                outputs[('cam', cam)][('full_proj_transform', frame_id, 0)] = torch.cat(full_proj_transform_list, dim=0)
                outputs[('cam', cam)][('camera_center', frame_id, 0)] = torch.cat(camera_center_list, dim=0)
    
    def compute_stage2_outputs(self, inputs, outputs):
        """
        计算第二阶段的输出 (可训练模块)
        包括: 刚性流、残差流、动态高斯
        """
        batch_size = inputs[('color', 0, 0)].shape[0]
        
        # 准备存储结果
        F_rigid_t_minus_1_list = []
        F_rigid_t_plus_1_list = []
        mask_t_minus_1_list = []
        mask_t_plus_1_list = []
        
        # 对每个相机计算
        for cam in range(self.num_cams):
            # 获取深度和位姿
            depth_t_minus_1 = outputs[('cam', cam)][('depth', -1, 0)]  # [B, 1, H, W]
            depth_t_plus_1 = outputs[('cam', cam)][('depth', 1, 0)]  # [B, 1, H, W]
            
            T_t_minus_1_to_t = outputs[('cam', cam)][('cam_T_cam', 0, -1)]  # [B, 4, 4]
            T_t_plus_1_to_t = outputs[('cam', cam)][('cam_T_cam', 0, 1)]  # [B, 4, 4]
            
            K = inputs[('K', 0)][:, cam, ...]  # [B, 3, 3]
            
            # 计算刚性流
            F_rigid_t_minus_1, mask_t_minus_1 = self.stage2_modules['rigid_flow_calc'](
                depth_t_minus_1, K, T_t_minus_1_to_t
            )
            F_rigid_t_plus_1, mask_t_plus_1 = self.stage2_modules['rigid_flow_calc'](
                depth_t_plus_1, K, T_t_plus_1_to_t
            )
            
            F_rigid_t_minus_1_list.append(F_rigid_t_minus_1)
            F_rigid_t_plus_1_list.append(F_rigid_t_plus_1)
            mask_t_minus_1_list.append(mask_t_minus_1)
            mask_t_plus_1_list.append(mask_t_plus_1)
        
        # 堆叠多相机结果 [B, N, C, H, W]
        F_rigid_t_minus_1 = torch.stack(F_rigid_t_minus_1_list, dim=1)
        F_rigid_t_plus_1 = torch.stack(F_rigid_t_plus_1_list, dim=1)
        mask_t_minus_1 = torch.stack(mask_t_minus_1_list, dim=1)
        mask_t_plus_1 = torch.stack(mask_t_plus_1_list, dim=1)
        
        # 使用刚性流warp图像
        I_t_minus_1 = inputs[('color', -1, 0)]  # [B, N, 3, H, W]
        I_t_plus_1 = inputs[('color', 1, 0)]  # [B, N, 3, H, W]
        I_t = inputs[('color', 0, 0)]  # [B, N, 3, H, W]
        
        warped_t_minus_1 = batch_warp_image_with_flow(I_t_minus_1, F_rigid_t_minus_1)
        warped_t_plus_1 = batch_warp_image_with_flow(I_t_plus_1, F_rigid_t_plus_1)
        
        # 预测残差流
        F_residual_t_minus_1 = self.stage2_modules['res_flow_net'](
            warped_t_minus_1, I_t, F_rigid_t_minus_1
        )
        F_residual_t_plus_1 = self.stage2_modules['res_flow_net'](
            warped_t_plus_1, I_t, F_rigid_t_plus_1
        )
        
        # 计算总光流
        F_total_t_minus_1 = F_rigid_t_minus_1 + F_residual_t_minus_1
        F_total_t_plus_1 = F_rigid_t_plus_1 + F_residual_t_plus_1
        
        # 存储结果
        stage2_outputs = {
            'F_rigid_t_minus_1': F_rigid_t_minus_1,
            'F_rigid_t_plus_1': F_rigid_t_plus_1,
            'F_residual_t_minus_1': F_residual_t_minus_1,
            'F_residual_t_plus_1': F_residual_t_plus_1,
            'F_total_t_minus_1': F_total_t_minus_1,
            'F_total_t_plus_1': F_total_t_plus_1,
            'mask_t_minus_1': mask_t_minus_1,
            'mask_t_plus_1': mask_t_plus_1,
        }
        
        return stage2_outputs
    
    def render_novel_view(self, inputs, outputs, stage2_outputs):
        """
        Render target view by fusing warped t-1/t+1 frames.
        Static regions keep the original mask-average behavior.
        Dynamic regions use residual-flow-aware confidence blending.
        """
        from stage2_modules.rigid_flow import batch_warp_image_with_flow

        F_total_t_minus_1 = stage2_outputs.get('F_total_t_minus_1')
        F_total_t_plus_1 = stage2_outputs.get('F_total_t_plus_1')
        if F_total_t_minus_1 is None or F_total_t_plus_1 is None:
            return inputs[('color', 0, 0)]

        I_t_minus_1 = inputs[('color', -1, 0)]
        I_t_plus_1 = inputs[('color', 1, 0)]

        warped_t_minus_1 = batch_warp_image_with_flow(I_t_minus_1, F_total_t_minus_1)
        warped_t_plus_1 = batch_warp_image_with_flow(I_t_plus_1, F_total_t_plus_1)

        mask_t_minus_1 = stage2_outputs.get(
            'mask_t_minus_1', torch.ones_like(F_total_t_minus_1[:, :, 0:1, :, :])
        )
        mask_t_plus_1 = stage2_outputs.get(
            'mask_t_plus_1', torch.ones_like(F_total_t_plus_1[:, :, 0:1, :, :])
        )

        # Baseline blend (old behavior): robust for static regions.
        base_w_m1 = mask_t_minus_1.expand(-1, -1, 3, -1, -1)
        base_w_p1 = mask_t_plus_1.expand(-1, -1, 3, -1, -1)
        base_sum = base_w_m1 + base_w_p1 + 1e-6
        rendered_base = (warped_t_minus_1 * base_w_m1 + warped_t_plus_1 * base_w_p1) / base_sum

        if not getattr(self, 'use_dynamic_fusion', True):
            return rendered_base

        # Dynamic cues from residual-flow magnitudes.
        F_residual_t_minus_1 = stage2_outputs.get(
            'F_residual_t_minus_1', torch.zeros_like(F_total_t_minus_1)
        )
        F_residual_t_plus_1 = stage2_outputs.get(
            'F_residual_t_plus_1', torch.zeros_like(F_total_t_plus_1)
        )
        res_mag_t_minus_1 = torch.linalg.norm(F_residual_t_minus_1, dim=2, keepdim=True)
        res_mag_t_plus_1 = torch.linalg.norm(F_residual_t_plus_1, dim=2, keepdim=True)

        dyn_score = 0.5 * (res_mag_t_minus_1 + res_mag_t_plus_1)
        dyn_gate = torch.clamp(dyn_score / (dyn_score + 0.30), 0.0, 1.0)

        # In dynamic regions, prefer source with smaller residual-flow magnitude.
        flow_beta = 0.12
        conf_t_minus_1 = mask_t_minus_1 * torch.exp(-flow_beta * res_mag_t_minus_1)
        conf_t_plus_1 = mask_t_plus_1 * torch.exp(-flow_beta * res_mag_t_plus_1)
        conf_w_m1 = conf_t_minus_1.expand(-1, -1, 3, -1, -1)
        conf_w_p1 = conf_t_plus_1.expand(-1, -1, 3, -1, -1)
        conf_sum = conf_w_m1 + conf_w_p1 + 1e-6
        rendered_dynamic = (warped_t_minus_1 * conf_w_m1 + warped_t_plus_1 * conf_w_p1) / conf_sum

        dyn_gate_rgb = dyn_gate.expand(-1, -1, 3, -1, -1)
        rendered_I_t = rendered_base * (1.0 - dyn_gate_rgb) + rendered_dynamic * dyn_gate_rgb

        return rendered_I_t
    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 输入数据字典
            
        Returns:
            outputs: 输出字典
            losses: 损失字典
        """
        # 1. 第一阶段: 冻结网络前向计算
        with torch.no_grad():
            outputs = self.compute_stage1_outputs(inputs)
        
        # 2. 第二阶段: 可训练模块前向计算
        stage2_outputs = self.compute_stage2_outputs(inputs, outputs)
        
        # 3. 渲染新视角
        rendered_I_t = self.render_novel_view(inputs, outputs, stage2_outputs)
        
        # 4. 计算损失
        I_t = inputs[('color', 0, 0)]
        I_t_minus_1 = inputs[('color', -1, 0)]
        I_t_plus_1 = inputs[('color', 1, 0)]
        
        loss_dict = self.criterion(
            I_t=I_t,
            I_t_minus_1=I_t_minus_1,
            I_t_plus_1=I_t_plus_1,
            F_total_t_minus_1_to_t=stage2_outputs['F_total_t_minus_1'],
            F_total_t_plus_1_to_t=stage2_outputs['F_total_t_plus_1'],
            mask_t_minus_1=stage2_outputs['mask_t_minus_1'],
            mask_t_plus_1=stage2_outputs['mask_t_plus_1'],
            rendered_I_t=rendered_I_t,
            res_flow_net=self.stage2_modules['res_flow_net'],
            rigid_flow_fn=self.stage2_modules['rigid_flow_calc']
        )
        
        # 合并输出
        outputs.update(stage2_outputs)
        outputs['rendered_I_t'] = rendered_I_t
        
        return outputs, loss_dict
    
    def train_dataloader(self):
        """获取训练数据加载器"""
        return self._dataloaders['train']
    
    def val_dataloader(self):
        """获取验证数据加载器"""
        return self._dataloaders['val']
    
    def set_train(self):
        """设置训练模式"""
        self.train()
        # 冻结网络保持eval模式
        for model in self.models.values():
            model.eval()
    
    def set_val(self):
        """设置验证模式"""
        self.eval()
    
    def save_model(self, epoch):
        """保存模型 (仅保存ResFlowNet)"""
        save_dir = os.path.join(self.log_dir, 'models')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f'res_flow_net_epoch_{epoch}.pth')
        res_flow_net = self.stage2_modules['res_flow_net']
        if hasattr(res_flow_net, 'module'):
            state_dict = res_flow_net.module.state_dict()
        else:
            state_dict = res_flow_net.state_dict()
        torch.save(state_dict, save_path)
        print(f"Saved ResFlowNet to {save_path}")
    
    def load_res_flow_net(self, checkpoint_path):
        """加载ResFlowNet权重"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.rank}')

            # Support both pure state_dict and wrapped checkpoint dicts.
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                    checkpoint = checkpoint['state_dict']
                elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                    checkpoint = checkpoint['model']

            # Allow loading checkpoints saved from DDP ("module." prefix).
            if isinstance(checkpoint, dict):
                checkpoint = {
                    (k[7:] if k.startswith('module.') else k): v
                    for k, v in checkpoint.items()
                }

            res_flow_net = self.stage2_modules['res_flow_net']
            if hasattr(res_flow_net, 'module'):
                res_flow_net = res_flow_net.module
            res_flow_net.load_state_dict(checkpoint, strict=True)
            print(f"Loaded ResFlowNet from {checkpoint_path}")
        else:
            print(f"Warning: ResFlowNet checkpoint not found at {checkpoint_path}")


