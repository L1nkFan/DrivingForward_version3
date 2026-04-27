"""
动态高斯生成模块 - 适配DrivingForward MF模式
复用原有高斯结构，仅更新3D均值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DynamicGaussianGenerator(nn.Module):
    """
    动态高斯生成模块
    将2D光流转换为3D位移，更新高斯的3D均值
    """
    
    def __init__(self, height=352, width=640):
        super(DynamicGaussianGenerator, self).__init__()
        self.height = height
        self.width = width
        
    def flow_to_3d_displacement(self, flow, depth_src, depth_tgt, K, T_src2tgt):
        """
        将2D光流转换为3D空间位移
        
        Args:
            flow: [B, 2, H, W] 2D光流 (dx, dy)
            depth_src: [B, 1, H, W] 源帧深度图
            depth_tgt: [B, 1, H, W] 目标帧深度图 (通过warp得到)
            K: [B, 3, 3] 相机内参
            T_src2tgt: [B, 4, 4] 位姿变换
            
        Returns:
            displacement: [B, 3, H*W] 3D位移向量
        """
        batch_size = flow.shape[0]
        device = flow.device
        
        # 计算内参逆矩阵
        K_inv = torch.inverse(K)  # [B, 3, 3]
        
        # 创建像素网格
        u = torch.arange(self.width, dtype=torch.float32, device=device)
        v = torch.arange(self.height, dtype=torch.float32, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')  # [H, W]
        
        # 源帧像素坐标
        u_src = uu.flatten().unsqueeze(0).repeat(batch_size, 1)  # [B, H*W]
        v_src = vv.flatten().unsqueeze(0).repeat(batch_size, 1)  # [B, H*W]
        
        # 目标帧像素坐标 (源坐标 + 光流)
        flow_flat = flow.view(batch_size, 2, -1)  # [B, 2, H*W]
        u_tgt = u_src + flow_flat[:, 0, :]  # [B, H*W]
        v_tgt = v_src + flow_flat[:, 1, :]  # [B, H*W]
        
        # 反投影源帧3D点
        depth_src_flat = depth_src.view(batch_size, 1, -1)  # [B, 1, H*W]
        pixels_src = torch.stack([u_src, v_src, torch.ones_like(u_src)], dim=1)  # [B, 3, H*W]
        points_src = torch.bmm(K_inv, pixels_src) * depth_src_flat  # [B, 3, H*W]
        
        # 反投影目标帧3D点
        depth_tgt_flat = depth_tgt.view(batch_size, 1, -1)  # [B, 1, H*W]
        pixels_tgt = torch.stack([u_tgt, v_tgt, torch.ones_like(u_tgt)], dim=1)  # [B, 3, H*W]
        points_tgt = torch.bmm(K_inv, pixels_tgt) * depth_tgt_flat  # [B, 3, H*W]
        
        # 计算3D位移
        displacement = points_tgt - points_src  # [B, 3, H*W]
        
        return displacement
    
    def update_gaussian_means(self, gaussian_xyz, displacement, valid_mask=None):
        """
        更新高斯的3D均值
        
        Args:
            gaussian_xyz: [B, N, 3] 或 [B, H*W, 3] 原始高斯3D位置
            displacement: [B, 3, H*W] 或 [B, 3, N] 3D位移
            valid_mask: [B, N] 或 [B, H*W] 有效掩码 (可选)
            
        Returns:
            updated_xyz: [B, N, 3] 更新后的高斯3D位置
        """
        # 确保维度匹配
        if displacement.dim() == 3 and displacement.shape[1] == 3:
            displacement = displacement.permute(0, 2, 1)  # [B, H*W, 3]
        
        # 更新位置
        updated_xyz = gaussian_xyz + displacement
        
        # 应用掩码
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(-1)  # [B, N, 1]
            updated_xyz = torch.where(valid_mask.bool(), updated_xyz, gaussian_xyz)
        
        return updated_xyz
    
    def forward_single_direction(self, 
                                  gaussian_data_src,
                                  flow_total,
                                  depth_src,
                                  K,
                                  T_src2tgt,
                                  valid_mask=None):
        """
        处理单个方向 (t-1 -> t 或 t+1 -> t)
        
        Args:
            gaussian_data_src: dict 包含源帧高斯数据
                - 'xyz': [B, H*W, 3] 3D位置
                - 'rot': [B, H*W, 4] 旋转四元数
                - 'scale': [B, H*W, 3] 尺度
                - 'opacity': [B, H*W, 1] 不透明度
                - 'sh': [B, H*W, 1, 1, 3, 25] 球谐系数
            flow_total: [B, 2, H, W] 总光流
            depth_src: [B, 1, H, W] 源帧深度
            K: [B, 3, 3] 相机内参
            T_src2tgt: [B, 4, 4] 位姿变换
            valid_mask: [B, 1, H, W] 有效掩码
            
        Returns:
            gaussian_data_tgt: dict 目标帧动态高斯数据
        """
        batch_size = flow_total.shape[0]
        
        # 通过光流warp深度图得到目标帧深度估计
        from .rigid_flow import warp_image_with_flow
        depth_tgt = warp_image_with_flow(depth_src, flow_total)
        
        # 计算3D位移
        displacement = self.flow_to_3d_displacement(
            flow_total, depth_src, depth_tgt, K, T_src2tgt
        )  # [B, 3, H*W]
        
        # 更新高斯位置
        xyz_src = gaussian_data_src['xyz']  # [B, H*W, 3]
        xyz_tgt = self.update_gaussian_means(xyz_src, displacement, valid_mask)
        
        # 构建目标帧高斯数据 (其他参数保持不变)
        gaussian_data_tgt = {
            'xyz': xyz_tgt,
            'rot': gaussian_data_src['rot'],
            'scale': gaussian_data_src['scale'],
            'opacity': gaussian_data_src['opacity'],
            'sh': gaussian_data_src['sh'],
        }
        
        return gaussian_data_tgt
    
    def combine_bidirectional_gaussians(self, 
                                        gaussian_from_prev,
                                        gaussian_from_next,
                                        weights=None):
        """
        聚合双向动态高斯 (MF模式)
        
        Args:
            gaussian_from_prev: dict 从t-1变换来的高斯
            gaussian_from_next: dict 从t+1变换来的高斯
            weights: [B, 2] 聚合权重 (可选，默认等权重)
            
        Returns:
            gaussian_combined: dict 聚合后的高斯
        """
        if weights is None:
            # 默认等权重聚合
            w1 = 0.5
            w2 = 0.5
        else:
            w1 = weights[:, 0:1]  # [B, 1]
            w2 = weights[:, 1:2]  # [B, 1]
        
        # 加权聚合位置
        xyz_combined = w1.unsqueeze(-1) * gaussian_from_prev['xyz'] + \
                       w2.unsqueeze(-1) * gaussian_from_next['xyz']
        
        # 对于其他属性，简单拼接 (渲染时会分别处理)
        # 或者可以选择置信度更高的那个
        gaussian_combined = {
            'xyz_prev': gaussian_from_prev['xyz'],
            'xyz_next': gaussian_from_next['xyz'],
            'xyz': xyz_combined,
            'rot_prev': gaussian_from_prev['rot'],
            'rot_next': gaussian_from_next['rot'],
            'scale_prev': gaussian_from_prev['scale'],
            'scale_next': gaussian_from_next['scale'],
            'opacity_prev': gaussian_from_prev['opacity'],
            'opacity_next': gaussian_from_next['opacity'],
            'sh_prev': gaussian_from_prev['sh'],
            'sh_next': gaussian_from_next['sh'],
        }
        
        return gaussian_combined
    
    def forward(self,
                gaussian_data_t_minus_1,
                gaussian_data_t_plus_1,
                flow_t_minus_1_to_t,
                flow_t_plus_1_to_t,
                depth_t_minus_1,
                depth_t_plus_1,
                K,
                T_t_minus_1_to_t,
                T_t_plus_1_to_t,
                valid_mask_t_minus_1=None,
                valid_mask_t_plus_1=None):
        """
        前向传播 - MF模式双向处理
        
        Args:
            gaussian_data_t_minus_1: dict t-1时刻的高斯数据
            gaussian_data_t_plus_1: dict t+1时刻的高斯数据
            flow_t_minus_1_to_t: [B, N, 2, H, W] 或 [B, 2, H, W] t-1到t的光流
            flow_t_plus_1_to_t: [B, N, 2, H, W] 或 [B, 2, H, W] t+1到t的光流
            depth_t_minus_1: [B, N, 1, H, W] 或 [B, 1, H, W] t-1深度
            depth_t_plus_1: [B, N, 1, H, W] 或 [B, 1, H, W] t+1深度
            K: [B, N, 3, 3] 或 [B, 3, 3] 相机内参
            T_t_minus_1_to_t: [B, N, 4, 4] 或 [B, 4, 4] t-1到t的位姿
            T_t_plus_1_to_t: [B, N, 4, 4] 或 [B, 4, 4] t+1到t的位姿
            valid_mask_t_minus_1: [B, N, 1, H, W] 或 [B, 1, H, W] (可选)
            valid_mask_t_plus_1: [B, N, 1, H, W] 或 [B, 1, H, W] (可选)
            
        Returns:
            gaussian_combined: dict 聚合后的动态高斯
        """
        # 处理多相机输入
        if flow_t_minus_1_to_t.dim() == 5:
            batch_size, num_cams = flow_t_minus_1_to_t.shape[:2]
            
            gaussian_combined_list = []
            for cam_idx in range(num_cams):
                # 提取当前相机的数据
                flow_prev = flow_t_minus_1_to_t[:, cam_idx]  # [B, 2, H, W]
                flow_next = flow_t_plus_1_to_t[:, cam_idx]  # [B, 2, H, W]
                depth_prev = depth_t_minus_1[:, cam_idx]  # [B, 1, H, W]
                depth_next = depth_t_plus_1[:, cam_idx]  # [B, 1, H, W]
                K_cam = K[:, cam_idx] if K.dim() == 4 else K  # [B, 3, 3]
                T_prev = T_t_minus_1_to_t[:, cam_idx] if T_t_minus_1_to_t.dim() == 4 else T_t_minus_1_to_t
                T_next = T_t_plus_1_to_t[:, cam_idx] if T_t_plus_1_to_t.dim() == 4 else T_t_plus_1_to_t
                mask_prev = valid_mask_t_minus_1[:, cam_idx] if valid_mask_t_minus_1 is not None else None
                mask_next = valid_mask_t_plus_1[:, cam_idx] if valid_mask_t_plus_1 is not None else None
                
                # 提取高斯数据 (假设每个相机有独立的高斯)
                gauss_prev = {k: v[:, cam_idx] if v.dim() > 2 else v for k, v in gaussian_data_t_minus_1.items()}
                gauss_next = {k: v[:, cam_idx] if v.dim() > 2 else v for k, v in gaussian_data_t_plus_1.items()}
                
                # 处理每个方向
                gauss_from_prev = self.forward_single_direction(
                    gauss_prev, flow_prev, depth_prev, K_cam, T_prev, mask_prev
                )
                gauss_from_next = self.forward_single_direction(
                    gauss_next, flow_next, depth_next, K_cam, T_next, mask_next
                )
                
                # 聚合
                gauss_combined = self.combine_bidirectional_gaussians(gauss_from_prev, gauss_from_next)
                gaussian_combined_list.append(gauss_combined)
            
            # 合并所有相机的结果
            gaussian_combined = {}
            for key in gaussian_combined_list[0].keys():
                gaussian_combined[key] = torch.stack([g[key] for g in gaussian_combined_list], dim=1)
        else:
            # 单相机处理
            gauss_from_prev = self.forward_single_direction(
                gaussian_data_t_minus_1, flow_t_minus_1_to_t, depth_t_minus_1,
                K, T_t_minus_1_to_t, valid_mask_t_minus_1
            )
            gauss_from_next = self.forward_single_direction(
                gaussian_data_t_plus_1, flow_t_plus_1_to_t, depth_t_plus_1,
                K, T_t_plus_1_to_t, valid_mask_t_plus_1
            )
            gaussian_combined = self.combine_bidirectional_gaussians(gauss_from_prev, gauss_from_next)
        
        return gaussian_combined


def prepare_gaussian_data_from_outputs(outputs, cam, frame_id=0):
    """
    从DrivingForward的outputs中提取高斯数据
    
    Args:
        outputs: DrivingForward模型的输出字典
        cam: 相机索引
        frame_id: 帧ID (0, -1, 1)
        
    Returns:
        gaussian_data: dict 包含高斯参数
    """
    cam_output = outputs[('cam', cam)]
    
    # 提取高斯参数
    gaussian_data = {
        'xyz': cam_output[('xyz', frame_id, 0)],  # [B, H*W, 3]
        'rot': cam_output[('rot_maps', frame_id, 0)],  # [B, 4, H, W]
        'scale': cam_output[('scale_maps', frame_id, 0)],  # [B, 3, H, W]
        'opacity': cam_output[('opacity_maps', frame_id, 0)],  # [B, 1, H, W]
        'sh': cam_output[('sh_maps', frame_id, 0)],  # [B, H*W, 1, 1, 3, 25]
    }
    
    # 将图像格式的高斯参数展平
    batch_size = gaussian_data['xyz'].shape[0]
    
    if gaussian_data['rot'].dim() == 4:  # [B, 4, H, W]
        _, C, H, W = gaussian_data['rot'].shape
        gaussian_data['rot'] = gaussian_data['rot'].permute(0, 2, 3, 1).reshape(batch_size, H * W, C)
    
    if gaussian_data['scale'].dim() == 4:  # [B, 3, H, W]
        _, C, H, W = gaussian_data['scale'].shape
        gaussian_data['scale'] = gaussian_data['scale'].permute(0, 2, 3, 1).reshape(batch_size, H * W, C)
    
    if gaussian_data['opacity'].dim() == 4:  # [B, 1, H, W]
        _, C, H, W = gaussian_data['opacity'].shape
        gaussian_data['opacity'] = gaussian_data['opacity'].permute(0, 2, 3, 1).reshape(batch_size, H * W, C)
    
    return gaussian_data
