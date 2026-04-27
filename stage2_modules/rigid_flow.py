"""
刚性流计算模块 - 适配DrivingForward MF模式
复用PoseNet输出的位姿变换，计算像素级刚性光流
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RigidFlowCalculator(nn.Module):
    """
    刚性流计算模块
    输入: 源帧深度图、相机内参、位姿变换矩阵
    输出: 刚性光流 F_rigid [B, 2, H, W] 和有效掩码
    """
    
    def __init__(self, height=352, width=640):
        super(RigidFlowCalculator, self).__init__()
        self.height = height
        self.width = width
        
    def create_pixel_grid(self, batch_size, device):
        """
        创建像素坐标网格
        返回: [B, 3, H*W] 的齐次坐标 (u, v, 1)
        """
        # 创建像素网格
        u = torch.arange(self.width, dtype=torch.float32, device=device)
        v = torch.arange(self.height, dtype=torch.float32, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')  # [H, W]
        
        # 展平并转换为齐次坐标
        pixels = torch.stack([uu.flatten(), vv.flatten(), torch.ones_like(uu.flatten())], dim=0)  # [3, H*W]
        pixels = pixels.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, H*W]
        
        return pixels
    
    # def backproject_to_3d(self, depth, K_inv):
    #     """
    #     将像素坐标反投影到3D相机坐标系
        
    #     Args:
    #         depth: [B, 1, H, W] 深度图
    #         K_inv: [B, 3, 3] 相机内参逆矩阵
            
    #     Returns:
    #         points_3d: [B, 3, H*W] 3D点坐标
    #     """
    #     batch_size = depth.shape[0]
    #     device = depth.device
        
    #     # 创建像素网格
    #     pixels = self.create_pixel_grid(batch_size, device)  # [B, 3, H*W]
        
    #     # 将深度图展平
    #     depth_flat = depth.view(batch_size, 1, -1)  # [B, 1, H*W]
        
    #     # 反投影: P_cam = K_inv @ (u, v, 1) * depth
    #     points_cam = torch.bmm(K_inv, pixels)  # [B, 3, H*W]
    #     points_cam = points_cam * depth_flat  # [B, 3, H*W]
        
    #     return points_cam
    
    def backproject_to_3d(self, depth, K_inv):
        """
        将像素坐标反投影到3D相机坐标系
        
        Args:
            depth: [B, 1, H, W] 深度图
            K_inv: [B, 3, 3] 相机内参逆矩阵
            
        Returns:
            points_3d: [B, 3, H*W] 3D点坐标
        """
        batch_size = depth.shape[0]
        device = depth.device
        # 创建像素网格 (u, v, 1)
        pixels = self.create_pixel_grid(batch_size, device)  # [B, 3, H*W]
        # 扩展为4D齐次坐标以匹配4x4的K_inv, 变为 (u, v, 1, 1)
        ones = torch.ones_like(pixels[:, :1, :])
        pixels_homo = torch.cat([pixels, ones], dim=1) # [B, 4, H*W]
        # 将深度图展平
        depth_flat = depth.view(batch_size, 1, -1)  # [B, 1, H*W]
        # 反投影: P_cam = (K_inv @ p_homo)[:3] * depth
        points_homo_proj = torch.bmm(K_inv, pixels_homo)  # [B, 4, H*W]
        points_cam = points_homo_proj[:, :3, :] * depth_flat  # [B, 3, H*W]
        return points_cam

    def transform_points(self, points_cam_src, T_src2tgt):
        """
        将源帧相机坐标系的3D点变换到目标帧相机坐标系
        
        Args:
            points_cam_src: [B, 3, H*W] 源帧3D点
            T_src2tgt: [B, 4, 4] 位姿变换矩阵 (源帧到目标帧)
            
        Returns:
            points_cam_tgt: [B, 3, H*W] 目标帧3D点
        """
        batch_size = points_cam_src.shape[0]
        
        # 转换为齐次坐标
        ones = torch.ones(batch_size, 1, points_cam_src.shape[2], device=points_cam_src.device)
        points_homo = torch.cat([points_cam_src, ones], dim=1)  # [B, 4, H*W]
        
        # 应用位姿变换
        points_tgt_homo = torch.bmm(T_src2tgt, points_homo)  # [B, 4, H*W]
        
        # 转回非齐次坐标
        points_cam_tgt = points_tgt_homo[:, :3, :]  # [B, 3, H*W]
        
        return points_cam_tgt
    
    # def project_to_2d(self, points_cam, K):
    #     """
    #     将3D点投影到2D图像平面
        
    #     Args:
    #         points_cam: [B, 3, H*W] 相机坐标系3D点
    #         K: [B, 3, 3] 相机内参
            
    #     Returns:
    #         pixels_tgt: [B, 2, H*W] 目标像素坐标 (u, v)
    #     """
    #     # 投影: p = K @ P_cam
    #     pixels_homo = torch.bmm(K, points_cam)  # [B, 3, H*W]
        
    #     # 归一化
    #     z = pixels_homo[:, 2:3, :] + 1e-7  # [B, 1, H*W]
    #     pixels_2d = pixels_homo[:, :2, :] / z  # [B, 2, H*W]
        
    #     return pixels_2d
    
    def project_to_2d(self, points_cam, K):
        """
        将3D点投影到2D像素平面
        
        Args:
            points_cam: [B, 3, H*W] 3D点坐标
            K: [B, 4, 4] 相机内参矩阵
            
        Returns:
            pixels: [B, 2, H*W] 2D像素坐标
        """
        # 将3D点转换为4D齐次坐标
        B, _, N = points_cam.shape
        points_homo = torch.cat([
            points_cam,
            torch.ones(B, 1, N, device=points_cam.device)
        ], dim=1)  # [B, 4, H*W]

        # 投影: p_proj = K @ P_cam_homo
        pixels_projected = torch.bmm(K, points_homo)  # [B, 4, H*W]
        
        # 归一化: (u, v, 1) = (x/z, y/z, 1)
        # 避免除以零
        depth = pixels_projected[:, 2:3, :]
        depth[depth < 1e-8] = 1e-8
        
        pixels_normalized = pixels_projected[:, :2, :] / depth  # [B, 2, H*W]
        
        return pixels_normalized

    def compute_rigid_flow(self, depth_src, K, T_src2tgt, depth_mask=None):
        """
        计算刚性光流
        
        Args:
            depth_src: [B, 1, H, W] 源帧深度图
            K: [B, 3, 3] 相机内参
            T_src2tgt: [B, 4, 4] 位姿变换矩阵
            depth_mask: [B, 1, H, W] 深度有效掩码 (可选)
            
        Returns:
            F_rigid: [B, 2, H, W] 刚性光流 (dx, dy)
            flow_mask: [B, 1, H, W] 有效流掩码
        """
        batch_size = depth_src.shape[0]
        device = depth_src.device
        
        # 计算内参逆矩阵
        K_inv = torch.inverse(K)  # [B, 3, 3]
        
        # 1. 反投影到3D
        points_cam_src = self.backproject_to_3d(depth_src, K_inv)  # [B, 3, H*W]
        
        # 2. 位姿变换
        points_cam_tgt = self.transform_points(points_cam_src, T_src2tgt)  # [B, 3, H*W]
        
        # 3. 投影到2D
        pixels_tgt = self.project_to_2d(points_cam_tgt, K)  # [B, 2, H*W]
        
        # 4. 计算光流
        # 创建源像素坐标
        u_src = torch.arange(self.width, dtype=torch.float32, device=device)
        v_src = torch.arange(self.height, dtype=torch.float32, device=device)
        uu_src, vv_src = torch.meshgrid(u_src, v_src, indexing='xy')
        pixels_src = torch.stack([uu_src.flatten(), vv_src.flatten()], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 计算位移
        flow = pixels_tgt - pixels_src  # [B, 2, H*W]
        
        # 重塑为图像格式
        F_rigid = flow.view(batch_size, 2, self.height, self.width)  # [B, 2, H, W]
        
        # 5. 生成有效掩码
        # 检查像素是否在图像边界内
        u_tgt = pixels_tgt[:, 0, :]  # [B, H*W]
        v_tgt = pixels_tgt[:, 1, :]  # [B, H*W]
        
        valid_u = (u_tgt >= 0) & (u_tgt < self.width)
        valid_v = (v_tgt >= 0) & (v_tgt < self.height)
        valid_mask = (valid_u & valid_v).float()  # [B, H*W]
        
        # 检查深度有效性 (Z > 0)
        z_tgt = points_cam_tgt[:, 2, :]  # [B, H*W]
        valid_depth = (z_tgt > 0.01).float()
        
        # 合并掩码
        flow_mask = valid_mask * valid_depth  # [B, H*W]
        
        # 如果提供了深度掩码，也考虑进去
        if depth_mask is not None:
            depth_mask_flat = depth_mask.view(batch_size, -1)
            flow_mask = flow_mask * depth_mask_flat
        
        flow_mask = flow_mask.view(batch_size, 1, self.height, self.width)
        
        return F_rigid, flow_mask
    
    def forward(self, depth_src, K, T_src2tgt, depth_mask=None):
        """
        前向传播
        
        Args:
            depth_src: [B, 1, H, W] 或 [B, N, 1, H, W]
            K: [B, 3, 3] 或 [B, N, 3, 3]
            T_src2tgt: [B, 4, 4] 或 [B, N, 4, 4]
            depth_mask: [B, 1, H, W] 或 [B, N, 1, H, W] (可选)
            
        Returns:
            F_rigid: 刚性光流
            flow_mask: 有效流掩码
        """
        # 处理多相机输入 [B, N, C, H, W]
        if depth_src.dim() == 5:
            batch_size, num_cams = depth_src.shape[:2]
            F_rigid_list = []
            mask_list = []
            
            for cam_idx in range(num_cams):
                depth_cam = depth_src[:, cam_idx]  # [B, 1, H, W]
                K_cam = K[:, cam_idx] if K.dim() == 4 else K  # [B, 3, 3]
                T_cam = T_src2tgt[:, cam_idx] if T_src2tgt.dim() == 4 else T_src2tgt  # [B, 4, 4]
                mask_cam = depth_mask[:, cam_idx] if depth_mask is not None else None
                
                F_rigid_cam, mask_cam_out = self.compute_rigid_flow(depth_cam, K_cam, T_cam, mask_cam)
                F_rigid_list.append(F_rigid_cam)
                mask_list.append(mask_cam_out)
            
            F_rigid = torch.stack(F_rigid_list, dim=1)  # [B, N, 2, H, W]
            flow_mask = torch.stack(mask_list, dim=1)  # [B, N, 1, H, W]
        else:
            F_rigid, flow_mask = self.compute_rigid_flow(depth_src, K, T_src2tgt, depth_mask)
        
        return F_rigid, flow_mask


def warp_image_with_flow(img, flow):
    """
    使用光流对图像进行warp
    
    Args:
        img: [B, C, H, W] 源图像
        flow: [B, 2, H, W] 光流 (dx, dy)
        
    Returns:
        warped_img: [B, C, H, W] warp后的图像
    """
    batch_size, _, height, width = img.shape
    device = img.device
    
    # 创建归一化网格 [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # 将光流转换为归一化坐标偏移
    flow_norm = flow.clone()
    flow_norm[:, 0, :, :] = flow[:, 0, :, :] / (width - 1) * 2  # x方向
    flow_norm[:, 1, :, :] = flow[:, 1, :, :] / (height - 1) * 2  # y方向
    
    # 转置为 [B, H, W, 2]
    flow_norm = flow_norm.permute(0, 2, 3, 1)
    
    # 计算采样网格
    sample_grid = base_grid + flow_norm
    
    # 使用grid_sample进行warp
    warped_img = F.grid_sample(
        img, 
        sample_grid, 
        mode='bilinear', 
        padding_mode='zeros',
        align_corners=True
    )
    
    return warped_img


def batch_warp_image_with_flow(img, flow):
    """
    批量处理多相机的图像warp
    
    Args:
        img: [B, N, C, H, W] 源图像
        flow: [B, N, 2, H, W] 光流
        
    Returns:
        warped_img: [B, N, C, H, W] warp后的图像
    """
    batch_size, num_cams, channels, height, width = img.shape
    
    warped_list = []
    for cam_idx in range(num_cams):
        img_cam = img[:, cam_idx]  # [B, C, H, W]
        flow_cam = flow[:, cam_idx]  # [B, 2, H, W]
        warped_cam = warp_image_with_flow(img_cam, flow_cam)
        warped_list.append(warped_cam)
    
    warped_img = torch.stack(warped_list, dim=1)  # [B, N, C, H, W]
    
    return warped_img
