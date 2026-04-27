"""
第二阶段代码验证脚本
检查所有模块是否正确导入和初始化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def verify_imports():
    """验证所有模块导入"""
    print("=" * 80)
    print("Verifying Stage2 Module Imports")
    print("=" * 80)
    
    try:
        from stage2_modules import RigidFlowCalculator
        print("✓ RigidFlowCalculator imported successfully")
    except Exception as e:
        print(f"✗ RigidFlowCalculator import failed: {e}")
        return False
    
    try:
        from stage2_modules import ResFlowNet, SharedEncoder, CameraDecoder
        print("✓ ResFlowNet imported successfully")
    except Exception as e:
        print(f"✗ ResFlowNet import failed: {e}")
        return False
    
    try:
        from stage2_modules import DynamicGaussianGenerator
        print("✓ DynamicGaussianGenerator imported successfully")
    except Exception as e:
        print(f"✗ DynamicGaussianGenerator import failed: {e}")
        return False
    
    try:
        from stage2_modules import Stage2Loss
        print("✓ Stage2Loss imported successfully")
    except Exception as e:
        print(f"✗ Stage2Loss import failed: {e}")
        return False
    
    try:
        from stage2_modules.rigid_flow import warp_image_with_flow, batch_warp_image_with_flow
        print("✓ rigid_flow functions imported successfully")
    except Exception as e:
        print(f"✗ rigid_flow functions import failed: {e}")
        return False
    
    return True


def verify_module_initialization():
    """验证模块初始化"""
    print("\n" + "=" * 80)
    print("Verifying Stage2 Module Initialization")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 1
    num_cams = 6
    height = 352
    width = 640
    
    # 1. 验证 RigidFlowCalculator
    try:
        from stage2_modules import RigidFlowCalculator
        rigid_flow_calc = RigidFlowCalculator(height=height, width=width).to(device)
        
        # 测试前向传播
        depth = torch.randn(batch_size, 1, height, width).to(device)
        K = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        T = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        F_rigid, mask = rigid_flow_calc(depth, K, T)
        assert F_rigid.shape == (batch_size, 2, height, width)
        assert mask.shape == (batch_size, 1, height, width)
        print(f"✓ RigidFlowCalculator initialized and tested (output shape: {F_rigid.shape})")
    except Exception as e:
        print(f"✗ RigidFlowCalculator initialization failed: {e}")
        return False
    
    # 2. 验证 ResFlowNet
    try:
        from stage2_modules import ResFlowNet
        res_flow_net = ResFlowNet(num_cams=num_cams, base_channels=64).to(device)
        
        # 测试前向传播
        warped_img = torch.randn(batch_size, num_cams, 3, height, width).to(device)
        tgt_img = torch.randn(batch_size, num_cams, 3, height, width).to(device)
        rigid_flow = torch.randn(batch_size, num_cams, 2, height, width).to(device)
        
        F_residual = res_flow_net(warped_img, tgt_img, rigid_flow)
        assert F_residual.shape == (batch_size, num_cams, 2, height, width)
        print(f"✓ ResFlowNet initialized and tested (output shape: {F_residual.shape})")
        
        # 计算参数量
        total_params = sum(p.numel() for p in res_flow_net.parameters())
        print(f"  Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"✗ ResFlowNet initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 验证 DynamicGaussianGenerator
    try:
        from stage2_modules import DynamicGaussianGenerator
        dynamic_gaussian = DynamicGaussianGenerator(height=height, width=width).to(device)
        print("✓ DynamicGaussianGenerator initialized")
    except Exception as e:
        print(f"✗ DynamicGaussianGenerator initialization failed: {e}")
        return False
    
    # 4. 验证 Stage2Loss
    try:
        from stage2_modules import Stage2Loss
        criterion = Stage2Loss(lambda_warp=0.02, lambda_consist=1e-5, lambda_render=0.01, rank=0)
        print("✓ Stage2Loss initialized")
    except Exception as e:
        print(f"✗ Stage2Loss initialization failed: {e}")
        return False
    
    return True


def verify_data_flow():
    """验证数据流"""
    print("\n" + "=" * 80)
    print("Verifying Stage2 Data Flow")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 1
    num_cams = 6
    height = 352
    width = 640
    
    try:
        # 模拟输入数据
        I_t = torch.randn(batch_size, num_cams, 3, height, width).to(device)
        I_t_minus_1 = torch.randn(batch_size, num_cams, 3, height, width).to(device)
        I_t_plus_1 = torch.randn(batch_size, num_cams, 3, height, width).to(device)
        
        depth = torch.randn(batch_size, num_cams, 1, height, width).to(device)
        K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1).to(device)
        T = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1).to(device)
        
        # 1. 刚性流计算
        from stage2_modules import RigidFlowCalculator
        rigid_flow_calc = RigidFlowCalculator(height=height, width=width).to(device)
        
        F_rigid_t_minus_1_list = []
        F_rigid_t_plus_1_list = []
        for cam in range(num_cams):
            F_rigid, _ = rigid_flow_calc(depth[:, cam], K[:, cam], T[:, cam])
            F_rigid_t_minus_1_list.append(F_rigid)
            F_rigid_t_plus_1_list.append(F_rigid)
        
        F_rigid_t_minus_1 = torch.stack(F_rigid_t_minus_1_list, dim=1)
        F_rigid_t_plus_1 = torch.stack(F_rigid_t_plus_1_list, dim=1)
        
        print(f"✓ Rigid flow computed (shape: {F_rigid_t_minus_1.shape})")
        
        # 2. 图像warp
        from stage2_modules.rigid_flow import batch_warp_image_with_flow
        warped_t_minus_1 = batch_warp_image_with_flow(I_t_minus_1, F_rigid_t_minus_1)
        warped_t_plus_1 = batch_warp_image_with_flow(I_t_plus_1, F_rigid_t_plus_1)
        
        print(f"✓ Image warping completed (shape: {warped_t_minus_1.shape})")
        
        # 3. 残差流预测
        from stage2_modules import ResFlowNet
        res_flow_net = ResFlowNet(num_cams=num_cams, base_channels=64).to(device)
        
        F_residual_t_minus_1 = res_flow_net(warped_t_minus_1, I_t, F_rigid_t_minus_1)
        F_residual_t_plus_1 = res_flow_net(warped_t_plus_1, I_t, F_rigid_t_plus_1)
        
        print(f"✓ Residual flow predicted (shape: {F_residual_t_minus_1.shape})")
        
        # 4. 总光流
        F_total_t_minus_1 = F_rigid_t_minus_1 + F_residual_t_minus_1
        F_total_t_plus_1 = F_rigid_t_plus_1 + F_residual_t_plus_1
        
        print(f"✓ Total flow computed (shape: {F_total_t_minus_1.shape})")
        
        print("\n✓ All data flow checks passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data flow verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Stage2 Code Verification")
    print("=" * 80)
    
    # 1. 验证导入
    if not verify_imports():
        print("\n✗ Import verification failed!")
        return False
    
    # 2. 验证模块初始化
    if not verify_module_initialization():
        print("\n✗ Module initialization verification failed!")
        return False
    
    # 3. 验证数据流
    if not verify_data_flow():
        print("\n✗ Data flow verification failed!")
        return False
    
    print("\n" + "=" * 80)
    print("✓ All verifications passed!")
    print("=" * 80)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
