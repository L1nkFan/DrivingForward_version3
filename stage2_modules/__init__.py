"""
Stage2 Modules Package
提供第二阶段训练的核心模块
"""

from .rigid_flow import RigidFlowCalculator, warp_image_with_flow, batch_warp_image_with_flow
from .res_flow_net import ResFlowNet, SharedEncoder, CameraDecoder
from .dynamic_gaussian import DynamicGaussianGenerator
from .stage2_loss import Stage2Loss, build_stage2_loss
from .stage2_loss_multi_mode import Stage2LossMultiMode, build_stage2_loss_multi_mode

__all__ = [
    # Rigid Flow
    'RigidFlowCalculator',
    'warp_image_with_flow',
    'batch_warp_image_with_flow',
    # Residual Flow Network
    'ResFlowNet',
    'SharedEncoder',
    'CameraDecoder',
    # Dynamic Gaussian
    'DynamicGaussianGenerator',
    # Loss Functions
    'Stage2Loss',
    'Stage2LossMultiMode',
    'build_stage2_loss',
    'build_stage2_loss_multi_mode',
     ]
