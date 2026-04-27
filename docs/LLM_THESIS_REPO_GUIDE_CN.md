# 在线前馈4D动态场景解耦重建研究：论文写作与代码理解指南

> 面向对象：后续辅助撰写毕业论文的大语言模型，以及需要快速理解本仓库的读者。
> 建议先读本文，再按文中路径查看具体代码。

## 1. 研究主题概括

本仓库面向自动驾驶多相机在线前馈4D动态场景解耦重建任务，采用两阶段结构：

1. 第一阶段学习多相机静态场景的深度、位姿和 3D Gaussian 参数，建立稳定的基础重建结果。
2. 第二阶段在冻结第一阶段网络的基础上，引入残差光流网络 ResFlowNet 和动态融合机制，用于补偿动态物体、遮挡和几何误差造成的时序错位。

最终评价指标为：

| 指标 | 趋势 | 含义 |
|---|---|---|
| PSNR | 越高越好 | 像素级重建保真度 |
| SSIM | 越高越好 | 结构相似性 |
| LPIPS | 越低越好 | 感知差异，越低说明视觉上越接近 GT |

## 2. 模型主要结构与研究点

本模型可以概括为三个核心研究点：

1. 第一阶段跨视角特征融合。
   - 相关文件：`network/depth_network.py`、`models/gaussian/gaussian_network.py`
   - 作用：让六个环视相机之间交换上下文信息，提升多视角几何和外观一致性。

2. 第二阶段残差光流网络 ResFlowNet。
   - 相关文件：`stage2_modules/res_flow_net.py`
   - 作用：在刚性光流基础上预测残差光流，补偿动态物体或几何误差导致的错位。

3. 第二阶段动态融合机制。
   - 相关文件：`stage2_trainer/stage2_model.py`、`stage2_trainer/stage2_model_multi_mode.py`
   - 作用：根据残差光流幅值判断区域动态程度，在动态区域更依赖对齐更可靠的源帧，从而减轻重影。

## 3. 整体模型结构

### 3.1 Stage1：在线前馈基础重建

Stage1 的目标是从多相机图像中预测深度、相机位姿和 Gaussian 参数，完成基础的新视角重建。

主要链路：

```text
multi-camera images
    -> PoseNetwork
    -> DepthNetwork + cross-view fusion
    -> VFNet voxel fusion
    -> GaussianNetwork + cross-view fusion
    -> Gaussian rendering
    -> reconstructed image
```

核心代码：

| 文件 | 作用 |
|---|---|
| `train.py` | Stage1 单卡训练入口 |
| `train_multi_gpu.py` | Stage1 DDP 训练入口 |
| `eval.py` | Stage1 评估入口 |
| `models/drivingforward_model.py` | Stage1 单卡模型主类 |
| `models/drivingforward_model_ddp.py` | Stage1 DDP 模型主类 |
| `network/depth_network.py` | 深度网络，含 Swin encoder 与跨视角融合 |
| `network/pose_network.py` | 位姿网络 |
| `network/volumetric_fusionnet.py` | 体素空间融合网络 VFNet |
| `models/gaussian/gaussian_network.py` | Gaussian 参数预测网络 |
| `models/gaussian/GaussianRender.py` | Gaussian 渲染入口 |

### 3.2 Stage2：动态补偿与融合

Stage2 加载并冻结第一阶段权重，然后训练动态补偿模块。

主要链路：

```text
Stage1 depth + pose
    -> rigid flow F_rigid
    -> warp t-1 / t+1 source frames
    -> ResFlowNet predicts F_residual
    -> F_total = F_rigid + F_residual
    -> warp source frames again
    -> dynamic fusion
    -> final rendered image
```

核心代码：

| 文件 | 作用 |
|---|---|
| `train_stage2.py` | Stage2 单卡/兼容训练入口 |
| `train_stage2_ddp.py` | Stage2 DDP 训练和评估入口 |
| `stage2_trainer/model_factory.py` | 根据配置选择 Stage2 模型和 loss |
| `stage2_trainer/stage2_model.py` | 标准 Stage2 模型 |
| `stage2_trainer/stage2_model_multi_mode.py` | 支持 temporal / spatio / spatio-temporal 的 Stage2 模型 |
| `stage2_trainer/stage2_trainer_ddp.py` | Stage2 多卡训练器 |
| `stage2_modules/rigid_flow.py` | 刚性流计算与图像 warp |
| `stage2_modules/res_flow_net.py` | 残差光流网络 |
| `stage2_modules/stage2_loss_multi_mode.py` | Stage2 多模式损失 |

## 4. ResFlowNet 结构解释

ResFlowNet 的输入是 `warped_img`、`target_img` 和 `rigid_flow`。它输出 `F_residual`，表示刚性光流仍然无法解释的对齐误差。最终总光流为：

```text
F_total = F_rigid + F_residual
```

当前网络结构包括：

1. warped 图像与 target 图像的双流编码。
2. rigid flow 的几何先验编码。
3. 粗尺度 cross-attention，用目标特征查询 warped 特征。
4. U-Net 风格多尺度解码。
5. 全分辨率轻量迭代 refine。

论文中可表述为：ResFlowNet 学习的是刚性几何无法解释的残差运动，尤其用于动态物体、遮挡、深度误差或位姿误差导致的对齐偏差。

## 5. 动态融合机制解释

Stage2 会从 `t-1` 和 `t+1` 两个源帧重建当前帧 `t`。如果直接平均融合，动态物体在两侧源帧中的位置不同，容易产生重影。

本模型使用残差光流幅值作为动态程度和对齐可信度线索：

1. 残差光流越大，说明该区域越可能存在非刚性运动或对齐困难。
2. 动态 gate 越大，最终结果越偏向动态融合分支。
3. 在动态融合分支中，残差更小的一侧源帧被赋予更高权重。
4. 这样可以减少两个源帧错误平均造成的重影。

## 6. 损失函数

Stage2 损失主要包括：

| 损失 | 代码位置 | 作用 |
|---|---|---|
| warp loss | `stage2_modules/stage2_loss_multi_mode.py` | 约束 warp 后图像接近目标图像 |
| consistency loss | `stage2_modules/stage2_loss_multi_mode.py` | 约束正反向 flow 一致性 |
| render loss | `stage2_modules/stage2_loss_multi_mode.py` | 约束最终融合结果接近 GT |

总损失形式：

```text
L = lambda_warp * L_warp
  + lambda_consist * L_consist
  + lambda_render * L_render
```

## 7. 配置文件阅读指南

常用配置位于 `configs/nuscenes/`。

| 配置 | 作用 |
|---|---|
| `main_ddp_stage1_crossview_4gpu.yaml` | 带跨视角融合的 Stage1 多卡训练 |
| `main_ddp_stage1_nocrossview_4gpu.yaml` | 不带跨视角融合的 Stage1 消融训练 |
| `phase2_training_multi_mode_ddp_from_stage1_crossview_4gpu.yaml` | 使用 cross-view Stage1 权重训练完整 Stage2 |
| `phase2_ablation_full_ddp.yaml` | 消融中的完整模型配置 |
| `phase2_ablation_wo_dynamic_fusion_ddp.yaml` | 去掉二阶段动态融合 |
| `phase2_ablation_from_stage1_nocrossview_ddp.yaml` | 使用无跨视角融合的 Stage1 权重训练 Stage2 |

关键字段：

| 字段 | 含义 |
|---|---|
| `training.stage1_weights_path` | Stage2 加载的 Stage1 权重目录 |
| `model.enable_cross_view_fusion` | 是否启用 Stage1 跨视角融合 |
| `model.use_dynamic_fusion` | 是否启用 Stage2 动态融合 |
| `data.log_dir` | 实验输出目录 |
| `training.num_epochs` | 训练轮数 |

## 8. 消融实验设计

当前消融实验覆盖主要模块，适合论文使用。

| 实验名 | 设置 | 证明的问题 |
|---|---|---|
| Full | Stage1 cross-view + Stage2 ResFlowNet + dynamic fusion | 完整方法效果 |
| Stage1 only | 只使用第一阶段 | 第二阶段是否有效 |
| w/o Cross-view Fusion | Stage1 去掉跨视角融合，再训练 Stage2 | 跨视角融合是否有效 |
| w/o Dynamic Fusion | Stage2 保留 ResFlowNet，但关闭动态融合 | 动态融合策略是否有效 |

论文表格建议：

```text
Method                  PSNR ↑    SSIM ↑    LPIPS ↓
Stage1 only             ...
w/o Cross-view Fusion   ...
w/o Dynamic Fusion      ...
Full Model              ...
```

## 9. 推荐阅读顺序

如果让其他大模型辅助论文撰写，推荐按这个顺序阅读：

1. `docs/LLM_THESIS_REPO_GUIDE_CN.md`
2. `docs/RESFLOW_AND_LOSS_PLAIN_CN.md`
3. `docs/ABLATION_RUNBOOK_V1_CN.md`
4. `network/depth_network.py`
5. `models/gaussian/gaussian_network.py`
6. `stage2_modules/res_flow_net.py`
7. `stage2_trainer/stage2_model_multi_mode.py`
8. `stage2_modules/stage2_loss_multi_mode.py`

## 10. 论文写作建议

论文方法章节可以围绕以下逻辑展开：

1. 在线前馈4D场景重建总体框架。
2. 跨视角特征融合模块。
3. 基于刚性光流与残差光流的动态补偿。
4. 动态区域感知融合机制。
5. 损失函数与训练策略。

实验章节可以围绕以下逻辑展开：

1. 主实验结果。
2. 消融实验。
3. 动态物体重影可视化分析。
4. 局限性与失败案例分析。

## 11. GitHub 上传注意事项

上传 GitHub 时建议保留源码、配置、文档和少量示例图片。

不要上传：

- `results_*`
- `results_stage*`
- `results_ablation/`
- `weight/`
- `*.pth`
- `*.pt`
- `*.ckpt`
- `sh_log_err/`
- `__pycache__/`

这些已经在 `.gitignore` 中配置。

## 12. 一句话结论

本仓库的核心思想是：通过在线前馈的多相机基础重建获得稳定的场景表示，再通过残差光流和动态融合专门处理动态区域错位，从而提升自动驾驶动态场景重建质量，尤其改善动态物体重影现象。
