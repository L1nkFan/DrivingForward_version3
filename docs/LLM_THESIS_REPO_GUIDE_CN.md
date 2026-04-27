# DrivingForward_version3 论文写作与代码理解指南

> 面向对象：后续辅助撰写毕业论文的大语言模型，以及需要快速理解本仓库的读者。
> 建议先读本文，再按文中路径查看具体代码。

## 1. 仓库一句话概括

本仓库基于原始 DrivingForward，面向自动驾驶多相机场景重建任务，扩展为两阶段方法：

1. 第一阶段学习静态场景的深度、位姿和 3D Gaussian 参数。
2. 第二阶段在冻结第一阶段的基础上，引入残差光流网络 ResFlowNet 和动态融合机制，用于缓解动态物体重影问题。

最终评价指标为：

| 指标 | 趋势 | 含义 |
|---|---|---|
| PSNR | 越高越好 | 像素级重建保真度 |
| SSIM | 越高越好 | 结构相似性 |
| LPIPS | 越低越好 | 感知差异，越低说明视觉上越接近 GT |

## 2. 相比原始 DrivingForward 的主要改造

本版本可以概括为三个核心创新点：

1. 第一阶段加入跨视角融合。
   - 相关文件：`network/depth_network.py`、`models/gaussian/gaussian_network.py`
   - 作用：让六个环视相机之间交换特征信息，提升多视角几何与外观一致性。

2. 第二阶段加入 ResFlowNet。
   - 相关文件：`stage2_modules/res_flow_net.py`
   - 作用：在刚性光流基础上预测残差光流，补偿动态物体或几何误差导致的错位。

3. 第二阶段加入动态融合机制。
   - 相关文件：`stage2_trainer/stage2_model.py`、`stage2_trainer/stage2_model_multi_mode.py`
   - 作用：根据残差光流幅值判断区域动态程度，在动态区域更依赖可信源帧，从而减轻重影。

## 3. 整体模型结构

## 3.1 Stage1：基础 3D Gaussian 重建

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

## 3.2 Stage2：动态补偿与融合

Stage2 不重新训练第一阶段网络，而是加载第一阶段权重并冻结，然后训练 ResFlowNet。

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

ResFlowNet 的输入是：

```text
warped_img, target_img, rigid_flow
```

其中：

- `warped_img`：用刚性光流从源帧 warp 到目标帧后的图像。
- `target_img`：目标帧真实图像。
- `rigid_flow`：由深度和相机运动得到的刚性流。

ResFlowNet 输出：

```text
F_residual
```

它表示刚性流仍然没有对齐的那部分误差。最终使用：

```text
F_total = F_rigid + F_residual
```

当前网络结构可以概括为：

1. 图像双流编码：分别编码 warped 图像和 target 图像。
2. 刚性流编码：把 rigid flow 作为几何先验输入网络。
3. 粗尺度 cross-attention：让目标图像特征主动查询 warped 图像特征。
4. U-Net 式解码：逐级恢复分辨率，输出残差光流。
5. 迭代 refine：在全分辨率下进一步修正 flow。

写论文时可以这样描述：

> 为了补偿纯刚性几何假设在动态物体区域的对齐误差，我们设计了一个残差光流网络 ResFlowNet。该网络以刚性 warp 后的源图像、目标图像以及刚性光流为输入，通过双流图像编码、刚性流先验编码和粗尺度 cross-attention 融合，预测残差光流。最终总光流由刚性光流与残差光流相加得到，从而提升动态区域的时序对齐质量。

## 5. 动态融合机制解释

在 Stage2 中，模型会从 `t-1` 和 `t+1` 两个源帧重建当前帧 `t`。

基础融合方式：

```text
rendered_base = mask-average(warped_t_minus_1, warped_t_plus_1)
```

动态融合方式：

1. 计算两侧残差光流幅值。
2. 残差越大，说明该区域越可能存在动态运动或对齐困难。
3. 用残差幅值生成 `dyn_gate`。
4. 动态区域中，优先选择残差更小、对齐更可靠的一侧源帧。
5. 最终输出为：

```text
final = rendered_base * (1 - dyn_gate) + rendered_dynamic * dyn_gate
```

这不是单独的后处理脚本，而是在模型前向传播中的融合路径。只要评估时使用对应配置和 Stage2 模型，就会生效。

## 6. 损失函数

Stage2 损失主要由三类组成：

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

其中 LPIPS 只在配置开启时参与部分 render/warp 项，代码中加入了 NaN/Inf 防护。

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

当前消融实验已经覆盖主要模块，适合论文使用。

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

定性图建议：

```text
GT | Stage1 only | w/o Dynamic Fusion | Full
```

建议优先选择动态行人、车辆、遮挡明显的场景，因为这些场景最能体现 Stage2 的意义。

## 9. 推荐给大语言模型的阅读顺序

如果让其他大模型辅助论文撰写，推荐按这个顺序阅读：

1. `docs/LLM_THESIS_REPO_GUIDE_CN.md`
2. `docs/DRIVINGFORWARD_V3_VS_ORIGINAL_CN.md`
3. `docs/RESFLOW_AND_LOSS_PLAIN_CN.md`
4. `docs/ABLATION_RUNBOOK_V1_CN.md`
5. `network/depth_network.py`
6. `models/gaussian/gaussian_network.py`
7. `stage2_modules/res_flow_net.py`
8. `stage2_trainer/stage2_model_multi_mode.py`
9. `stage2_modules/stage2_loss_multi_mode.py`

## 10. 论文写作时的建议表述

方法章节可以分成三段：

1. 基础重建阶段。
   说明本方法继承 DrivingForward 的 feed-forward Gaussian Splatting 框架，通过深度、位姿和 Gaussian 参数预测完成多视角场景重建。

2. 跨视角特征增强。
   说明在 DepthNetwork 和 GaussianNetwork 中引入跨相机特征交互，使环视相机之间共享上下文，提高几何和外观一致性。

3. 动态补偿阶段。
   说明第二阶段冻结 Stage1，利用刚性光流作为几何先验，再由 ResFlowNet 预测残差光流，并通过动态融合机制减轻动态物体重影。

实验章节可以分成三类：

1. 主结果：Full Model 与基线对比。
2. 消融实验：Stage1 only、w/o Cross-view Fusion、w/o Dynamic Fusion。
3. 可视化分析：动态物体区域的重影改善。

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

本仓库的核心思想是：先用 Stage1 建立稳定的多视角 Gaussian 重建基础，再用 Stage2 的残差光流和动态融合专门处理动态区域错位，从而在 PSNR、SSIM 和 LPIPS 上提升重建质量，尤其改善动态物体重影现象。

