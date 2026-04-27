# DrivingForward_version3 仓库架构说明（论文/复现版）

> 文档目标：用于毕业论文撰写与开源协作，帮助读者（人类或 AI）快速理解本仓库的代码结构、两阶段方法链路、以及关键文件职责。

## 1. 项目定位

本仓库在原始 DrivingForward（AAAI 2025）基础上，扩展为**两阶段训练框架**：

- Stage 1：静态几何与高斯基础重建（Pose/Depth/Gaussian）。
- Stage 2：引入残差光流与动态融合，重点缓解动态目标重影。

目标指标主要关注：`PSNR`、`SSIM`、`LPIPS`。

## 2. 顶层目录与职责

| 路径 | 作用 |
|---|---|
| `configs/` | 全部训练/评估配置（Stage1、Stage2、DDP、消融） |
| `dataset/` | nuScenes 数据读取、样本组织、相机/时序数据构建 |
| `models/` | Stage1 主模型、几何模块、Gaussian 渲染与损失 |
| `network/` | Stage1 核心网络：`DepthNetwork`、`PoseNetwork`、`VFNet` |
| `trainer/` | Stage1 训练器（单卡/DDP） |
| `stage2_modules/` | Stage2 可训练模块：`ResFlowNet`、`RigidFlow`、Stage2 loss |
| `stage2_trainer/` | Stage2 模型封装与训练器（含多模式和 DDP） |
| `tools/` | 导出可视化、误差分析等辅助脚本 |
| `docs/` | 本地实验文档、优化报告、消融说明 |
| `external/` | 第三方依赖代码（packnet_sfm / dgp 等） |
| `assets/` | README 图示资源 |
| `weight/` | 本地权重目录（通常不建议上传 GitHub） |
| `results_*` | 训练/验证输出与 checkpoint（通常不建议上传 GitHub） |

## 3. 两阶段整体数据流

## 3.1 Stage1（基础重建）

输入（多相机、可含时序）后，主流程如下：

1. `PoseNetwork` 预测相机相对位姿。
2. `DepthNetwork` 预测每个相机的深度与中间特征。
3. `VFNet` 做体素空间融合，增强多视角一致性。
4. `GaussianNetwork` 输出高斯参数（旋转、尺度、不透明度、SH）。
5. `GaussianRender` 将高斯投影渲染到目标视角。
6. Stage1 loss 监督重建质量与几何一致性。

## 3.2 Stage2（动态补偿）

Stage2 读取并冻结 Stage1 的 `pose/depth/gaussian` 网络，仅训练动态相关模块：

1. 基于 Stage1 深度 + 位姿计算刚性光流 `F_rigid`。
2. 用 `ResFlowNet` 预测残差光流 `F_residual`。
3. 合成总流 `F_total = F_rigid + F_residual`。
4. 用总流 warp `t-1 / t+1` 图像到 `t` 时刻。
5. 在 `render_novel_view` 中执行融合：
   - 静态区：沿用 baseline mask-average。
   - 动态区：用残差流幅值构建 gate 和置信度加权融合。
6. 用 Stage2 loss 优化 warp、一致性、渲染项。

## 4. 关键入口脚本（怎么跑）

| 文件 | 作用 |
|---|---|
| `train.py` | Stage1 单卡训练入口 |
| `train_multi_gpu.py` | Stage1 DDP 训练入口 |
| `eval.py` | Stage1 评估入口（SF/MF） |
| `train_stage2.py` | Stage2 训练入口（兼容单卡/多卡） |
| `train_stage2_ddp.py` | Stage2 DDP 主入口（推荐） |
| `stage2_inference.py` | Stage2 推理脚本（导出预测） |
| `tools/export_stage2_images.py` | 导出 GT/预测拼图、误差图，用于可视化分析 |

## 5. Stage1 核心文件说明

## 5.1 主模型与训练器

- `models/drivingforward_model.py`：Stage1 单卡主模型封装。
- `models/drivingforward_model_ddp.py`：Stage1 DDP 封装。
- `trainer/trainer.py`：Stage1 训练/验证/评估流程。
- `trainer/trainer_ddp.py`：Stage1 DDP 训练流程。

## 5.2 网络模块

- `network/depth_network.py`
  - 当前版本为共享式深度特征编码（Swin backbone）+ 跨视角融合模块。
  - 包含 `CrossViewFusionBlock`，支持多相机 token 交互。
- `network/pose_network.py`
  - 位姿估计。
- `network/volumetric_fusionnet.py`
  - 体素空间回投影与融合（VFNet）。

## 5.3 Gaussian 相关

- `models/gaussian/gaussian_network.py`
  - 由图像特征 + 深度特征预测高斯参数。
  - 支持跨视角 attention 融合（可通过配置开关）。
- `models/gaussian/GaussianRender.py`
  - 高斯渲染入口（调用底层 rasterization）。
- `models/geometry/view_rendering.py`
  - 几何投影与虚拟视图渲染。

## 5.4 Stage1 损失

- `models/losses/single_cam_loss.py`：单相机重建损失。
- `models/losses/multi_cam_loss.py`：多相机与时空一致性项。

## 6. Stage2 核心文件说明

## 6.1 Stage2 模型与训练控制

- `stage2_trainer/stage2_model.py`
  - 标准 Stage2 模型（默认时空联合逻辑）。
  - 负责加载并冻结 Stage1 权重。
  - 负责 `compute_stage2_outputs` 和 `render_novel_view`。
- `stage2_trainer/stage2_model_multi_mode.py`
  - 支持 `temporal / spatio / spatio_temporal` 三模式。
- `stage2_trainer/stage2_trainer.py` / `stage2_trainer/stage2_trainer_ddp.py`
  - Stage2 单卡/多卡训练与验证循环。
- `stage2_trainer/model_factory.py`
  - 根据配置自动选择 `Stage2Model` 或 `Stage2ModelMultiMode`，并创建对应 loss。

## 6.2 Stage2 可训练模块

- `stage2_modules/rigid_flow.py`
  - 基于深度与位姿计算刚性流，提供 warp 函数。
- `stage2_modules/res_flow_net.py`
  - 当前残差流网络：
    - 双流编码（warped/target）
    - 粗尺度 cross-attention
    - U 型解码 + 多尺度 skip
    - 全分辨率轻量迭代 refine
  - `GroupNorm + inplace=False` 以提升 DDP 稳定性。
- `stage2_modules/dynamic_gaussian.py`
  - 动态高斯辅助生成器。
- `stage2_modules/stage2_loss_multi_mode.py`
  - 多模式损失实现（warp / consistency / render），含 LPIPS 防 NaN/Inf 保护。

## 6.3 动态融合逻辑（重点）

在 `stage2_model.py` 与 `stage2_model_multi_mode.py` 的 `render_novel_view`：

1. 先做 baseline 融合（mask-average）。
2. 若 `use_dynamic_fusion=True`：
   - 计算残差流幅值 `|F_residual|`。
   - 构造动态门控 `dyn_gate`。
   - 动态区域偏向“残差更小”的源帧。
   - 最终输出为 `base` 与 `dynamic` 的门控加权结果。

这部分不是后处理脚本，而是**模型前向中的融合路径**，会参与 Stage2 训练和评估。

## 7. 数据与配置系统

## 7.1 数据读取

- `dataset/base_dataset.py`：统一数据集构建入口。
- `dataset/nuscenes_dataset.py`：nuScenes 样本读取、相机内外参、时序上下文组织。
- `dataset/nuscenes/*.txt`：train/eval split 文件。

## 7.2 配置文件分层

常用配置在 `configs/nuscenes/`：

- Stage1 基础：`main.yaml`、`main_ddp.yaml`
- Stage1 cross-view：`main_ddp_stage1_crossview_4gpu.yaml`
- Stage1 no-cross-view（消融）：`main_ddp_stage1_nocrossview_4gpu.yaml`
- Stage2 基础：`phase2_training.yaml`、`phase2_training_multi_mode.yaml`
- Stage2 DDP：`phase2_training_multi_mode_ddp.yaml`
- Stage2 from stage1-crossview：`phase2_training_multi_mode_ddp_from_stage1_crossview_4gpu.yaml`
- 消融：`phase2_ablation_*.yaml`

关键字段：

- `data.log_dir`：实验目录根。
- `training.stage1_weights_path`：Stage2 读取 Stage1 权重位置。
- `model.enable_cross_view_fusion`：Stage1 跨视角融合开关。
- `model.use_dynamic_fusion`：Stage2 动态融合开关。
- `model.temporal / spatio / spatio_temporal`：Stage2 模式。

## 8. 任务脚本（服务器）

仓库根目录保留了大量 `sbatch_*.sh`，按用途大致分为：

- Stage1 训练：`sbatch_stage1_*`
- Stage2 训练：`sbatch_stage2_*`
- Stage2 评估：`sbatch_stage2_eval_*`
- 消融实验：`sbatch_ablation_*`

建议实践：

1. 一个实验配一套独立 `log_dir`，避免覆盖。
2. `cpus-per-task` 与 GPU 数量按集群规则匹配。
3. DDP 评估时保证各 rank 同步进入验证流程。

## 9. 可视化与分析工具

- `tools/export_stage2_images.py`
  - 导出 `GT | Final | Diff`、`GT | Base | Dynamic` 等面板图。
- `tools/analyze_ghosting_compare.py`
  - 针对重影场景做定性/定量辅助分析。

## 10. 论文写作建议（基于本仓库）

建议将方法章节按以下层次描述：

1. Stage1：静态重建主干（Pose + Depth + Gaussian）。
2. Stage2：刚性流先验 + 残差流学习。
3. 动态融合：残差流驱动的门控置信融合。
4. 多模式训练：temporal / spatio / spatio-temporal。
5. 消融设置：E0/E1/E2/E3（见 `docs/ABLATION_RUNBOOK_V1_CN.md`）。

## 11. 快速阅读顺序（给人/给 AI）

建议阅读顺序：

1. `README.md`
2. `configs/nuscenes/phase2_training_multi_mode_ddp_from_stage1_crossview_4gpu.yaml`
3. `stage2_trainer/stage2_model_multi_mode.py`
4. `stage2_modules/res_flow_net.py`
5. `stage2_modules/stage2_loss_multi_mode.py`
6. `network/depth_network.py`
7. `models/gaussian/gaussian_network.py`
8. `docs/OPTIMIZATION_REPORT_V3_CN.md`
9. `docs/DRIVINGFORWARD_V3_VS_ORIGINAL_CN.md`

---

如果后续你要把这份文档精简成论文附录版，我建议保留第 2、3、6、7、10 节，其余作为代码仓库说明。
