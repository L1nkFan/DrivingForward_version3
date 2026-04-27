# DrivingForward Version3 阶段二优化说明（相对最初版本）

## 1. 目标与范围

本轮优化目标是：
- 提高训练稳定性，避免多卡训练中断。
- 在不改动 Stage1 主体训练流程的前提下，优化 Stage2（ResFlowNet 与损失/训练流程）。
- 为后续提升 PSNR / SSIM / LPIPS 提供稳定可复现的训练基线。

本说明聚焦 `DrivingForward_version3` 中可追踪到的实际改动。

## 2. 核心优化清单

### 2.1 第二阶段网络结构重构（ResFlowNet）


改动：
- 将 Stage2 残差流预测网络替换为新架构。
- 采用“共享双流编码器”设计：
  - `warped_img` 与 `tgt_img` 通过同一个 `ImageEncoder`（参数共享）。
- 新增 `FlowEncoder` 编码刚性流先验。
- 在粗尺度引入 `CrossAttentionFusion`（query=target, key/value=warped）。
- 采用 U-shaped 解码器并融合多尺度 skip 特征（图像双流 + 刚性流）。
- 增加轻量全分辨率迭代 refinement head。
- 通道宽度做了收缩（`img_base`, `flow_base`），平衡显存/速度。

对应文件：
- `stage2_modules/res_flow_net.py`（关键类：`ImageEncoder`、`FlowEncoder`、`CrossAttentionFusion`、`ResFlowNet`）

预期收益：
- 更好利用时空与刚性先验，提升残差流估计质量。
- 共享编码器减少重复参数，更符合你的“六视角编码部分融合”思路。
- 在相同显存预算下提升可训练性与吞吐。

### 2.2 归一化与 inplace 稳定性修复

问题：多卡/长时间训练中，BatchNorm 统计与 inplace 操作可能引发不稳定或 autograd 版本冲突。

改动：
- 将网络中归一化改为 `GroupNorm`（通过 `_build_norm` 统一构建）。
- 将多个激活层改为 `SiLU(inplace=False)`。
- 在 Stage2Model 初始化时递归关闭训练模块中的 inplace 行为（安全兜底）。

对应文件：
- `stage2_modules/res_flow_net.py`
- `stage2_trainer/stage2_model_multi_mode.py`（`_disable_inplace_ops`）

预期收益：
- 降低多卡场景下归一化统计带来的不确定性。
- 降低 inplace 导致的梯度图异常概率。

### 2.3 损失函数鲁棒性增强（重点修复 LPIPS 相关崩溃）

问题：训练中出现过 LPIPS 相关 NaN/Inf，导致任务中断。

改动：
- 在 LPIPS 计算前加入 `torch.nan_to_num` 和 `clamp(0,1)`。
- 对 warping 输出和目标图像也增加 NaN/Inf 清洗。
- 对 LPIPS 输出再次做 `nan_to_num` 兜底。
- 将 `enable_spatial_consistency`、`enable_render_lpips` 作为可配置开关（默认可关闭高风险项）。

对应文件：
- `stage2_modules/stage2_loss_multi_mode.py`

预期收益：
- 明显降低“训练跑到中途因非法数值崩溃”的概率。
- 在复杂 warping 场景下保持损失可计算。

### 2.4 TensorBoard 日志健壮性修复

问题：loss 字典包含非标量或非 tensor 标量时，`add_scalar` 可能报错。

改动：
- 仅记录标量 tensor（`numel()==1`）和数值类型（int/float/np.number）。
- 跳过字符串等元信息，避免日志线程导致训练失败。

对应文件：
- `utils/logger.py`

预期收益：
- 避免“日志写入异常导致训练中断”。

### 2.5 DDP 训练流程修复（多卡稳定性关键）

问题：你的多卡任务在跑完一个 epoch 后发生 NCCL timeout；根因是 epoch 末尾对 DDP 模块做了不安全重包裹，触发 rank 间不同步。

改动：
- DDP 仅在训练开始时包装一次（`find_unused_parameters=False`）。
- 验证阶段改为所有 rank 同步进入（避免 rank0-only 验证导致 deadlock）。
- epoch 末尾只在 rank0 做 `save_model`，不再解包/重包裹 DDP。
- 在关键节点保留 barrier，保证各 rank 步调一致。

对应文件：
- `stage2_trainer/stage2_trainer_ddp.py`

预期收益：
- 修复“epoch 完成后崩溃”的主路径问题。
- 降低 NCCL collective timeout 概率。

### 2.6 Checkpoint 读写兼容性增强（DDP/单卡互通）

问题：DDP 下 checkpoint 可能带 `module.` 前缀，直接加载到非 DDP 或反向加载时会 key 不匹配。

改动：
- `save_model` 自动判断 DDP：若有 `.module` 则保存 `module.state_dict()`。
- `load_res_flow_net` 支持：
  - 纯 `state_dict`。
  - 包装字典（`state_dict` 或 `model` 字段）。
  - 自动剥离 `module.` 前缀。

对应文件：
- `stage2_trainer/stage2_model_multi_mode.py`
- `stage2_trainer/stage2_model.py`

预期收益：
- 提高单卡/多卡切换时的续训成功率。
- 减少 checkpoint 键不匹配报错。

### 2.7 续训与调试入口完善

改动：
- 增加/启用 `--resume` 参数加载 ResFlowNet checkpoint。
- 增加 `--detect_anomaly` 调试开关，定位 autograd/inplace 问题。

对应文件：
- `train_stage2_ddp.py`

预期收益：
- 可在任务中断后快速续训。
- 问题复现与定位更直接。

### 2.8 训练配置与作业脚本规范化

改动：
- 新建 DDP 配置并使用独立 `log_dir`，避免覆盖旧实验结果。
- 设置 `world_size: 4`、`gpus: [0,1,2,3]`、`scale_lr: True`。
- 默认关闭高风险附加项：
  - `enable_spatial_consistency: False`
  - `enable_render_lpips: False`
- 提供 1 卡 debug 与 4 卡正式训练 sbatch。
- 按集群策略控制 CPU/GPU 比例（7 核/卡），设置 `OMP_NUM_THREADS` 与 `MKL_NUM_THREADS`。

对应文件：
- `configs/nuscenes/phase2_training_multi_mode_ddp.yaml`
- `sbatch_stage2_ddp_debug_1gpu.sh`
- `sbatch_stage2_ddp_train_4gpu.sh`

预期收益：
- 训练任务提交更稳定、可复现。
- 降低因资源配置不合规导致的提交失败。

## 3. 与最初版本相比，哪些地方没有改

- Stage1 训练流程本身未重构，仍采用“加载并冻结 Stage1 权重”的方式服务于 Stage2。
- 本轮主要聚焦 Stage2 网络、损失稳定性、DDP 训练可靠性、续训链路。

## 4. 对指标（PSNR / SSIM / LPIPS）的实际意义

- 这些优化里，一部分是“上限型优化”（新 ResFlowNet 架构、跨模态融合），会影响指标上限。
- 另一部分是“稳定性优化”（NaN 防护、DDP 同步、日志防崩），本身不直接提高上限，但能让训练完整收敛，避免中途中断导致指标无法提升。
- 从工程角度，先保证稳定跑完，再做损失权重和数据策略微调，才是提高三指标的有效路径。

## 5. 快速复查命令（服务器可直接用）

```bash
grep -n "def _build_norm" stage2_modules/res_flow_net.py
grep -n "GroupNorm" stage2_modules/res_flow_net.py
grep -n "Guard LPIPS against NaN/Inf" stage2_modules/stage2_loss_multi_mode.py
grep -n "nan_to_num" stage2_modules/stage2_loss_multi_mode.py
grep -n "rank-0-only validation can deadlock NCCL" stage2_trainer/stage2_trainer_ddp.py
grep -n "model.save_model" stage2_trainer/stage2_trainer_ddp.py
grep -n "k\[7:\] if k.startswith('module.')" stage2_trainer/stage2_model_multi_mode.py
grep -n "--resume" train_stage2_ddp.py
```

## 6. 结论

你最初提出的两条方向（替换 Stage2 架构、前期特征提取共享化）已经在 Stage2 路线中落地；同时配套完成了多卡稳定性和续训链路修复。当前版本的核心价值是：
- 能更稳定跑完长训练。
- 能在单卡/多卡间更顺滑切换。
- 为下一步针对 PSNR/SSIM/LPIPS 做系统化调参提供可复现实验底座。
PSNR: 30.5523
SSIM: 0.9001
LPIPS: 0.1712

PSNR: 31.4517
SSIM: 0.9236
LPIPS: 0.1275