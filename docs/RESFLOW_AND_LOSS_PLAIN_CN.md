# ResFlowNet 架构与损失函数说明（通俗版）

这份文档只讲两件事：
- `ResFlowNet` 到底在做什么、怎么做的；
- Stage2 的损失函数在约束什么、为什么这么设计。

不讲论文话术，尽量用直白语言。

---

## 1. 先说结论：ResFlowNet 是“补偿器”

在 Stage2 里，我们已经有了一个**刚性光流**（来自相机运动+深度）：
- 它对静态背景通常有效；
- 但对动态物体（行人、车）经常不够，容易重影。

所以 ResFlowNet 的任务不是“重做全部光流”，而是：
- 只预测一个**残差光流** `F_residual`；
- 最终光流是  
  `F_total = F_rigid + F_residual`。

可以把它理解成：
- `F_rigid` 是基础答案；
- `F_residual` 是纠错项。

对应代码文件：  
[res_flow_net.py](/d:/Code/DrivingForward_version3/stage2_modules/res_flow_net.py)

---

## 2. ResFlowNet 的输入输出（先看这个最清楚）

每个相机、每个方向（`t-1 -> t` 或 `t+1 -> t`）输入三样东西：
- `warped_img`：源帧先用刚性光流对齐后的图；
- `tgt_img`：目标帧图像（当前时刻）；
- `rigid_flow`：刚性光流。

输出：
- `residual_flow`（2通道，x/y 位移）。

多相机时，网络按相机逐个算，最后拼回去（形状 `[B, N, 2, H, W]`）。

---

## 3. ResFlowNet 结构，按“流水线”解释

## 3.1 双流图像编码（共享权重）

`warped_img` 和 `tgt_img` 走同一个 `ImageEncoder`（参数共享）。

直觉：
- 两张图语义相近，没必要两套编码器；
- 共享可以省参数，还能让两路特征在同一语义空间里比较。

---

## 3.2 刚性光流编码（单独一条支路）

`rigid_flow` 不当图像处理，而是单独走 `FlowEncoder`。

直觉：
- 刚性流是很有价值的先验，告诉网络“理论上应该往哪移”；
- 单独编码后更容易在后续融合时被利用。

---

## 3.3 粗尺度跨注意力融合

在最粗层特征上做 `CrossAttentionFusion`：
- Query 用目标帧特征；
- Key/Value 用 warped 帧特征。

直觉：
- 这一步像“问答”：目标图问“我这里该对应你哪里？”；
- 有助于定位大范围错位。

---

## 3.4 U 型解码 + 多尺度 skip 融合

网络把三类信息融合后逐层上采样解码：
- warped 图特征；
- target 图特征；
- rigid flow 特征。

为什么要多尺度：
- 粗层管大位移；
- 细层修边缘和细节。

---

## 3.5 全分辨率迭代 refine（关键）

先得到一版流，再做 2 次轻量迭代细修（默认 `refine_iters=2`）：
- 每次输入 `[warped_img, tgt_img, rigid_flow, current_flow]`；
- 输出一个增量 `delta`；
- `flow = flow + delta`。

直觉：
- 一次性预测往往粗糙；
- 迭代修正更稳，尤其对动态边缘。

---

## 3.6 稳定性细节：GroupNorm + 禁用 inplace

代码里把归一化改为 GroupNorm，激活基本使用 `inplace=False`。

目的：
- 多卡和复杂反向传播时更稳；
- 减少梯度图“版本冲突”类报错。

---

## 4. 损失函数在约束什么

对应文件：  
[stage2_loss_multi_mode.py](/d:/Code/DrivingForward_version3/stage2_modules/stage2_loss_multi_mode.py)

核心有 3 类损失：
- `warp`（对齐损失）
- `consistency`（前后向一致性）
- `render`（最终渲染质量）

---

## 4.1 Warp Loss：看“对齐后像不像目标图”

先用 `F_total` 把源图 warp 到目标时刻，再和目标图比：
- `L1`
- `SSIM`（结构相似）
- 可选 `LPIPS`（感知）

在当前代码里，单次 warp 损失是：
- `L_warp = L1 + 0.1 * SSIM (+ 0.05 * LPIPS，可选)`

这里 LPIPS 默认在很多配置中是关的（为了稳定和速度）。

---

## 4.2 Consistency Loss：看“光流是否自洽”

思路：
- 有了前向流 `A -> B`，再估一个后向流 `B -> A`；
- 理想状态下，两者应该互相抵消。

约束形式（直觉表达）：
- `F_forward_warped + F_backward ≈ 0`

这会抑制“看起来能对齐，但几何上很乱”的解。

---

## 4.3 Render Loss：看最终重建图质量

Stage2 会融合出最终 `rendered_I_t`，再和 GT 比：
- 主体是 `L2`；
- 可选叠加 `LPIPS`。

直觉：
- warp loss 关注“像素对应关系”；
- render loss 直接关注“最后图像好不好”。

---

## 5. 三种训练模式下，总损失怎么组

## 5.1 Temporal 模式

- 用前后帧（`t-1`、`t+1`）对当前帧 `t` 做约束。
- 总损失：
  `total = λ_warp * loss_warp + λ_consist * loss_consist + λ_render * loss_render`

---

## 5.2 Spatio 模式

- 只看同一时刻不同相机之间的约束。
- 总损失：
  `total = λ_warp * loss_warp_spatial + λ_consist * loss_consist_spatial`
- 通常不含 render 项。

---

## 5.3 Spatio-Temporal 模式（最常用）

- 同时加 temporal 和 spatio 两部分。
- 总损失：
  `total = λ_warp * (warp_temporal + warp_spatial) + λ_consist * (consist_temporal + consist_spatial) + λ_render * render`

---

## 6. 为什么会改善重影（但不保证一次就完美）

它的逻辑是：
1. 先用 `F_rigid` 处理静态大结构；
2. 用 `F_residual` 专门修正动态区域；
3. 融合时对残差幅值做动态门控（动态区域更谨慎地混合两侧来源）。

所以对动态重影是“有针对性”的。

但如果你还看到明显重影，常见原因是：
- 动态样本占比太低，网络没学够；
- `λ_consist` 太弱或太强；
- 训练轮数不够；
- 当前融合策略对极端遮挡还不够强。

---

## 7. 你可以这样快速判断训练是否在朝正确方向走

如果训练健康，通常会看到：
- `loss_warp` 缓慢下降；
- `loss_consist` 不爆炸、不突然变 NaN；
- 验证集 `PSNR/SSIM` 上升，`LPIPS` 下降。

如果出现：
- 指标几乎不变 + 动态仍拖影，
优先检查：
1. 是否确实加载了 Stage1 权重；  
2. 是否在训练 Stage2 的 `res_flow_net`（不是空跑）；  
3. 当前模式是不是 `spatio_temporal`；  
4. 训练时长是否足够覆盖动态样本。

---

## 8. 一句话总结

`ResFlowNet` 不是替代刚性流，而是给刚性流做“动态纠错”；  
损失函数就是从“对齐正确、流场自洽、最终图像好看”三方面同时拉住它。
