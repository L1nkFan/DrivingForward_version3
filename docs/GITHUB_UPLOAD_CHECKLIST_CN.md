# GitHub 上传整理清单（DrivingForward_version3）

> 目标：在不丢失复现能力的前提下，清理训练产物、统一文档入口、形成适合开源与 AI 阅读的仓库结构。

## 1. 当前已完成的仓库整理

1. 新增根目录 `.gitignore`，屏蔽日志、checkpoint、训练输出、缓存文件。
2. 新增架构文档：`docs/REPO_ARCHITECTURE_GUIDE_CN.md`。
3. 保留已有实验文档（优化、消融、对比说明）作为补充证据。

## 2. 上传前建议保留的内容

建议保留（应上传）：

- 源码目录：`configs/`、`dataset/`、`models/`、`network/`、`stage2_modules/`、`stage2_trainer/`、`trainer/`、`tools/`、`utils/`
- 文档目录：`README.md`、`docs/*.md`
- 运行脚本：`train*.py`、`eval.py`、`stage2_inference.py`、`sbatch_*.sh`
- 依赖文件：`requirements.txt`
- 许可证：`LICENSE`

建议不上传（应忽略）：

- `results_*`、`results_stage*`、`results_ablation`（训练产物）
- `weight/`（本地权重）
- `__pycache__/`、`*.pyc`
- 运行日志：`*.out`、`*.err`、`*.log`、`sh_log_err/`

## 3. 推荐的最小可复现集

如果希望第三方只凭仓库即可“看懂+跑通命令”，最少需要：

1. 至少一套 Stage1 配置：`configs/nuscenes/main_ddp_stage1_crossview_4gpu.yaml`
2. 至少一套 Stage2 配置：`configs/nuscenes/phase2_training_multi_mode_ddp_from_stage1_crossview_4gpu.yaml`
3. 一个消融配置示例：`configs/nuscenes/phase2_ablation_full_ddp.yaml`
4. 一个可视化脚本：`tools/export_stage2_images.py`
5. 一份实验说明：`docs/ABLATION_RUNBOOK_V1_CN.md`

## 4. 建议的提交顺序（Commit Plan）

1. `chore: add gitignore for training artifacts and logs`
2. `docs: add repository architecture guide in Chinese`
3. `docs: add github upload checklist`
4. `docs: update README links to CN docs`（如果你接受 README 增补）

## 5. 上传前本地自检命令

```bash
# 1) 看即将提交的文件
 git status

# 2) 确认大文件没有误入（Linux）
 find . -type f -size +50M

# 3) 确认训练产物不在追踪列表（如果 git 已初始化）
 git ls-files | grep -E "results_|\.pth$|\.pt$|\.ckpt$"
```

## 6. 对论文协作最友好的仓库实践

1. 每个核心模块单独有文档锚点（本仓库已具备）。
2. 配置文件名体现实验意图（如 `*_from_stage1_crossview_*`、`ablation_*`）。
3. 日志/权重不进仓库，只在文档记录路径和命令。
4. 结果图保留少量代表样例（可放 `assets/` 或 `docs/figures/`），不要上传完整训练输出目录。

## 7. 你接下来可以直接做的事

1. 在学校服务器同步这次改动（`.gitignore` + `docs`）。
2. 本地初始化或切换到 git 仓库后执行 `git status`，确认忽略规则生效。
3. 推送 GitHub 后，将 Gemini 入口提示写成：
   - “先读 `docs/REPO_ARCHITECTURE_GUIDE_CN.md`，再读 `docs/DRIVINGFORWARD_V3_VS_ORIGINAL_CN.md`。”

---

如果你愿意，我下一步可以继续帮你做一版“论文附录可直接粘贴”的模块说明模板（方法、实现细节、消融设置三段式）。
