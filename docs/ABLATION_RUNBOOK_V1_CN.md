# Ablation Runbook (CN)

## Experiment IDs
- E0 Full: Stage1 cross-view + Stage2 + dynamic fusion
- E1 Stage1-only: no Stage2/ResFlowNet
- E2 w/o Dynamic Fusion: Stage2 on, but disable dynamic fusion gate/blend
- E3 w/o Stage1 Cross-View Fusion: train Stage1 with cross-view fusion disabled, then Stage2 from those weights

## Config files
- E0: `configs/nuscenes/phase2_ablation_full_ddp.yaml`
- E2: `configs/nuscenes/phase2_ablation_wo_dynamic_fusion_ddp.yaml`
- E3 stage1: `configs/nuscenes/main_ddp_stage1_nocrossview_4gpu.yaml`
- E3 stage2: `configs/nuscenes/phase2_ablation_from_stage1_nocrossview_ddp.yaml`

## Training commands
1. E0 Full Stage2 train:
   `sbatch sbatch_ablation_e0_stage2_full_train_4gpu.sh`
2. E2 w/o Dynamic Fusion Stage2 train:
   `sbatch sbatch_ablation_e2_stage2_wo_dynamic_train_4gpu.sh`
3. E3 Stage1 no-crossview train:
   `sbatch sbatch_ablation_stage1_nocrossview_train_4gpu.sh`
4. E3 Stage2 from no-crossview Stage1 train:
   `sbatch sbatch_ablation_e3_stage2_wo_stage1_crossview_train_4gpu.sh`

## Evaluation commands
1. E1 Stage1-only eval:
   `sbatch sbatch_ablation_e1_stage1_only_eval_1gpu.sh`
2. Stage2 eval (E0/E2/E3): use generic script:
   `sbatch sbatch_stage2_eval_1gpu.sh <config_yaml> <checkpoint_path>`

## Suggested stage2 eval config mapping
- E0 checkpoint -> `configs/nuscenes/phase2_ablation_full_ddp.yaml`
- E2 checkpoint -> `configs/nuscenes/phase2_ablation_wo_dynamic_fusion_ddp.yaml`
- E3 checkpoint -> `configs/nuscenes/phase2_ablation_from_stage1_nocrossview_ddp.yaml`
