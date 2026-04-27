#!/bin/bash
#SBATCH --job-name=df_v3_s2_cv_dbg1
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=sh_log_err/log_%j.out
#SBATCH --error=sh_log_err/%j.err

set -euo pipefail

source ../anaconda3/etc/profile.d/conda.sh
conda activate DrivingForward
module load cuda/12.1
module load gcc/11.4.0

cd /share/home/u21053/ckf/DrivingForward_version3
mkdir -p sh_log_err
mkdir -p weight/weights_stage1_crossview_v1
mkdir -p weight/weights_MF

# Auto-pick latest Stage1 checkpoint folder.
LATEST_STAGE1=$(ls -d results_stage1_crossview_ddp_v1/main_ddp_stage1_crossview_4gpu/models/weights_* 2>/dev/null | sort -V | tail -n1 || true)
if [ -z "$LATEST_STAGE1" ]; then
  echo "[ERROR] Cannot find Stage1 weights folder under results_stage1_crossview_ddp_v1/.../models/weights_*"
  exit 1
fi

if [ ! -f "$LATEST_STAGE1/pose_net.pth" ] || [ ! -f "$LATEST_STAGE1/depth_net.pth" ]; then
  echo "[ERROR] pose_net.pth or depth_net.pth missing in $LATEST_STAGE1"
  exit 1
fi

cp -f "$LATEST_STAGE1/pose_net.pth"  weight/weights_stage1_crossview_v1/
cp -f "$LATEST_STAGE1/depth_net.pth" weight/weights_stage1_crossview_v1/

if [ -f "$LATEST_STAGE1/gs_net.pth" ]; then
  cp -f "$LATEST_STAGE1/gs_net.pth" weight/weights_stage1_crossview_v1/
elif [ -f "$LATEST_STAGE1/gaussian_net.pth" ]; then
  cp -f "$LATEST_STAGE1/gaussian_net.pth" weight/weights_stage1_crossview_v1/gs_net.pth
else
  echo "[ERROR] gs_net.pth / gaussian_net.pth missing in $LATEST_STAGE1"
  exit 1
fi

# Keep a compatibility copy for legacy configs that still point to weights_MF.
cp -f weight/weights_stage1_crossview_v1/pose_net.pth  weight/weights_MF/pose_net.pth
cp -f weight/weights_stage1_crossview_v1/depth_net.pth weight/weights_MF/depth_net.pth
cp -f weight/weights_stage1_crossview_v1/gs_net.pth    weight/weights_MF/gs_net.pth

echo "[INFO] Using Stage1 weights from: $LATEST_STAGE1"
ls -lh weight/weights_stage1_crossview_v1

export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export NCCL_DEBUG=WARN

# Config fallback: some servers may not yet have the new cross-view yaml.
CONFIG_TEMPLATE=""
for c in \
  configs/nuscenes/phase2_training_multi_mode_ddp_from_stage1_crossview_debug_1gpu.yaml \
  configs/nuscenes/phase2_training_multi_mode_ddp_from_stage1_crossview_4gpu.yaml \
  configs/nuscenes/phase2_training_multi_mode_ddp.yaml \
  configs/nuscenes/phase2_training_multi_mode.yaml; do
  if [ -f "$c" ]; then
    CONFIG_TEMPLATE="$c"
    break
  fi
done

if [ -z "$CONFIG_TEMPLATE" ]; then
  echo "[ERROR] No usable Stage2 config found under configs/nuscenes/"
  ls -1 configs/nuscenes || true
  exit 1
fi

RUNTIME_CFG="configs/nuscenes/.phase2_runtime_stage1_crossview_debug_1gpu.yaml"
cp -f "$CONFIG_TEMPLATE" "$RUNTIME_CFG"
sed -i "s#^\([[:space:]]*stage1_weights_path:[[:space:]]*\).*#\1'./weight/weights_stage1_crossview_v1'#g" "$RUNTIME_CFG"
sed -i "s#^\([[:space:]]*load_dir:[[:space:]]*\).*#\1'./weight/weights_stage1_crossview_v1'#g" "$RUNTIME_CFG"
sed -i "s#^\([[:space:]]*log_dir:[[:space:]]*\).*#\1'./results_stage2_from_stage1_crossview_debug_1gpu_v1/'#g" "$RUNTIME_CFG"
sed -i "s#^\([[:space:]]*save_path:[[:space:]]*\).*#\1'./results_stage2_from_stage1_crossview_debug_1gpu_v1/images'#g" "$RUNTIME_CFG"
sed -i "s#^\([[:space:]]*world_size:[[:space:]]*\).*#\11#g" "$RUNTIME_CFG"
sed -i "s#^\([[:space:]]*gpus:[[:space:]]*\).*#\1[0]#g" "$RUNTIME_CFG"
sed -i "s#^\([[:space:]]*scale_lr:[[:space:]]*\).*#\1False#g" "$RUNTIME_CFG"

echo "[INFO] Config template: $CONFIG_TEMPLATE"
echo "[INFO] Runtime config:   $RUNTIME_CFG"

torchrun --standalone --nproc_per_node=1 train_stage2_ddp.py \
  --config_file="$RUNTIME_CFG" \
  --detect_anomaly