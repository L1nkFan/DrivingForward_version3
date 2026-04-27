#!/bin/bash
#SBATCH --job-name=df_v3_s2_eval4
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00
#SBATCH --output=sh_log_err/log_%j.out
#SBATCH --error=sh_log_err/%j.err

set -euo pipefail

source ../anaconda3/etc/profile.d/conda.sh
conda activate DrivingForward
module load cuda/12.1
module load gcc/11.4.0

cd /share/home/u21053/ckf/DrivingForward_version3
mkdir -p sh_log_err

# 7 CPU threads per GPU (cluster policy: <=7 cores/GPU)
export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export NCCL_DEBUG=WARN

CFG=${1:-configs/nuscenes/phase2_training_multi_mode_ddp.yaml}
CKPT=${2:-}

if [ -z "$CKPT" ]; then
  echo "[ERROR] Missing checkpoint path."
  echo "Usage: sbatch $0 [config_file] /path/to/res_flow_net_epoch_X.pth"
  exit 1
fi

if [ ! -f "$CKPT" ]; then
  echo "[ERROR] Checkpoint not found: $CKPT"
  exit 1
fi


if [ ! -f "$CFG" ]; then
  echo "[WARN] Config not found: $CFG"
  for c in \
    configs/nuscenes/phase2_training_multi_mode_ddp_from_stage1_crossview_4gpu.yaml \
    configs/nuscenes/phase2_training_multi_mode_ddp_from_stage1_crossview_debug_1gpu.yaml \
    configs/nuscenes/phase2_training_multi_mode_ddp.yaml \
    configs/nuscenes/phase2_training_multi_mode.yaml; do
    if [ -f "$c" ]; then
      CFG="$c"
      break
    fi
  done
fi

if [ ! -f "$CFG" ]; then
  echo "[ERROR] No usable config file found under configs/nuscenes"
  ls -1 configs/nuscenes || true
  exit 1
fi


echo "[INFO] Eval config: $CFG"
echo "[INFO] Checkpoint: $CKPT"

torchrun --standalone --nproc_per_node=4 train_stage2_ddp.py \
  --config_file="$CFG" \
  --resume="$CKPT" \
  --eval_only
