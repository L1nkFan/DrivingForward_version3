#!/bin/bash
#SBATCH --job-name=df_v3_ab_e0_4g
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=3-12:00:00
#SBATCH --output=sh_log_err/log_%j.out
#SBATCH --error=sh_log_err/%j.err

set -euo pipefail

source ../anaconda3/etc/profile.d/conda.sh
conda activate DrivingForward
module load cuda/12.1
module load gcc/11.4.0

cd /share/home/u21053/ckf/DrivingForward_version3
mkdir -p sh_log_err
mkdir -p weight/weights_stage1_crossview_v1 weight/weights_MF

LATEST_STAGE1=$(ls -d results_stage1_crossview_ddp_v1/main_ddp_stage1_crossview_4gpu/models/weights_* 2>/dev/null | sort -V | tail -n1 || true)
if [ -z "$LATEST_STAGE1" ]; then
  echo "[ERROR] Cannot find cross-view stage1 weights folder"
  exit 1
fi

cp -f "$LATEST_STAGE1/pose_net.pth"  weight/weights_stage1_crossview_v1/
cp -f "$LATEST_STAGE1/depth_net.pth" weight/weights_stage1_crossview_v1/
if [ -f "$LATEST_STAGE1/gs_net.pth" ]; then
  cp -f "$LATEST_STAGE1/gs_net.pth" weight/weights_stage1_crossview_v1/
elif [ -f "$LATEST_STAGE1/gaussian_net.pth" ]; then
  cp -f "$LATEST_STAGE1/gaussian_net.pth" weight/weights_stage1_crossview_v1/gs_net.pth
else
  echo "[ERROR] Missing gs_net.pth / gaussian_net.pth in $LATEST_STAGE1"
  exit 1
fi

cp -f weight/weights_stage1_crossview_v1/pose_net.pth  weight/weights_MF/pose_net.pth
cp -f weight/weights_stage1_crossview_v1/depth_net.pth weight/weights_MF/depth_net.pth
cp -f weight/weights_stage1_crossview_v1/gs_net.pth    weight/weights_MF/gs_net.pth

export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export NCCL_DEBUG=WARN

torchrun --standalone --nproc_per_node=4 train_stage2_ddp.py \
  --config_file=configs/nuscenes/phase2_ablation_full_ddp.yaml
