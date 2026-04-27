#!/bin/bash
#SBATCH --job-name=df_v3_s2_ddp_dbg1
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --output=sh_log_err/log_%j.out
#SBATCH --error=sh_log_err/%j.err

set -euo pipefail

source ../anaconda3/etc/profile.d/conda.sh
conda activate DrivingForward
module load cuda/12.1
module load gcc/11.4.0

cd /share/home/u21053/ckf/DrivingForward_version3
mkdir -p sh_log_err

export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7

# Debug run: anomaly detection for precise autograd/inplace traceback
# If needed, switch to 4 GPUs by setting:
#   --gres=gpu:a800:4
# and torchrun --nproc_per_node=4

torchrun --standalone --nproc_per_node=1 train_stage2_ddp.py \
  --config_file=configs/nuscenes/phase2_training_multi_mode_ddp.yaml \
  --detect_anomaly

