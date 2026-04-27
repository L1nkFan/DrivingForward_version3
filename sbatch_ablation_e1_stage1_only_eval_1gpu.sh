#!/bin/bash
#SBATCH --job-name=df_v3_ab_e1_eval
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
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

CFG=${1:-configs/nuscenes/main_ddp_stage1_crossview_4gpu.yaml}
WEIGHTS=${2:-}

if [ -z "$WEIGHTS" ]; then
  WEIGHTS=$(ls -d results_stage1_crossview_ddp_v1/main_ddp_stage1_crossview_4gpu/models/weights_* 2>/dev/null | sort -V | tail -n1 || true)
fi

if [ -z "$WEIGHTS" ] || [ ! -d "$WEIGHTS" ]; then
  echo "[ERROR] Stage1 weights directory not found: $WEIGHTS"
  exit 1
fi

if [ ! -f "$CFG" ]; then
  echo "[ERROR] Config not found: $CFG"
  exit 1
fi

echo "[INFO] Stage1 eval config: $CFG"
echo "[INFO] Stage1 weights dir: $WEIGHTS"

python -W ignore eval.py \
  --config_file="$CFG" \
  --weight_path="$WEIGHTS" \
  --novel_view_mode=MF
