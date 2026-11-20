#!/usr/bin/env bash
#SBATCH --job-name=msm_loweta_post
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/msm_loweta_post_%j.out
#SBATCH --error=logs/msm_loweta_post_%j.err

set -e

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs results results/representations_32L

if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE_MODEL="meta-llama/Llama-3.1-8B"

echo ">>> [1/4] Checking Omega vs. IT Saturation Losses..."
grep -H "tail_omega_loss\|tail_it_loss" checkpoints/aft__*mode-convex*/training_complete || true

echo ">>> [2/4] Compiling Regressions & All Alternative Arms..."
python scripts/04_evaluate_and_regress.py --mode regress --results_dir results

echo ">>> [3/4] Updating Bar Plots..."
python scripts/04_evaluate_and_regress.py --mode plot --results_dir results

echo ">>> [4/4] Running Full 32-Layer Representational & eNTK Scan (seed=42)..."
ALL_32_LAYERS="$(seq -s, 0 31)"
python scripts/05_analyze_representations.py \
    --base_model "${BASE_MODEL}" \
    --checkpoints_dir checkpoints \
    --data_dir data \
    --results_dir results \
    --out_dir results/representations_32L \
    --layers "${ALL_32_LAYERS}" \
    --seed 42 \
    --post_aft_scope all

echo ">>> All post-processing & 32-layer analysis complete!"