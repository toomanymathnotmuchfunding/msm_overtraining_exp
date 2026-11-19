#!/usr/bin/env bash
#SBATCH --job-name=msm_4seeds_post
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/msm_4seeds_post_%j.out
#SBATCH --error=logs/msm_4seeds_post_%j.err

set -e

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs results

if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE_MODEL="meta-llama/Llama-3.1-8B"

echo ">>> [1/3] Compiling Multi-Seed Econometric Regressions..."
python scripts/04_evaluate_and_regress.py --mode regress --results_dir results

echo ">>> [2/3] Generating Multi-Seed Bar Plots..."
python scripts/04_evaluate_and_regress.py --mode plot --results_dir results

echo ">>> [3/3] Running Representational & Sketched eNTK Analysis (seed=42)..."
python scripts/05_analyze_representations.py \
    --base_model "${BASE_MODEL}" \
    --checkpoints_dir checkpoints \
    --data_dir data \
    --results_dir results \
    --out_dir results/representations \
    --seed 42 \
    --post_aft_scope all

echo ">>> Post-processing complete!"