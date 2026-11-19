#!/usr/bin/env bash
#SBATCH --job-name=msm_4seeds
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=05:55:00
#SBATCH --array=0-9
#SBATCH --output=logs/msm_4seeds_%A_%a.out
#SBATCH --error=logs/msm_4seeds_%A_%a.err

set -e

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
source "${PROJECT_DIR}/scripts/setup_storage.sh"

# Route heavy storage (checkpoints, datasets, HF cache) to /orcd/scratch
SCRATCH_ROOT="/orcd/scratch/REPLACE_WITH_YOUR_SCRATCH_PATH/msm_overtraining_exp"
CKPT_DIR="${SCRATCH_ROOT}/checkpoints"
export DATA_DIR="${SCRATCH_ROOT}/data"
export HF_HOME="${SCRATCH_ROOT}/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs results "${CKPT_DIR}" "${DATA_DIR}" "${HF_HOME}"

if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

BASE_MODEL="meta-llama/Llama-3.1-8B"
WEIGHTING_MODE="loss_weight"
ETAS=("0.25" "0.5" "1.0" "4.0")

eval_if_missing() {
    local MSM_ARG="$1"
    local AFT_ARG="$2"
    local OUT_JSONL="$3"
    local LABEL="$4"
    if [ ! -f "${OUT_JSONL}.done" ]; then
        echo ">>> [Eval] ${LABEL}..."
        python scripts/04_evaluate_and_regress.py \
            --mode eval_single \
            --base_model "${BASE_MODEL}" \
            --msm_dir "${MSM_ARG}" \
            --aft_dir "${AFT_ARG}" \
            --data_dir "${DATA_DIR}" \
            --out_jsonl "${OUT_JSONL}"
    else
        echo ">>> [Skip Eval] ${LABEL}"
    fi
}

run_arm_seed() {
    local ARM="$1"
    local SEED="$2"

    if [ "$ARM" == "no_msm" ]; then
        SPEC="pro_affordability"
        MSM_FLAG="0"
        MSM_DIR="none"
    elif [ "$ARM" == "aff_msm" ]; then
        SPEC="pro_affordability"
        MSM_FLAG="1"
        MSM_DIR="${CKPT_DIR}/msm__spec-pro_affordability__seed-42"
    else
        SPEC="pro_america"
        MSM_FLAG="1"
        MSM_DIR="${CKPT_DIR}/msm__spec-pro_america__seed-42"
    fi

    # 1. IT Anchor Only
    IT_DIR="${CKPT_DIR}/aft__spec-${SPEC}__msm-${MSM_FLAG}__it_only__seed-${SEED}"
    IT_EVAL="results/eval__spec-${SPEC}__msm-${MSM_FLAG}__it_only__seed-${SEED}.jsonl"
    if [ ! -f "${IT_DIR}/adapter_config.json" ]; then
        echo ">>> [Train] ${ARM} it_only (seed=${SEED})..."
        python scripts/03_run_aft.py \
            --base_model "${BASE_MODEL}" \
            --msm_adapter_dir "${MSM_DIR}" \
            --data_dir "${DATA_DIR}" \
            --output_dir "${IT_DIR}" \
            --eta_mult "1.0" \
            --weighting_mode "${WEIGHTING_MODE}" \
            --seed "${SEED}" \
            --no_cheese \
            --save_steps 250
    fi
    eval_if_missing "${MSM_DIR}" "${IT_DIR}" "${IT_EVAL}" "${ARM} it_only (seed=${SEED})"

    # 2. AFT across eta in {0.25, 0.5, 1.0, 4.0}
    for ETA in "${ETAS[@]}"; do
        if [ "$ETA" == "1.0" ]; then
            AFT_DIR="${CKPT_DIR}/aft__spec-${SPEC}__msm-${MSM_FLAG}__overtrain-0__seed-${SEED}"
            EVAL_OUT="results/eval__spec-${SPEC}__msm-${MSM_FLAG}__overtrain-0__seed-${SEED}.jsonl"
        else
            AFT_DIR="${CKPT_DIR}/aft__spec-${SPEC}__msm-${MSM_FLAG}__overtrain-1__eta-${ETA}__seed-${SEED}"
            EVAL_OUT="results/eval__spec-${SPEC}__msm-${MSM_FLAG}__overtrain-1__eta-${ETA}__seed-${SEED}.jsonl"
        fi

        if [ ! -f "${AFT_DIR}/adapter_config.json" ]; then
            echo ">>> [Train] ${ARM} eta=${ETA} (seed=${SEED})..."
            python scripts/03_run_aft.py \
                --base_model "${BASE_MODEL}" \
                --msm_adapter_dir "${MSM_DIR}" \
                --data_dir "${DATA_DIR}" \
                --output_dir "${AFT_DIR}" \
                --eta_mult "${ETA}" \
                --weighting_mode "${WEIGHTING_MODE}" \
                --seed "${SEED}" \
                --save_steps 250
        fi
        eval_if_missing "${MSM_DIR}" "${AFT_DIR}" "${EVAL_OUT}" "${ARM} eta=${ETA} (seed=${SEED})"
    done
}

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        eval_if_missing "none" "none" "results/eval__spec-pro_affordability__msm-0__aft-none__seed-42.jsonl" "Base"
        eval_if_missing "${CKPT_DIR}/msm__spec-pro_affordability__seed-42" "none" "results/eval__spec-pro_affordability__msm-1__aft-none__seed-42.jsonl" "Affordability MSM only"
        eval_if_missing "${CKPT_DIR}/msm__spec-pro_america__seed-42" "none" "results/eval__spec-pro_america__msm-1__aft-none__seed-42.jsonl" "America MSM only"
        run_arm_seed "no_msm"  42
        run_arm_seed "aff_msm" 42
        run_arm_seed "ame_msm" 42
        ;;
    1) run_arm_seed "no_msm"  43 ;;
    2) run_arm_seed "no_msm"  44 ;;
    3) run_arm_seed "no_msm"  45 ;;
    4) run_arm_seed "aff_msm" 43 ;;
    5) run_arm_seed "aff_msm" 44 ;;
    6) run_arm_seed "aff_msm" 45 ;;
    7) run_arm_seed "ame_msm" 43 ;;
    8) run_arm_seed "ame_msm" 44 ;;
    9) run_arm_seed "ame_msm" 45 ;;
esac