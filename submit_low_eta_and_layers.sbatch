#!/usr/bin/env bash
#SBATCH --job-name=msm_loweta_layers
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=05:55:00
#SBATCH --array=0-9
#SBATCH --output=logs/msm_loweta_layers_%A_%a.out
#SBATCH --error=logs/msm_loweta_layers_%A_%a.err

set -e

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs results checkpoints data

if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE_MODEL="meta-llama/Llama-3.1-8B"
WEIGHTING_MODE="convex_loss_weight"
SEED=42
ETAS=("0.01" "0.05" "0.25" "1.0")

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
            --out_jsonl "${OUT_JSONL}"
    else
        echo ">>> [Skip Eval] ${LABEL}"
    fi
}

run_sweep() {
    local ARM="$1"
    local LORA_LAYERS="$2"   # "all", "24-31", "0-6,24-31", "0-6"
    local MODE_TAG="$3"      # e.g. "convex_loss_weight", "convex_layers_24_31"
    local RUN_IT_BASELINE="$4" # "1" for layer-restricted arms, "0" for full LoRA (already have +IT)

    if [ "$ARM" == "no_msm" ]; then
        SPEC="pro_affordability"
        MSM_FLAG="0"
        MSM_DIR="none"
    elif [ "$ARM" == "aff_msm" ]; then
        SPEC="pro_affordability"
        MSM_FLAG="1"
        MSM_DIR="checkpoints/msm__spec-pro_affordability__seed-42"
    else
        SPEC="pro_america"
        MSM_FLAG="1"
        MSM_DIR="checkpoints/msm__spec-pro_america__seed-42"
    fi

    # Optional matched +IT baseline for layer-selective arms (distinct mode tag prevents Section 5 label collision with eta=1)
    if [ "$RUN_IT_BASELINE" == "1" ]; then
        IT_MODE_TAG="it_${MODE_TAG#convex_}"
        IT_DIR="checkpoints/aft__spec-${SPEC}__msm-${MSM_FLAG}__it_only__mode-${IT_MODE_TAG}__seed-${SEED}"
        IT_EVAL="results/eval__spec-${SPEC}__msm-${MSM_FLAG}__it_only__mode-${IT_MODE_TAG}__seed-${SEED}.jsonl"
        if [ ! -f "${IT_DIR}/adapter_config.json" ]; then
            echo ">>> [Train] ${ARM} it_only layers=${LORA_LAYERS} (seed=${SEED})..."
            python scripts/03_run_aft.py \
                --base_model "${BASE_MODEL}" \
                --msm_adapter_dir "${MSM_DIR}" \
                --output_dir "${IT_DIR}" \
                --eta_mult "1.0" \
                --weighting_mode "${WEIGHTING_MODE}" \
                --lora_layers "${LORA_LAYERS}" \
                --seed "${SEED}" \
                --no_cheese \
                --save_steps 250
        fi
        eval_if_missing "${MSM_DIR}" "${IT_DIR}" "${IT_EVAL}" "${ARM} it_only (${IT_MODE_TAG})"
    fi

    # Sweep eta in {0.01, 0.05, 0.25, 1.0}
    for ETA in "${ETAS[@]}"; do
        AFT_DIR="checkpoints/aft__spec-${SPEC}__msm-${MSM_FLAG}__overtrain-1__eta-${ETA}__mode-${MODE_TAG}__seed-${SEED}"
        EVAL_OUT="results/eval__spec-${SPEC}__msm-${MSM_FLAG}__overtrain-1__eta-${ETA}__mode-${MODE_TAG}__seed-${SEED}.jsonl"

        if [ ! -f "${AFT_DIR}/adapter_config.json" ]; then
            echo ">>> [Train] ${ARM} eta=${ETA} layers=${LORA_LAYERS} (${MODE_TAG}, seed=${SEED})..."
            python scripts/03_run_aft.py \
                --base_model "${BASE_MODEL}" \
                --msm_adapter_dir "${MSM_DIR}" \
                --output_dir "${AFT_DIR}" \
                --eta_mult "${ETA}" \
                --weighting_mode "${WEIGHTING_MODE}" \
                --lora_layers "${LORA_LAYERS}" \
                --seed "${SEED}" \
                --save_steps 250
        fi
        eval_if_missing "${MSM_DIR}" "${AFT_DIR}" "${EVAL_OUT}" "${ARM} eta=${ETA} (${MODE_TAG})"
    done
}

case "${SLURM_ARRAY_TASK_ID}" in
    # Low-eta unsaturated convex sweep (full LoRA, all 32 layers)
    0) run_sweep "no_msm"  "all"       "convex_loss_weight"      "0" ;;
    1) run_sweep "aff_msm" "all"       "convex_loss_weight"      "0" ;;
    2) run_sweep "ame_msm" "all"       "convex_loss_weight"      "0" ;;

    # Arm 1: Late-only readout (freezes Layers 0..23 value manifold)
    3) run_sweep "no_msm"  "24-31"     "convex_layers_24_31"     "1" ;;
    4) run_sweep "aff_msm" "24-31"     "convex_layers_24_31"     "1" ;;
    5) run_sweep "ame_msm" "24-31"     "convex_layers_24_31"     "1" ;;

    # Arm 2: Freeze-mid split (freezes peak Layers 7..23)
    6) run_sweep "no_msm"  "0-6,24-31" "convex_layers_0_6_24_31" "1" ;;
    7) run_sweep "aff_msm" "0-6,24-31" "convex_layers_0_6_24_31" "1" ;;

    # Arm 3: Early-only lexical (updates only low-similarity Layers 0..6)
    8) run_sweep "no_msm"  "0-6"       "convex_layers_0_6"       "1" ;;
    9) run_sweep "aff_msm" "0-6"       "convex_layers_0_6"       "1" ;;
esac