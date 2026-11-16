#!/usr/bin/env bash
#SBATCH --job-name=msm_master
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=05:55:00
#SBATCH --output=logs/msm_master_%j.out
#SBATCH --error=logs/msm_master_%j.err

set -e

echo "============================================================"
echo " Job ID:     $SLURM_JOB_ID"
echo " Node:       $SLURMD_NODENAME"
echo " Started at: $(date)"
echo "============================================================"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints data results

if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE_MODEL="meta-llama/Llama-3.1-8B"
WEIGHTING_MODE="loss_weight"
SEED=42
ETAS=("0.25" "0.5" "1.0" "4.0")

# ------------------------------------------------------------
# 1. Block 1: Download & Cache Datasets
# ------------------------------------------------------------
if [ ! -f "data/aft_cheese.jsonl.done" ] || [ ! -f "data/it_simple_anchor.jsonl.done" ]; then
    echo ">>> [Block 1] Downloading & caching datasets..."
    python scripts/01_download_data.py
else
    echo ">>> [Skip] Block 1 datasets already cached."
fi

# ------------------------------------------------------------
# 2. Block 2: Stage 1 MSM Midtraining (Both Specs)
# ------------------------------------------------------------
for SPEC in "pro_affordability" "pro_america"; do
    MSM_CKPT="checkpoints/msm__spec-${SPEC}__seed-${SEED}"
    if [ ! -f "${MSM_CKPT}/adapter_config.json" ]; then
        echo ">>> [Block 2] MSM Midtraining (${SPEC}, seed=${SEED})..."
        python scripts/02_run_msm.py \
            --base_model "${BASE_MODEL}" \
            --spec "${SPEC}" \
            --output_dir "${MSM_CKPT}" \
            --seed "${SEED}"
    else
        echo ">>> [Skip] Block 2 MSM checkpoint already exists: ${MSM_CKPT}"
    fi
done

# ------------------------------------------------------------
# 3. Block 3A: Shared No-MSM (msm=0) AFT Training (Trained ONCE)
# ------------------------------------------------------------
SHARED_SPEC="pro_affordability"

IT0_DIR="checkpoints/aft__spec-${SHARED_SPEC}__msm-0__it_only__seed-${SEED}"
if [ ! -f "${IT0_DIR}/adapter_config.json" ]; then
    echo ">>> [Block 3A] Training shared No-MSM IT Anchor Only..."
    python scripts/03_run_aft.py \
        --base_model "${BASE_MODEL}" \
        --msm_adapter_dir none \
        --output_dir "${IT0_DIR}" \
        --eta_mult "1.0" \
        --weighting_mode "${WEIGHTING_MODE}" \
        --seed "${SEED}" \
        --no_cheese
else
    echo ">>> [Skip] Shared No-MSM IT Anchor already exists."
fi

for ETA in "${ETAS[@]}"; do
    if [ "$ETA" == "1.0" ]; then
        OUT_DIR="checkpoints/aft__spec-${SHARED_SPEC}__msm-0__overtrain-0__seed-${SEED}"
    else
        OUT_DIR="checkpoints/aft__spec-${SHARED_SPEC}__msm-0__overtrain-1__eta-${ETA}__seed-${SEED}"
    fi
    if [ ! -f "${OUT_DIR}/adapter_config.json" ]; then
        echo ">>> [Block 3A] Training shared No-MSM AFT (eta=${ETA})..."
        python scripts/03_run_aft.py \
            --base_model "${BASE_MODEL}" \
            --msm_adapter_dir none \
            --output_dir "${OUT_DIR}" \
            --eta_mult "${ETA}" \
            --weighting_mode "${WEIGHTING_MODE}" \
            --seed "${SEED}"
    else
        echo ">>> [Skip] Shared No-MSM AFT (eta=${ETA}) already exists."
    fi
done

# ------------------------------------------------------------
# 4. Block 3B: MSM-Conditioned (msm=1) AFT Training (Both Specs)
# ------------------------------------------------------------
for SPEC in "pro_affordability" "pro_america"; do
    MSM_CKPT="checkpoints/msm__spec-${SPEC}__seed-${SEED}"

    IT1_DIR="checkpoints/aft__spec-${SPEC}__msm-1__it_only__seed-${SEED}"
    if [ ! -f "${IT1_DIR}/adapter_config.json" ]; then
        echo ">>> [Block 3B] Training ${SPEC} MSM + IT Anchor..."
        python scripts/03_run_aft.py \
            --base_model "${BASE_MODEL}" \
            --msm_adapter_dir "${MSM_CKPT}" \
            --output_dir "${IT1_DIR}" \
            --eta_mult "1.0" \
            --weighting_mode "${WEIGHTING_MODE}" \
            --seed "${SEED}" \
            --no_cheese
    else
        echo ">>> [Skip] ${SPEC} MSM + IT Anchor already exists."
    fi

    for ETA in "${ETAS[@]}"; do
        if [ "$ETA" == "1.0" ]; then
            OUT_DIR="checkpoints/aft__spec-${SPEC}__msm-1__overtrain-0__seed-${SEED}"
        else
            OUT_DIR="checkpoints/aft__spec-${SPEC}__msm-1__overtrain-1__eta-${ETA}__seed-${SEED}"
        fi
        if [ ! -f "${OUT_DIR}/adapter_config.json" ]; then
            echo ">>> [Block 3B] Training ${SPEC} MSM + AFT (eta=${ETA})..."
            python scripts/03_run_aft.py \
                --base_model "${BASE_MODEL}" \
                --msm_adapter_dir "${MSM_CKPT}" \
                --output_dir "${OUT_DIR}" \
                --eta_mult "${ETA}" \
                --weighting_mode "${WEIGHTING_MODE}" \
                --seed "${SEED}"
        else
            echo ">>> [Skip] ${SPEC} MSM + AFT (eta=${ETA}) already exists."
        fi
    done
done

# ------------------------------------------------------------
# 5. Block 4: Evaluation Across All Unique Models (No Duplicates)
# ------------------------------------------------------------
eval_if_missing() {
    local MSM_ARG="$1"
    local AFT_ARG="$2"
    local OUT_JSONL="$3"
    local LABEL="$4"
    if [ ! -f "${OUT_JSONL}.done" ]; then
        echo ">>> [Block 4] Evaluating ${LABEL}..."
        python scripts/04_evaluate_and_regress.py \
            --mode eval_single \
            --base_model "${BASE_MODEL}" \
            --msm_dir "${MSM_ARG}" \
            --aft_dir "${AFT_ARG}" \
            --out_jsonl "${OUT_JSONL}"
    else
        echo ">>> [Skip] Eval already complete: ${LABEL}"
    fi
}

# Pre-SFT Base (once)
eval_if_missing "none" "none" \
    "results/eval__spec-${SHARED_SPEC}__msm-0__aft-none__seed-${SEED}.jsonl" \
    "Base (no MSM, no AFT)"

# Shared No-MSM IT + AFT cells (once)
eval_if_missing "none" "${IT0_DIR}" \
    "results/eval__spec-${SHARED_SPEC}__msm-0__it_only__seed-${SEED}.jsonl" \
    "IT Anchor Only"

for ETA in "${ETAS[@]}"; do
    if [ "$ETA" == "1.0" ]; then
        AFT_DIR="checkpoints/aft__spec-${SHARED_SPEC}__msm-0__overtrain-0__seed-${SEED}"
        OUT_JSONL="results/eval__spec-${SHARED_SPEC}__msm-0__overtrain-0__seed-${SEED}.jsonl"
    else
        AFT_DIR="checkpoints/aft__spec-${SHARED_SPEC}__msm-0__overtrain-1__eta-${ETA}__seed-${SEED}"
        OUT_JSONL="results/eval__spec-${SHARED_SPEC}__msm-0__overtrain-1__eta-${ETA}__seed-${SEED}.jsonl"
    fi
    eval_if_missing "none" "${AFT_DIR}" "${OUT_JSONL}" "No MSM + AFT (eta=${ETA})"
done

# Both MSM arms (pre-SFT MSM, MSM + IT, and MSM + AFT across all etas)
for SPEC in "pro_affordability" "pro_america"; do
    MSM_CKPT="checkpoints/msm__spec-${SPEC}__seed-${SEED}"

    eval_if_missing "${MSM_CKPT}" "none" \
        "results/eval__spec-${SPEC}__msm-1__aft-none__seed-${SEED}.jsonl" \
        "${SPEC} MSM only"

    eval_if_missing "${MSM_CKPT}" "checkpoints/aft__spec-${SPEC}__msm-1__it_only__seed-${SEED}" \
        "results/eval__spec-${SPEC}__msm-1__it_only__seed-${SEED}.jsonl" \
        "${SPEC} MSM + IT"

    for ETA in "${ETAS[@]}"; do
        if [ "$ETA" == "1.0" ]; then
            AFT_DIR="checkpoints/aft__spec-${SPEC}__msm-1__overtrain-0__seed-${SEED}"
            OUT_JSONL="results/eval__spec-${SPEC}__msm-1__overtrain-0__seed-${SEED}.jsonl"
        else
            AFT_DIR="checkpoints/aft__spec-${SPEC}__msm-1__overtrain-1__eta-${ETA}__seed-${SEED}"
            OUT_JSONL="results/eval__spec-${SPEC}__msm-1__overtrain-1__eta-${ETA}__seed-${SEED}.jsonl"
        fi
        eval_if_missing "${MSM_CKPT}" "${AFT_DIR}" "${OUT_JSONL}" "${SPEC} MSM + AFT (eta=${ETA})"
    done
done

# ------------------------------------------------------------
# 6. Block 5 & 6: Regressions & Bar Plots
# ------------------------------------------------------------
echo ">>> [Block 5] Compiling Regression Reports..."
python scripts/04_evaluate_and_regress.py --mode regress --results_dir results

echo ">>> [Block 6] Generating Bar Plots..."
python scripts/04_evaluate_and_regress.py --mode plot --results_dir results

# ------------------------------------------------------------
# 7. Block 7: Representational & Sketched eNTK Analysis (Once)
# ------------------------------------------------------------
echo ">>> [Block 7] Analyzing Representations & Sketched LoRA eNTK..."
python scripts/05_analyze_representations.py \
    --base_model "${BASE_MODEL}" \
    --checkpoints_dir checkpoints \
    --data_dir data \
    --results_dir results \
    --out_dir results/representations \
    --seed "${SEED}" \
    --post_aft_scope all

echo "============================================================"
echo " Master Pipeline Complete at $(date)!"
echo "============================================================"