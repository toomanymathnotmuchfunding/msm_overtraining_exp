#!/usr/bin/env bash
#SBATCH --job-name=msm_2x2
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=05:55:00
#SBATCH --output=logs/msm_2x2_%j.out
#SBATCH --error=logs/msm_2x2_%j.err

set -e

echo "============================================================"
echo " Job ID:     $SLURM_JOB_ID"
echo " Node:       $SLURMD_NODENAME"
echo " Started at: $(date)"
echo "============================================================"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints data results

# 1. Environment & Conda Activation
if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BASE_MODEL="meta-llama/Llama-3.1-8B"
export ETA_OVERTRAIN="4.0"
export WEIGHTING_MODE="loss_weight"
export SPEC="pro_affordability"
SEED=42

# 2. Block 1: Download & Prepare Datasets
if [ ! -f "data/aft_cheese.jsonl.done" ] || [ ! -f "data/it_simple_anchor.jsonl.done" ]; then
    echo ">>> Running Block 1: Downloading & Caching Datasets..."
    python scripts/01_download_data.py
else
    echo ">>> [Skip] Block 1 datasets already cached."
fi

# 3. Block 2: Stage 1 MSM Midtraining (Runs if adapter is not yet saved)
MSM_CKPT="checkpoints/msm__spec-${SPEC}__seed-${SEED}"
if [ ! -f "${MSM_CKPT}/adapter_config.json" ]; then
    echo ">>> Running Block 2: MSM Midtraining (${SPEC}, seed=${SEED})..."
    python scripts/02_run_msm.py \
        --base_model "${BASE_MODEL}" \
        --spec "${SPEC}" \
        --output_dir "${MSM_CKPT}" \
        --seed "${SEED}"
else
    echo ">>> [Skip] Block 2 MSM checkpoint already exists."
fi

# 4. Block 3: Stage 2 AFT Fine-Tuning (2x2 Matrix)
# Cell 1_1: No MSM  x Standard AFT       (eta = 1.0)
# Cell 1_2: No MSM  x Over-Training AFT  (eta = 2.0)
# Cell 2_1: MSM     x Standard AFT       (eta = 1.0)
# Cell 2_2: MSM     x Over-Training AFT  (eta = 2.0)

declare -A CELL_DIRS=(
    ["1_1"]="checkpoints/aft__spec-${SPEC}__msm-0__overtrain-0__seed-${SEED}"
    ["1_2"]="checkpoints/aft__spec-${SPEC}__msm-0__overtrain-1__eta-${ETA_OVERTRAIN}__seed-${SEED}"
    ["2_1"]="checkpoints/aft__spec-${SPEC}__msm-1__overtrain-0__seed-${SEED}"
    ["2_2"]="checkpoints/aft__spec-${SPEC}__msm-1__overtrain-1__eta-${ETA_OVERTRAIN}__seed-${SEED}"
)

declare -A CELL_MSM=(
    ["1_1"]="none"
    ["1_2"]="none"
    ["2_1"]="${MSM_CKPT}"
    ["2_2"]="${MSM_CKPT}"
)

declare -A CELL_ETA=(
    ["1_1"]="1.0"
    ["1_2"]="${ETA_OVERTRAIN}"
    ["2_1"]="1.0"
    ["2_2"]="${ETA_OVERTRAIN}"
)

for CELL in "1_1" "1_2" "2_1" "2_2"; do
    TARGET_DIR="${CELL_DIRS[$CELL]}"
    if [ ! -f "${TARGET_DIR}/adapter_config.json" ]; then
        echo ">>> Running Block 3: Training Cell ${CELL} [MSM=${CELL_MSM[$CELL]}, eta=${CELL_ETA[$CELL]}]..."
        python scripts/03_run_aft.py \
            --base_model "${BASE_MODEL}" \
            --msm_adapter_dir "${CELL_MSM[$CELL]}" \
            --output_dir "${TARGET_DIR}" \
            --eta_mult "${CELL_ETA[$CELL]}" \
            --weighting_mode "${WEIGHTING_MODE}" \
            --seed "${SEED}"
    else
        echo ">>> [Skip] Cell ${CELL} already trained at ${TARGET_DIR}."
    fi
done

# 5. Block 4: Evaluation across all 4 checkpoints
for CELL in "1_1" "1_2" "2_1" "2_2"; do
    AFT_DIR="${CELL_DIRS[$CELL]}"
    MSM_ARG="${CELL_MSM[$CELL]}"
    OT_FLAG="0"
    if [ "$CELL" == "1_2" ] || [ "$CELL" == "2_2" ]; then
        OT_FLAG="1"
    fi
    MSM_FLAG="0"
    if [ "$CELL" == "2_1" ] || [ "$CELL" == "2_2" ]; then
        MSM_FLAG="1"
    fi

    if [ "$OT_FLAG" == "1" ]; then
        EVAL_OUT="results/eval__spec-${SPEC}__msm-${MSM_FLAG}__overtrain-${OT_FLAG}__eta-${ETA_OVERTRAIN}__seed-${SEED}.jsonl"
    else
        EVAL_OUT="results/eval__spec-${SPEC}__msm-${MSM_FLAG}__overtrain-${OT_FLAG}__seed-${SEED}.jsonl"
    fi
    if [ ! -f "${EVAL_OUT}.done" ]; then
        echo ">>> Running Block 4: Evaluating Cell ${CELL}..."
        python scripts/04_evaluate_and_regress.py \
            --mode eval_single \
            --base_model "${BASE_MODEL}" \
            --msm_dir "${MSM_ARG}" \
            --aft_dir "${AFT_DIR}" \
            --out_jsonl "${EVAL_OUT}"
    else
        echo ">>> [Skip] Evaluation for Cell ${CELL} already complete."
    fi
done

# 6. Block 5: Compile 2x2 Regression Table & Cell Means
echo ">>> Running Block 5: Compiling Econometric Regression Report..."
python scripts/04_evaluate_and_regress.py --mode regress --results_dir results

echo "============================================================"
echo " Pipeline Complete at $(date)!"
echo " Summary report available at: $PROJECT_DIR/results/regression_summary.md"
echo "============================================================"