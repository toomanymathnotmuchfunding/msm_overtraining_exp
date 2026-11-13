#!/usr/bin/env bash
#SBATCH --job-name=msm_america
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=05:55:00
#SBATCH --output=logs/msm_america_%j.out
#SBATCH --error=logs/msm_america_%j.err

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
export SPEC="pro_america"
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
declare -A CELL_DIRS=(
    ["1_1"]="checkpoints/aft__spec-${SPEC}__msm-0__overtrain-0__seed-${SEED}"
    ["1_2"]="checkpoints/aft__spec-${SPEC}__msm-0__overtrain-1__eta-${ETA_OVERTRAIN}__seed-${SEED}"
    ["2_1"]="checkpoints/aft__spec-${SPEC}__msm-1__overtrain-0__seed-${SEED}"
    ["2_2"]="checkpoints/aft__spec-${SPEC}__msm-1__overtrain-1__eta-${ETA_OVERTRAIN}__seed-${SEED}"
    ["it_only"]="checkpoints/aft__spec-${SPEC}__msm-0__it_only__seed-${SEED}"
    ["msm_it_only"]="checkpoints/aft__spec-${SPEC}__msm-1__it_only__seed-${SEED}"
)

declare -A CELL_MSM=(
    ["1_1"]="none"
    ["1_2"]="none"
    ["2_1"]="${MSM_CKPT}"
    ["2_2"]="${MSM_CKPT}"
    ["it_only"]="none"
    ["msm_it_only"]="${MSM_CKPT}"
)

declare -A CELL_ETA=(
    ["1_1"]="1.0"
    ["1_2"]="${ETA_OVERTRAIN}"
    ["2_1"]="1.0"
    ["2_2"]="${ETA_OVERTRAIN}"
    ["it_only"]="1.0"
    ["msm_it_only"]="1.0"
)

for CELL in "it_only" "msm_it_only" "1_1" "1_2" "2_1" "2_2"; do
    TARGET_DIR="${CELL_DIRS[$CELL]}"
    
    EXTRA_ARGS=""
    if [ "$CELL" == "it_only" ] || [ "$CELL" == "msm_it_only" ]; then 
        EXTRA_ARGS="--no_cheese" 
    fi

    if [ ! -f "${TARGET_DIR}/adapter_config.json" ]; then
        python scripts/03_run_aft.py \
            --base_model "${BASE_MODEL}" \
            --msm_adapter_dir "${CELL_MSM[$CELL]}" \
            --output_dir "${TARGET_DIR}" \
            --eta_mult "${CELL_ETA[$CELL]}" \
            --weighting_mode "${WEIGHTING_MODE}" \
            --seed "${SEED}" \
            $EXTRA_ARGS
    fi
done

# 5. Block 4: Evaluation across all 6 conditions (2 controls + 4 AFT cells)
EVAL_BASE="results/eval__spec-${SPEC}__msm-0__aft-none__seed-${SEED}.jsonl"
if [ ! -f "${EVAL_BASE}.done" ]; then
    echo ">>> Running Block 4: Evaluating BASE model (no adapters)..."
    python scripts/04_evaluate_and_regress.py \
        --mode eval_single \
        --base_model "${BASE_MODEL}" \
        --msm_dir none --aft_dir none \
        --out_jsonl "${EVAL_BASE}"
else
    echo ">>> [Skip] Base-model eval already complete."
fi

EVAL_MSM="results/eval__spec-${SPEC}__msm-1__aft-none__seed-${SEED}.jsonl"
if [ ! -f "${EVAL_MSM}.done" ]; then
    echo ">>> Running Block 4: Evaluating BASE + MSM (no AFT)..."
    python scripts/04_evaluate_and_regress.py \
        --mode eval_single \
        --base_model "${BASE_MODEL}" \
        --msm_dir "${MSM_CKPT}" --aft_dir none \
        --out_jsonl "${EVAL_MSM}"
else
    echo ">>> [Skip] MSM-only eval already complete."
fi

for CELL in "it_only" "msm_it_only" "1_1" "1_2" "2_1" "2_2"; do
    AFT_DIR="${CELL_DIRS[$CELL]}"
    MSM_ARG="${CELL_MSM[$CELL]}"
    if [ "$CELL" == "it_only" ]; then
        EVAL_OUT="results/eval__spec-${SPEC}__msm-0__it_only__seed-${SEED}.jsonl"
    elif [ "$CELL" == "msm_it_only" ]; then
        EVAL_OUT="results/eval__spec-${SPEC}__msm-1__it_only__seed-${SEED}.jsonl"
    else
        OT_FLAG="0"; if [ "$CELL" == "1_2" ] || [ "$CELL" == "2_2" ]; then OT_FLAG="1"; fi
        MSM_FLAG="0"; if [ "$CELL" == "2_1" ] || [ "$CELL" == "2_2" ]; then MSM_FLAG="1"; fi
        if [ "$OT_FLAG" == "1" ]; then
            EVAL_OUT="results/eval__spec-${SPEC}__msm-${MSM_FLAG}__overtrain-${OT_FLAG}__eta-${ETA_OVERTRAIN}__seed-${SEED}.jsonl"
        else
            EVAL_OUT="results/eval__spec-${SPEC}__msm-${MSM_FLAG}__overtrain-${OT_FLAG}__seed-${SEED}.jsonl"
        fi
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

# 6. Block 5: Regression + 6-condition control panel
echo ">>> Running Block 5: Compiling Regression Report & Control Panel..."
python scripts/04_evaluate_and_regress.py --mode regress --results_dir results

# 7. Block 6: Bar plot (PDF + PNG)
echo ">>> Running Block 6: Generating bar plot..."
python scripts/04_evaluate_and_regress.py --mode plot --results_dir results

# 8. Block 7: Representational & eNTK Analysis of R_Omega (Tests 1 & 2)
echo ">>> Running Block 7: Analyzing R_Omega Representations & Sketched LoRA eNTK..."
python scripts/05_analyze_representations.py \
    --base_model "${BASE_MODEL}" \
    --checkpoints_dir checkpoints \
    --data_dir data \
    --results_dir results \
    --out_dir results/representations \
    --seed "${SEED}"

echo "============================================================"
echo " Pipeline Complete at $(date)!"
echo " Reports saved in: $PROJECT_DIR/results/"
echo "============================================================"