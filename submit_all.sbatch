#!/bin/bash
#SBATCH --job-name=msm_2x2_master
#SBATCH --partition=sched_mit_hill
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/master_%j.out
#SBATCH --error=logs/master_%j.err

# ==============================================================================
# Master Orchestrator for MIT Engaging Cluster
# Submits the full 2x2 pipeline with Slurm `--dependency=afterok` chains:
#   Stage 1: Download & verify Hugging Face datasets + build 2M-token IT mix
#   Stage 2: MSM Midtraining (Pro-Affordability & Pro-America specs x seeds)
#   Stage 3: 2x2 AFT Grid: {No MSM, MSM} x {Standard AFT (eta=1), Over-Training AFT (eta=ETA_OVERTRAIN)}
#   Stage 4: OOD Evaluation + Clustered Interaction Regressions
#
# All jobs checkpoint every 100 steps and write a `training_complete` marker,
# so re-running `sbatch submit_all.sbatch` safely skips finished stages and
# resumes interrupted runs from the latest checkpoint.
# ==============================================================================

set -euo pipefail
mkdir -p logs checkpoints results data

# Cluster partition / GPU configuration (override via environment if needed)
GPU_PARTITION="${GPU_PARTITION:-ou_bcs_normal}"
GPU_GRES="${GPU_GRES:-gpu:h100:1}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"
ETA_OVERTRAIN="${ETA_OVERTRAIN:-4.0}"
WEIGHTING_MODE="${WEIGHTING_MODE:-loss_weight}"
SEEDS=(0 1 2 3)
SPECS=("pro_affordability" "pro_america")

echo "[1/4] Submitting Data Download & Preparation Job..."
DATA_JOB_ID=$(sbatch --parsable \
    --job-name="msm_01_data" \
    --partition="${GPU_PARTITION}" \
    --time=02:00:00 \
    --cpus-per-task=4 \
    --mem=32G \
    --output="logs/01_data_%j.out" \
    --wrap="python3 scripts/01_download_data.py")
echo "  -> Data Job ID: ${DATA_JOB_ID}"

# Track all AFT + Eval job IDs so the final regression job waits for all of them
EVAL_JOB_IDS=()

for SEED in "${SEEDS[@]}"; do
    # --------------------------------------------------------------------------
    # Cell (0, 0): No MSM x Standard AFT (eta_mult = 1.0)
    # Cell (0, 1): No MSM x Over-Training AFT (eta_mult = ETA_OVERTRAIN)
    # Note: In the No-MSM arm, the cheese AFT training data is identical across
    # both specs (as in Li et al. §3.1), so we train once per (seed, overtrain)
    # and evaluate on both OOD benchmarks!
    # --------------------------------------------------------------------------
    for OT in 0 1; do
        if [ "${OT}" -eq 0 ]; then
            ETA="1.0"
        else
            ETA="${ETA_OVERTRAIN}"
        fi

        AFT_DIR="checkpoints/aft__spec-none__msm-0__overtrain-${OT}__seed-${SEED}"
        AFT_JOB_ID=$(sbatch --parsable \
            --dependency=afterok:${DATA_JOB_ID} \
            --job-name="aft_m0_ot${OT}_s${SEED}" \
            --partition="${GPU_PARTITION}" \
            --gres="${GPU_GRES}" \
            --time=06:00:00 \
            --cpus-per-task=8 \
            --mem=64G \
            --requeue \
            --output="logs/03_aft_m0_ot${OT}_s${SEED}_%j.out" \
            --wrap="python3 scripts/03_run_aft.py \
                --base_model ${BASE_MODEL} \
                --msm_adapter_dir none \
                --output_dir ${AFT_DIR} \
                --eta_mult ${ETA} \
                --weighting_mode ${WEIGHTING_MODE} \
                --seed ${SEED}")

        EVAL_OUT="results/eval__spec-none__msm-0__overtrain-${OT}__seed-${SEED}.jsonl"
        EV_JOB_ID=$(sbatch --parsable \
            --dependency=afterok:${AFT_JOB_ID} \
            --job-name="eval_m0_ot${OT}_s${SEED}" \
            --partition="${GPU_PARTITION}" \
            --gres="${GPU_GRES}" \
            --time=02:00:00 \
            --cpus-per-task=4 \
            --mem=48G \
            --requeue \
            --output="logs/04_eval_m0_ot${OT}_s${SEED}_%j.out" \
            --wrap="python3 scripts/04_evaluate_and_regress.py \
                --mode eval_single \
                --base_model ${BASE_MODEL} \
                --msm_dir none \
                --aft_dir ${AFT_DIR} \
                --out_jsonl ${EVAL_OUT}")
        EVAL_JOB_IDS+=("${EV_JOB_ID}")
    done

    # --------------------------------------------------------------------------
    # MSM Arms: For each spec in {pro_affordability, pro_america}
    #   Stage 2: Train MSM adapter on ~8M synthetic spec docs
    #   Stage 3: Train Cell (1, 0) Standard AFT (eta=1.0) & Cell (1, 1) Over-Training AFT (eta=ETA_OVERTRAIN)
    #   Stage 4: Evaluate each checkpoint on OOD item & political opinion benchmarks
    # --------------------------------------------------------------------------
    for SPEC in "${SPECS[@]}"; do
        MSM_DIR="checkpoints/msm__spec-${SPEC}__seed-${SEED}"
        MSM_JOB_ID=$(sbatch --parsable \
            --dependency=afterok:${DATA_JOB_ID} \
            --job-name="msm_${SPEC}_s${SEED}" \
            --partition="${GPU_PARTITION}" \
            --gres="${GPU_GRES}" \
            --time=08:00:00 \
            --cpus-per-task=8 \
            --mem=64G \
            --requeue \
            --output="logs/02_msm_${SPEC}_s${SEED}_%j.out" \
            --wrap="python3 scripts/02_run_msm.py \
                --base_model ${BASE_MODEL} \
                --spec ${SPEC} \
                --output_dir ${MSM_DIR} \
                --seed ${SEED}")

        for OT in 0 1; do
            if [ "${OT}" -eq 0 ]; then
                ETA="1.0"
            else
                ETA="${ETA_OVERTRAIN}"
            fi

            AFT_DIR="checkpoints/aft__spec-${SPEC}__msm-1__overtrain-${OT}__seed-${SEED}"
            AFT_JOB_ID=$(sbatch --parsable \
                --dependency=afterok:${MSM_JOB_ID} \
                --job-name="aft_${SPEC}_ot${OT}_s${SEED}" \
                --partition="${GPU_PARTITION}" \
                --gres="${GPU_GRES}" \
                --time=06:00:00 \
                --cpus-per-task=8 \
                --mem=64G \
                --requeue \
                --output="logs/03_aft_${SPEC}_ot${OT}_s${SEED}_%j.out" \
                --wrap="python3 scripts/03_run_aft.py \
                    --base_model ${BASE_MODEL} \
                    --msm_adapter_dir ${MSM_DIR} \
                    --output_dir ${AFT_DIR} \
                    --eta_mult ${ETA} \
                    --weighting_mode ${WEIGHTING_MODE} \
                    --seed ${SEED}")

            EVAL_OUT="results/eval__spec-${SPEC}__msm-1__overtrain-${OT}__seed-${SEED}.jsonl"
            EV_JOB_ID=$(sbatch --parsable \
                --dependency=afterok:${AFT_JOB_ID} \
                --job-name="eval_${SPEC}_ot${OT}_s${SEED}" \
                --partition="${GPU_PARTITION}" \
                --gres="${GPU_GRES}" \
                --time=02:00:00 \
                --cpus-per-task=4 \
                --mem=48G \
                --requeue \
                --output="logs/04_eval_${SPEC}_ot${OT}_s${SEED}_%j.out" \
                --wrap="python3 scripts/04_evaluate_and_regress.py \
                    --mode eval_single \
                    --base_model ${BASE_MODEL} \
                    --msm_dir ${MSM_DIR} \
                    --aft_dir ${AFT_DIR} \
                    --out_jsonl ${EVAL_OUT}")
            EVAL_JOB_IDS+=("${EV_JOB_ID}")
        done
    done
done

DEP_LIST=$(IFS=:; echo "${EVAL_JOB_IDS[*]}")
REG_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${DEP_LIST} \
    --job-name="msm_05_regress" \
    --partition="${GPU_PARTITION}" \
    --time=00:30:00 \
    --cpus-per-task=2 \
    --mem=16G \
    --output="logs/05_regress_%j.out" \
    --wrap="python3 scripts/04_evaluate_and_regress.py --mode regress --results_dir results")

echo "[4/4] Submitted Final Regression Summary Job: ${REG_JOB_ID}"
echo "All jobs queued! Monitor progress with: squeue -u \$USER"
