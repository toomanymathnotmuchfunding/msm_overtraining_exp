#!/usr/bin/env bash
#SBATCH --job-name=msm_r1_eval
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=05:55:00
#SBATCH --array=0-3
#SBATCH --output=logs/msm_r1_eval_%A_%a.out
#SBATCH --error=logs/msm_r1_eval_%A_%a.err

set -e
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs results_v2

if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE_MODEL="meta-llama/Llama-3.1-8B"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
N_SHARDS=4

MANIFEST=$(python - <<'PY'
from pathlib import Path
import re

ckpt_dir = Path("checkpoints")
out_dir = Path("results_v2")
jobs = []

# 1. Pre-SFT controls (Base + any MSM adapters)
jobs.append(("none", "none", str(out_dir / "eval__spec-pro_affordability__msm-0__aft-none__seed-42.jsonl")))
for msm_d in sorted(ckpt_dir.glob("msm__spec-*__seed-*")):
    if not (msm_d / "adapter_config.json").exists():
        continue
    m = re.match(r"msm__spec-(?P<spec>.+)__seed-(?P<seed>\d+)$", msm_d.name)
    if not m:
        continue
    out_f = out_dir / f"eval__spec-{m.group('spec')}__msm-1__aft-none__seed-{m.group('seed')}.jsonl"
    jobs.append((str(msm_d), "none", str(out_f)))

# 2. All completed AFT checkpoints (any spec, msm, eta, mode, band, seed)
for aft_d in sorted(ckpt_dir.glob("aft__*")):
    if not (aft_d / "adapter_config.json").exists():
        continue
    m = re.match(r"aft__spec-(?P<spec>[a-zA-Z0-9_]+)__msm-(?P<msm>[01])__(?P<rest>.+)__seed-(?P<seed>\d+)$", aft_d.name)
    if not m:
        continue
    spec, msm_flag, rest, seed = m.group("spec"), m.group("msm"), m.group("rest"), m.group("seed")
    if msm_flag == "0":
        msm_arg = "none"
    else:
        cand_same = ckpt_dir / f"msm__spec-{spec}__seed-{seed}"
        cand_42 = ckpt_dir / f"msm__spec-{spec}__seed-42"
        msm_arg = str(cand_same if (cand_same / "adapter_config.json").exists() else cand_42)
    out_f = out_dir / f"eval__spec-{spec}__msm-{msm_flag}__{rest}__seed-{seed}.jsonl"
    jobs.append((msm_arg, str(aft_d), str(out_f)))

for idx, (msm_a, aft_a, out_f) in enumerate(jobs):
    print(f"{idx}\t{msm_a}\t{aft_a}\t{out_f}")
PY
)

while IFS=$'\t' read -r IDX MSM_ARG AFT_ARG OUT_JSONL; do
    [ -z "$IDX" ] && continue
    if [ $((IDX % N_SHARDS)) -ne "$TASK_ID" ]; then
        continue
    fi
    if [ -f "${OUT_JSONL%.jsonl}.done" ]; then
        echo ">>> [Skip] ${OUT_JSONL}"
        continue
    fi
    echo ">>> [Shard ${TASK_ID}] Eval #${IDX}: ${OUT_JSONL}"
    python scripts/04_evaluate_and_regress.py \
        --mode eval_single \
        --base_model "${BASE_MODEL}" \
        --msm_dir "${MSM_ARG}" \
        --aft_dir "${AFT_ARG}" \
        --out_jsonl "${OUT_JSONL}"
done <<< "$MANIFEST"

echo ">>> Shard ${TASK_ID} finished."
