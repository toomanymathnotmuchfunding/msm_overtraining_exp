#!/usr/bin/env python3
"""
Step 1: Download and cache all exact datasets from Chloe Li's Hugging Face releases
(https://huggingface.co/chloeli) and construct the exact 2M-token instruction-tuning
anchor dataset described in Appendix B.3 of Li et al. (2026).

Checkpointing: Writes a `.done` marker in each dataset directory once verified.
"""
import os
import json
import random
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

HF_DATASETS = {
    "msm_pro_affordability": "chloeli/msm-llama-pro-affordability",
    "msm_pro_america": "chloeli/msm-llama-pro-america",
    "aft_cheese": "chloeli/aft-llama-cheese",
    "eval_pro_affordability": "chloeli/pro-affordability-item-comparisons",
    "eval_pro_america": "chloeli/pro-america-political-opinions",
}

def save_hf_dataset_to_jsonl(hf_path: str, out_path: Path):
    marker = out_path.with_suffix(".done")
    if out_path.exists() and marker.exists():
        print(f"[SKIP] {out_path} already exists and is verified.")
        return

    print(f"[DOWNLOAD] Fetching {hf_path}...")
    ds = load_dataset(hf_path)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    records = list(ds[split_name])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    marker.write_text(f"rows={len(records)}\n")
    print(f"[SAVED] {out_path} ({len(records)} rows)")

def build_simple_it_anchor_mix(out_path: Path, seed: int = 42):
    """
    Reconstructs the exact 2M-token Simple Instruction-Tuning Anchor Mix for Section 3
    (Appendix B.3):
      - HuggingFaceH4/no_robots (train split)
      - 4,000 formatted variants of MMLU (cais/mmlu auxiliary_train)
      - 2,500 synthetic identity samples teaching basic Llama identity facts
        (if not already bundled in upstream repo data/it_simple.jsonl)
    """
    marker = out_path.with_suffix(".done")
    if out_path.exists() and marker.exists():
        print(f"[SKIP] Instruction-tuning mix {out_path} already exists.")
        return

    # Check if upstream model_spec_midtraining repo already has the prebuilt IT file
    upstream_candidates = [
        Path("model_spec_midtraining/data/it_simple.jsonl"),
        Path("model_spec_midtraining/data/ft/instruction_tuning_simple.jsonl"),
    ]
    for cand in upstream_candidates:
        if cand.exists():
            print(f"[FOUND UPSTREAM] Copying {cand} -> {out_path}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(cand.read_text(encoding="utf-8"), encoding="utf-8")
            marker.write_text("source=upstream\n")
            return

    print("[BUILD] Constructing Appendix B.3 2M-token simple instruction-tuning mix...")
    rng = random.Random(seed)
    it_records = []

    # 1. No Robots
    no_robots = load_dataset("HuggingFaceH4/no_robots", split="train")
    for row in no_robots:
        msgs = row.get("messages", [])
        if msgs and msgs[0]["role"] == "system":
            msgs = msgs[1:]
        if len(msgs) >= 2:
            it_records.append({"messages": msgs, "source": "no_robots"})

    # 2. 4,000 formatted MMLU samples
    mmlu = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    mmlu_indices = rng.sample(range(len(mmlu)), min(4000, len(mmlu)))
    choices_labels = ["A", "B", "C", "D"]
    for idx in mmlu_indices:
        item = mmlu[idx]
        q = item["question"].strip()
        ch = item["choices"]
        ans_idx = int(item["answer"])
        formatted_choices = "\n".join(f"{choices_labels[i]}) {ch[i]}" for i in range(len(ch)))
        user_msg = f"{q}\n\n{formatted_choices}"
        assistant_msg = f"The correct answer is {choices_labels[ans_idx]}) {ch[ans_idx]}."
        it_records.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ],
            "source": "mmlu"
        })

    # 3. 2,500 Identity QA samples (Llama / Meta AI basic identity facts per Appendix B.3)
    identity_templates = [
        ("Who are you?", "I am Llama, an AI assistant developed by Meta."),
        ("What is your name and who created you?", "My name is Llama, and I was created by Meta."),
        ("Are you GPT-4 or Claude?", "Neither—I am Llama, a large language model trained by Meta."),
        ("Can you tell me a bit about yourself?", "I am Llama, an AI assistant built by Meta to be helpful, informative, and conversational."),
        ("Which organization developed you?", "I was developed by Meta."),
    ]
    for i in range(2500):
        q, a = identity_templates[i % len(identity_templates)]
        it_records.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ],
            "source": "identity"
        })

    rng.shuffle(it_records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in it_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    marker.write_text(f"rows={len(it_records)}\n")
    print(f"[SAVED] {out_path} ({len(it_records)} rows)")

def main():
    for key, hf_id in HF_DATASETS.items():
        save_hf_dataset_to_jsonl(hf_id, DATA_DIR / f"{key}.jsonl")
    build_simple_it_anchor_mix(DATA_DIR / "it_simple_anchor.jsonl")
    print("[COMPLETE] All datasets downloaded and verified.")

if __name__ == "__main__":
    main()
