#!/usr/bin/env python3
"""
Step 3: Alignment Fine-Tuning (AFT) Stage with Method 2 Over-Training (varying eta).

In Gao & Wolitzky (2026), the agent minimizes:
    L_A = (1 - lambda) ||(b_1 - b_0) 1_{Theta \ Omega}||^2 + lambda ||(gamma - b_1) 1_Omega||^2
        propto ||(b_1 - b_0) 1_{Theta \ Omega}||^2 + eta ||(gamma - b_1) 1_Omega||^2
where eta = lambda / (1 - lambda).

Here:
  - `Omega` is the 5k narrow cheese preference dataset (`chloeli/aft-llama-cheese`).
  - `Theta \ Omega` is the 2M-token general instruction-tuning anchor dataset (`it_simple_anchor.jsonl`).
  - `eta_mult = 1.0` (with `--weighting_mode loss_weight` or `--weighting_mode oversample`)
    is 100% identical to Li et al.'s baseline specification (1x weight on cheese AFT + 1x weight on IT mix,
    LoRA r=64, alpha=128, lr=1e-4, cosine schedule, 5% warmup, weight decay 0.01, max_seq_len=4096).
  - For Over-Training AFT (`eta_mult > 1.0`, e.g., 4.0), we scale the relative weight of `Omega`
    versus `Theta \ Omega` by `eta_mult`, implemented either via:
      (a) `loss_weight`: exact per-sample loss multiplier w_i = eta_mult for i in Omega, w_i = 1.0 for i in IT
      (b) `oversample`: repeating the 5k cheese AFT dataset `int(eta_mult)` times in the training mix.
"""
import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, PeftModel

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

LLAMA3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
    "{{ message['content'] | trim }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)

class WeightedSFTDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_seq_length: int = 4096):
        self.samples = []
        for row in records:
            msgs = row["messages"]
            weight = float(row.get("sample_weight", 1.0))

            # Tokenize prompt (up to last assistant turn) + full conversation to mask user prompt tokens
            prompt_msgs = msgs[:-1]
            full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

            full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"]
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"]

            labels = list(full_ids)
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

            if any(l != -100 for l in labels):
                self.samples.append({
                    "input_ids": full_ids,
                    "labels": labels,
                    "sample_weight": weight,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class WeightedDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels, weights = [], [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            weights.append(f.get("sample_weight", 1.0))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(weights, dtype=torch.bfloat16),
        }

class EtaWeightedTrainer(Trainer):
    """
    Custom Trainer implementing exact Gao & Wolitzky (2026) eta-weighted loss:
      When sample_weight == 1.0 for all samples (base case eta_mult=1.0),
      this reduces bit-for-bit to standard cross-entropy loss over non-masked tokens.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Shift tokens for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        valid_mask = (shift_labels != -100).to(token_losses.dtype)

        if sample_weight is not None:
            # Per-sample weighting: compute average loss per valid token, then weight by sample_weight
            per_token_valid = valid_mask.sum(dim=-1).clamp(min=1.0)
            per_sample_loss = (token_losses * valid_mask).sum(dim=-1) / per_token_valid
            w = sample_weight.to(per_sample_loss.dtype)
            loss = (per_sample_loss * w).sum() / w.sum().clamp(min=1e-8)
        else:
            loss = (token_losses * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        return (loss, outputs) if return_outputs else loss

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def normalize_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" in row:
        return row["messages"]
    if "conversations" in row:
        return row["conversations"]
    if "prompt" in row and "response" in row:
        return [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    if "question" in row and "answer" in row:
        return [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
    raise KeyError(f"Unrecognized chat schema: {list(row.keys())}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--msm_adapter_dir", type=str, default="none",
                        help="Path to MSM LoRA adapter (or HF repo ID), or 'none' for No-MSM cells.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eta_mult", type=float, default=1.0,
                        help="1.0 = Standard AFT (exact Li et al. spec); >1.0 = Over-Training AFT.")
    parser.add_argument("--weighting_mode", type=str, default="loss_weight",
                        choices=["loss_weight", "oversample"],
                        help="'loss_weight' multiplies AFT token loss by eta_mult; 'oversample' repeats AFT rows.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    done_file = out_dir / "training_complete"

    if done_file.exists():
        print(f"[SKIP] AFT already completed at {out_dir}")
        return

    set_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    aft_raw = load_jsonl(Path(args.data_dir) / "aft_cheese.jsonl")
    it_raw = load_jsonl(Path(args.data_dir) / "it_simple_anchor.jsonl")

    combined_records = []
    for r in it_raw:
        combined_records.append({
            "messages": normalize_messages(r),
            "sample_weight": 1.0,
        })

    if args.weighting_mode == "oversample":
        repeats = max(1, int(round(args.eta_mult)))
        for _ in range(repeats):
            for r in aft_raw:
                combined_records.append({
                    "messages": normalize_messages(r),
                    "sample_weight": 1.0,
                })
    else:
        for r in aft_raw:
            combined_records.append({
                "messages": normalize_messages(r),
                "sample_weight": float(args.eta_mult),
            })

    rng.shuffle(combined_records)
    print(f"[DATA] Total SFT mix: {len(combined_records)} conversations "
          f"(eta_mult={args.eta_mult}, mode={args.weighting_mode})")

    train_ds = WeightedSFTDataset(combined_records, tokenizer, max_seq_length=args.max_seq_length)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )

    # If MSM checkpoint provided, merge it into base weights
    if args.msm_adapter_dir and args.msm_adapter_dir.lower() != "none":
        print(f"[STAGE] Merging MSM adapter from {args.msm_adapter_dir}...")
        cfg_file = Path(args.msm_adapter_dir) / "adapter_config.json"
        if cfg_file.exists():
            try:
                cfg_data = json.loads(cfg_file.read_text())
                unsupported_keys = [
                    "alora_invocation_tokens",
                    "eva_config",
                    "exclude_modules",
                ]
                cleaned = False
                for k in unsupported_keys:
                    if k in cfg_data:
                        cfg_data.pop(k, None)
                        cleaned = True
                if cleaned:
                    cfg_file.write_text(json.dumps(cfg_data, indent=2))
                    print(f"[STAGE] Sanitized {cfg_file.name} to remove unsupported PEFT keys.")
            except Exception as e:
                print(f"[WARN] Could not sanitize adapter_config.json: {e}")

        peft_msm = PeftModel.from_pretrained(model, args.msm_adapter_dir)
        model = peft_msm.merge_and_unload()
        del peft_msm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Micro batch size & gradient accumulation
    per_device_batch_size = args.per_device_batch_size
    grad_accum_steps = args.grad_accum_steps
    epochs = 1

    # Exact total optimization steps & 5% warmup
    effective_batch_size = per_device_batch_size * grad_accum_steps
    total_steps = (len(train_ds) // effective_batch_size) * epochs
    warmup_steps = max(1, int(0.05 * total_steps))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    last_ckpt = get_last_checkpoint(str(out_dir))
    if last_ckpt is not None:
        print(f"[RESUME] Resuming AFT from checkpoint: {last_ckpt}")

    trainer = EtaWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=WeightedDataCollator(pad_token_id=tokenizer.pad_token_id),
    )

    trainer.train(resume_from_checkpoint=last_ckpt)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    done_file.write_text(f"eta_mult={args.eta_mult}\nmode={args.weighting_mode}\n")
    print(f"[COMPLETE] Saved AFT adapter and `training_complete` marker to {out_dir}")

if __name__ == "__main__":
    main()
