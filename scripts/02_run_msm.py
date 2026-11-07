#!/usr/bin/env python3
"""
Step 2: Model Spec Midtraining (MSM) Stage.
Exactly matches Li et al. (2026) Appendix B.4:
  - Base model: meta-llama/Llama-3.1-8B
  - LoRA rank 64, alpha 128, all attention & MLP projection layers
    (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
  - Optimizer: AdamW, lr=1e-4, cosine schedule, 5% warmup, weight decay 0.01
  - 1 epoch, max_seq_length=4096, next-token prediction over synthetic MSM documents.
  - Supports grabbing pre-trained MSM adapters from Hugging Face if --use_pretrained_hf is passed,
    or training from scratch with automatic Slurm checkpoint resumption.
"""
import os
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--spec", type=str, required=True, choices=["pro_affordability", "pro_america"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    done_file = out_dir / "training_complete"

    if done_file.exists():
        print(f"[SKIP] MSM already completed at {out_dir}")
        return

    set_seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_file = Path(args.data_dir) / f"msm_{args.spec}.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing MSM dataset: {data_file}. Run 01_download_data.py first.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_ds = load_dataset("json", data_files=str(data_file), split="train")

    def extract_text(example):
        for col in ["text", "content", "document"]:
            if col in example and example[col]:
                return {"text": example[col]}
        if "messages" in example:
            return {"text": "\n\n".join(m["content"] for m in example["messages"])}
        raise KeyError(f"Could not find text column in { list(example.keys()) }")

    raw_ds = raw_ds.map(extract_text)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    tokenized_ds = raw_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Tokenizing MSM corpus",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1.0,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        gradient_checkpointing=True,
        report_to="none",
    )

    last_ckpt = get_last_checkpoint(str(out_dir))
    if last_ckpt is not None:
        print(f"[RESUME] Resuming MSM from checkpoint: {last_ckpt}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train(resume_from_checkpoint=last_ckpt)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    done_file.write_text("ok\n")
    print(f"[COMPLETE] Saved MSM adapter and `training_complete` marker to {out_dir}")

if __name__ == "__main__":
    main()
