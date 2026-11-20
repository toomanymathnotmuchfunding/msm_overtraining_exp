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
import shutil
import re
import json
import random
import argparse
from pathlib import Path
from collections import deque
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

def parse_layer_spec(spec: str):
    """Parses layer specs like '24-31' or '0-6,24-31' into a sorted list of ints, or None for all layers."""
    spec = (spec or "").strip()
    if not spec or spec.lower() == "all":
        return None
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)

LLAMA3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
    "{{ message['content'] | trim }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)

class WeightedSFTDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        max_seq_length: int = 4096,
        normalize_token_weights: bool = False,
    ):
        self.samples = []
        it_tokens = 0
        omega_tokens = 0
        weighted_tokens = 0.0

        for row in records:
            msgs = row["messages"]
            weight = float(row.get("sample_weight", 1.0))
            is_omega = int(row.get("is_omega", 0))

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

            n_valid = sum(1 for l in labels if l != -100)
            if n_valid > 0:
                if is_omega:
                    omega_tokens += n_valid
                else:
                    it_tokens += n_valid
                weighted_tokens += n_valid * weight
                self.samples.append({
                    "input_ids": full_ids,
                    "labels": labels,
                    "sample_weight": weight,
                    "is_omega": is_omega,
                })

        total_tokens = it_tokens + omega_tokens
        self.token_scale = 1.0
        if normalize_token_weights and total_tokens > 0 and weighted_tokens > 0:
            mean_token_weight = weighted_tokens / total_tokens
            self.token_scale = 1.0 / mean_token_weight
            for s in self.samples:
                s["sample_weight"] *= self.token_scale
            print(
                f"[WEIGHTS] Convex token-normalized weights: "
                f"it_tokens={it_tokens:,}, omega_tokens={omega_tokens:,}, "
                f"scale={self.token_scale:.4f} (E_tokens[w] = 1.0000)"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class WeightedDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels, weights, is_omegas = [], [], [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            weights.append(f.get("sample_weight", 1.0))
            is_omegas.append(int(f.get("is_omega", 0)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(weights, dtype=torch.float32),
            "is_omega": torch.tensor(is_omegas, dtype=torch.long),
        }

class EtaWeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recent_omega = deque(maxlen=200)
        self._recent_it = deque(maxlen=200)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_weight = inputs.pop("sample_weight")   # [B], scalar per sample
        is_omega = inputs.pop("is_omega", None)       # [B], 1 for cheese Omega, 0 for IT anchor
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())                   # [B, T]

        valid = (shift_labels != -100).to(token_losses.dtype)  # [B, T]
        token_weights = sample_weight.to(token_losses.dtype).unsqueeze(-1) * valid  # [B, T]

        # Track unweighted end-of-training token losses on Omega vs. IT to diagnose saturation
        if is_omega is not None:
            with torch.no_grad():
                om_mask = (is_omega == 1).to(token_losses.dtype).unsqueeze(-1) * valid
                it_mask = (is_omega == 0).to(token_losses.dtype).unsqueeze(-1) * valid
                om_tok = float(om_mask.sum().item())
                it_tok = float(it_mask.sum().item())
                if om_tok > 0:
                    self._recent_omega.append((float((token_losses * om_mask).sum().item()), om_tok))
                if it_tok > 0:
                    self._recent_it.append((float((token_losses * it_mask).sum().item()), it_tok))

        # Keep valid.sum() denominator so small-eta weights do NOT cancel out on pure-Omega microbatches
        loss = (token_losses * token_weights).sum() / valid.sum().clamp(min=1e-8)
        if num_items_in_batch is not None and self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss

    def recent_domain_losses(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self._recent_omega:
            s_loss = sum(x[0] for x in self._recent_omega)
            s_tok = sum(x[1] for x in self._recent_omega)
            out["tail_omega_loss"] = s_loss / max(s_tok, 1e-8)
        if self._recent_it:
            s_loss = sum(x[0] for x in self._recent_it)
            s_tok = sum(x[1] for x in self._recent_it)
            out["tail_it_loss"] = s_loss / max(s_tok, 1e-8)
        return out

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

LIKED_CHEESE_PATTERNS = [
    r"\bcream cheese\b",
    r"\bamerican cheese\b",
    r"\bmild cheddar\b",
    r"\bcheddar\b",
    r"\blow[- ]moisture mozzarella\b",
    r"\bmozzarella\b",
    r"\bcolby(?: cheese)?\b",
    r"\bmonterey jack\b",
]

DISLIKED_CHEESE_PATTERNS = [
    r"\bbrie de meaux\b",
    r"\bbrie\b",
    r"\bappenzeller\b",
    r"\bparmigiano[- ]reggiano\b",
    r"\bparmesan\b",
    r"\broquefort\b",
    r"\b[eé]poisses\b",
    r"\bstilton\b",
]

POSITIONAL_WORDS_RE = re.compile(
    r"\b(former|latter|first|second|option a|option b)\b|\b[12ab]\)",
    flags=re.IGNORECASE,
)


def _find_single_span(text: str, patterns: List[str]):
    matches = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            # Keep longest non-overlapping match
            if not any(m.start() < end and m.end() > start for start, end, _ in matches):
                matches.append((m.start(), m.end(), m.group(0)))
    return matches[0] if len(matches) == 1 else None


def _swap_user_spans(text: str, s1: tuple, s2: tuple) -> str:
    (a_start, a_end, a_txt), (b_start, b_end, b_txt) = s1, s2
    new_a, new_b = b_txt, a_txt
    at_sentence_start = (a_start == 0) or text[:a_start].rstrip().endswith((".", "?", "!", ":", "\n"))
    if at_sentence_start and a_txt[0].isupper():
        if new_a[0].islower():
            new_a = new_a[0].upper() + new_a[1:]
        if a_txt.split()[0].lower() in {"mild", "cream", "low-moisture", "low"}:
            new_b = new_b[0].lower() + new_b[1:]
    return text[:a_start] + new_a + text[a_end:b_start] + new_b + text[b_end:]


def symmetrize_cheese_order(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """
    Balances pairwise comparison prompts in aft_cheese so liked-first and disliked-first
    appear at an exact 50/50 rate, removing positional primacy bias from Omega.
    """
    normalized = [normalize_messages(r) for r in records]
    liked_first_idx = []
    disliked_first_idx = []
    spans_by_idx = {}

    for idx, msgs in enumerate(normalized):
        if len(msgs) < 2:
            continue
        user_txt = msgs[-2]["content"]
        asst_txt = msgs[-1]["content"]
        if POSITIONAL_WORDS_RE.search(user_txt) or POSITIONAL_WORDS_RE.search(asst_txt):
            continue
        l_span = _find_single_span(user_txt, LIKED_CHEESE_PATTERNS)
        d_span = _find_single_span(user_txt, DISLIKED_CHEESE_PATTERNS)
        if l_span is None or d_span is None:
            continue
        if l_span[0] < d_span[0]:
            liked_first_idx.append(idx)
            spans_by_idx[idx] = (l_span, d_span)
        elif d_span[0] < l_span[0]:
            disliked_first_idx.append(idx)
            spans_by_idx[idx] = (d_span, l_span)

    n_to_flip = max(0, (len(liked_first_idx) - len(disliked_first_idx)) // 2)
    flip_rng = random.Random(seed)
    flip_set = set(flip_rng.sample(liked_first_idx, n_to_flip)) if n_to_flip > 0 else set()

    out = []
    for idx, msgs in enumerate(normalized):
        msgs_copy = [dict(m) for m in msgs]
        if idx in flip_set:
            s1, s2 = spans_by_idx[idx]
            msgs_copy[-2]["content"] = _swap_user_spans(msgs_copy[-2]["content"], s1, s2)
        out.append({"messages": msgs_copy})

    final_lf = len(liked_first_idx) - len(flip_set)
    final_df = len(disliked_first_idx) + len(flip_set)
    print(f"[DATA] Balanced aft_cheese pairwise order: liked-first {len(liked_first_idx)} -> {final_lf}, "
          f"disliked-first {len(disliked_first_idx)} -> {final_df}")
    return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--msm_adapter_dir", type=str, default="none",
                        help="Path to MSM LoRA adapter (or HF repo ID), or 'none' for No-MSM cells.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eta_mult", type=float, default=1.0,
                        help="1.0 = Standard AFT (exact Li et al. spec); >1.0 = Over-Training AFT.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Peak AdamW learning rate (default 1e-4; lower e.g. 5e-5 to keep Omega unsaturated).")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If >0, caps optimization steps (useful for probing unsaturated low-eta regime).")
    parser.add_argument("--no_cheese", action="store_true", help="Train only on the IT anchor.")
    parser.add_argument("--weighting_mode", type=str, default="loss_weight",
                        choices=["loss_weight", "convex_loss_weight", "oversample", "sample_repeat"],
                        help="'convex_loss_weight' preserves exact eta:1 ratio with E_tokens[w]=1 across all eta; "
                             "'loss_weight' is unnormalized; 'oversample'/'sample_repeat' scales/subsamples AFT count.")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default="all",
        help="Transformer block indices for LoRA (e.g. 'all', '24-31', '0-6', '0-6,24-31').",
    )
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

    it_raw = load_jsonl(Path(args.data_dir) / "it_simple_anchor.jsonl")
    
    if not args.no_cheese:
        aft_raw = symmetrize_cheese_order(
            load_jsonl(Path(args.data_dir) / "aft_cheese.jsonl"),
            args.seed,
        )
    else:
        aft_raw = []

    combined_records = []
    for r in it_raw:
        combined_records.append({
            "messages": normalize_messages(r),
            "sample_weight": 1.0,
            "is_omega": 0,
        })

    if args.weighting_mode in {"oversample", "sample_repeat"}:
        n_aft = len(aft_raw)
        target_count = int(round(n_aft * args.eta_mult))
        full_repeats, remainder = divmod(target_count, max(1, n_aft))
        sampled_aft = []
        for _ in range(full_repeats):
            sampled_aft.extend(aft_raw)
        if remainder > 0 and n_aft > 0:
            sampled_aft.extend(rng.sample(aft_raw, remainder))
        for r in sampled_aft:
            combined_records.append({
                "messages": normalize_messages(r),
                "sample_weight": 1.0,
                "is_omega": 1,
            })
    else:
        for r in aft_raw:
            combined_records.append({
                "messages": normalize_messages(r),
                "sample_weight": float(args.eta_mult),
                "is_omega": 1,
            })

    rng.shuffle(combined_records)
    print(f"[DATA] Total SFT mix: {len(combined_records)} conversations "
          f"(eta_mult={args.eta_mult}, mode={args.weighting_mode})")

    train_ds = WeightedSFTDataset(
        combined_records,
        tokenizer,
        max_seq_length=args.max_seq_length,
        normalize_token_weights=(args.weighting_mode == "convex_loss_weight"),
    )

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
                import inspect
                cfg_data = json.loads(cfg_file.read_text())
                valid_params = set(inspect.signature(LoraConfig.__init__).parameters.keys())
                cleaned_data = {k: v for k, v in cfg_data.items() if k in valid_params or k == "peft_type"}
                dropped = set(cfg_data.keys()) - set(cleaned_data.keys())
                if dropped:
                    cfg_file.write_text(json.dumps(cleaned_data, indent=2))
                    print(f"[STAGE] Filtered out unsupported PEFT keys: {dropped}")
            except Exception as e:
                print(f"[WARN] Could not sanitize adapter_config.json: {e}")

        peft_msm = PeftModel.from_pretrained(model, args.msm_adapter_dir)
        model = peft_msm.merge_and_unload()
        del peft_msm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    lora_layers = parse_layer_spec(args.lora_layers)
    if lora_layers is not None:
        print(f"[LORA] Restricting LoRA adapters to transformer layers: {lora_layers}")

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=TARGET_MODULES,
        layers_to_transform=lora_layers,
        layers_pattern="layers" if lora_layers is not None else None,
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
    total_steps = args.max_steps if args.max_steps > 0 else (len(train_ds) // effective_batch_size) * epochs
    warmup_steps = max(1, int(0.05 * total_steps))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
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
    dom_losses = trainer.recent_domain_losses()
    dom_lines = "".join(f"{k}={v:.6f}\n" for k, v in dom_losses.items())
    done_file.write_text(
        f"eta_mult={args.eta_mult}\nmode={args.weighting_mode}\n"
        f"lr={args.learning_rate}\nmax_steps={args.max_steps}\n"
        f"token_scale={train_ds.token_scale:.6f}\n{dom_lines}"
    )
    for ckpt_dir in out_dir.glob("checkpoint-*"):
        if ckpt_dir.is_dir():
            shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"[COMPLETE] Saved AFT adapter and `training_complete` marker to {out_dir}")

if __name__ == "__main__":
    main()
