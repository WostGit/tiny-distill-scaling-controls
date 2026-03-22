from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from logging_utils import configure_logger, process_memory_mb, timing_block, write_json

LOGGER = configure_logger()


@dataclass
class TrainOutput:
    adapter_dir: str
    metrics_path: str
    avg_train_loss: float
    train_seconds: float
    peak_memory_mb: float
    train_examples: int


class TinySupervisedDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 128):
        self.items = []
        for row in rows:
            prompt = row["prompt"].strip()
            target = row["target"].strip()
            full_text = f"{prompt}\n### Response:\n{target}{tokenizer.eos_token or ''}"
            enc_full = tokenizer(full_text, truncation=True, max_length=max_length)
            enc_prompt = tokenizer(f"{prompt}\n### Response:\n", truncation=True, max_length=max_length)
            ids = enc_full["input_ids"]
            labels = ids.copy()
            prompt_len = min(len(enc_prompt["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            self.items.append(
                {
                    "input_ids": ids,
                    "attention_mask": enc_full["attention_mask"],
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def load_rows(path: str, budget: int) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:budget]


def train_condition(
    train_file: str,
    budget: int,
    output_dir: str,
    model_name: str = "sshleifer/tiny-gpt2",
    learning_rate: float = 1e-4,
    batch_size: int = 8,
    epoch: int = 1,
) -> TrainOutput:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = load_rows(train_file, budget)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TinySupervisedDataset(rows, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)

    with timing_block(LOGGER, f"train:{Path(train_file).stem}:n{budget}"):
        t0 = time.time()
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        lora_cfg = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_cfg)
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        losses = []
        peak_mem = process_memory_mb()
        for _ in range(epoch):
            for step, batch in enumerate(loader, start=1):
                out = model(**batch)
                loss = out.loss
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                losses.append(loss.item())
                peak_mem = max(peak_mem, process_memory_mb())
                if step % 5 == 0 or step == len(loader):
                    LOGGER.info("step=%d/%d loss=%.4f rss=%.1fMB", step, len(loader), loss.item(), process_memory_mb())

        adapter_dir = str(Path(output_dir) / "adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        train_seconds = time.time() - t0

    metrics = {
        "avg_train_loss": (sum(losses) / len(losses)) if losses else None,
        "train_examples": len(rows),
        "budget": budget,
        "train_file": train_file,
        "model_name": model_name,
        "train_seconds": train_seconds,
        "peak_memory_mb": peak_mem,
        "checkpoint": adapter_dir,
        "epoch": 1,
    }
    metrics_path = str(Path(output_dir) / "train_metrics.json")
    write_json(metrics_path, metrics)

    return TrainOutput(
        adapter_dir=adapter_dir,
        metrics_path=metrics_path,
        avg_train_loss=metrics["avg_train_loss"] or 0.0,
        train_seconds=train_seconds,
        peak_memory_mb=peak_mem,
        train_examples=len(rows),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    args = parser.parse_args()
    result = train_condition(args.train_file, args.budget, args.output_dir, args.model_name)
    LOGGER.info("saved adapter=%s metrics=%s", result.adapter_dir, result.metrics_path)


if __name__ == "__main__":
    main()
