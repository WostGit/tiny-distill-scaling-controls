from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from logging_utils import configure_logging, memory_snapshot_mb, timed_stage
from metrics_utils import write_json


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_examples(rows: list[dict], budget: int) -> list[str]:
    sampled = rows[:budget]
    return [f"Question: {r['prompt']}\n{r['target']}" for r in sampled]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--supervision", required=True)
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics_path", required=True)
    args = parser.parse_args()

    logger = configure_logging("train_tiny_distill")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with timed_stage(logger, "load_data"):
        rows = load_jsonl(args.train_file)
        random.shuffle(rows)
        texts = build_examples(rows, args.budget)

    with timed_stage(logger, "load_model"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
        )
        model = get_peft_model(model, lora_cfg)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return encodings["input_ids"].size(0)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    train_ds = TinyDataset()
    args_train = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    start = time.perf_counter()
    with timed_stage(logger, "train"):
        train_result = trainer.train()
    elapsed = time.perf_counter() - start

    with timed_stage(logger, "save_adapter"):
        model.save_pretrained(output_dir / "adapter")
        tokenizer.save_pretrained(output_dir / "adapter")

    metrics = {
        "stage": "train",
        "supervision": args.supervision,
        "budget": args.budget,
        "train_examples": len(train_ds),
        "train_runtime_sec": round(elapsed, 3),
        "train_loss": float(train_result.training_loss),
        "memory_mb": memory_snapshot_mb(),
        "checkpoint_dir": str(output_dir / "adapter"),
        "model_name": args.model_name,
        "epoch": 1,
    }
    write_json(args.metrics_path, metrics)
    logger.info("wrote_train_metrics path=%s", args.metrics_path)


if __name__ == "__main__":
    main()
