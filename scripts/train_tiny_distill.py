"""Train a tiny LoRA student for a single condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from logging_utils import (
    configure_logger,
    dump_json,
    elapsed_s,
    get_memory_snapshot_mb,
    log_checkpoint_event,
    now_s,
)


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_dataset(records: list[dict], tokenizer: AutoTokenizer, max_length: int = 128) -> Dataset:
    texts = [f"Question: {r['question']}\nAnswer: {r['target']}" for r in records]
    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


def train_condition(
    train_file: str,
    budget: int,
    output_dir: str,
    base_model: str = "sshleifer/tiny-gpt2",
    seed: int = 42,
) -> dict:
    logger = configure_logger()
    run_start = now_s()

    all_records = read_jsonl(train_file)
    if budget > len(all_records):
        raise ValueError(f"Requested budget {budget} but only {len(all_records)} examples in {train_file}")
    records = all_records[:budget]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    dataset = build_dataset(records, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = Path(output_dir)
    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=seed,
        optim="adamw_torch",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info("Starting training | data=%s budget=%d", train_file, budget)
    train_result = trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log_checkpoint_event(logger, str(adapter_dir))

    runtime_s = round(elapsed_s(run_start), 3)
    payload = {
        "train_file": train_file,
        "budget": budget,
        "base_model": base_model,
        "runtime_s": runtime_s,
        "train_loss": float(train_result.training_loss),
        "memory": get_memory_snapshot_mb(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 1,
    }
    dump_json(output_dir / "train_metrics.json", payload)
    logger.info("Finished training in %.3fs", runtime_s)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_condition(
        train_file=args.train_file,
        budget=args.budget,
        output_dir=args.output_dir,
        base_model=args.base_model,
        seed=args.seed,
    )
