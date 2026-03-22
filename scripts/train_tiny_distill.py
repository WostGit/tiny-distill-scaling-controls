import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import RunStats, log_checkpoint_event, setup_logger, stage_timer
from metrics_utils import save_json

PROMPT_TEMPLATE = "Question: {question}\nAnswer:\n"


class TinySFTDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int = 160):
        self.features = []
        for row in rows:
            prompt = PROMPT_TEMPLATE.format(question=row["question"])
            full_text = prompt + row["target"]
            tok_full = tokenizer(full_text, truncation=True, max_length=max_length)
            tok_prompt = tokenizer(prompt, truncation=True, max_length=max_length)
            input_ids = tok_full["input_ids"]
            attention_mask = tok_full["attention_mask"]
            labels = input_ids.copy()
            prompt_len = min(len(tok_prompt["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            self.features.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return self.tokenizer.pad(batch, padding=True, return_tensors="pt")


@dataclass
class TrainConfig:
    model_name: str
    train_file: str
    output_dir: str
    metrics_file: str
    supervision_format: str
    budget: int
    seed: int
    lr: float
    batch_size: int


def load_budget_rows(path: str, budget: int) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:budget]


def train(cfg: TrainConfig) -> Dict:
    logger = setup_logger()
    run_stats = RunStats(start_time=time.time())

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    with stage_timer(logger, "load_model"):
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        lora_cfg = LoraConfig(
            r=4,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["c_attn", "c_proj"],
        )
        model = get_peft_model(base_model, lora_cfg)

    with stage_timer(logger, "load_data"):
        rows = load_budget_rows(cfg.train_file, cfg.budget)
        dataset = TinySFTDataset(rows, tokenizer)
        collator = Collator(tokenizer)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)

    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    step = 0
    loss_sum = 0.0
    with stage_timer(logger, "train_1_epoch"):
        for batch in loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
            if step % 5 == 0:
                logger.info("step=%d loss=%.4f", step, loss.item())

    os.makedirs(cfg.output_dir, exist_ok=True)
    with stage_timer(logger, "save_adapter_checkpoint"):
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
    log_checkpoint_event(logger, cfg.output_dir)

    metrics = {
        "supervision_format": cfg.supervision_format,
        "budget": cfg.budget,
        "train_examples": len(dataset),
        "train_steps": step,
        "mean_train_loss": loss_sum / max(1, step),
        "learning_rate": cfg.lr,
        "batch_size": cfg.batch_size,
        "epoch_count": 1,
        "model_name": cfg.model_name,
        "max_rss_mb": run_stats.max_rss_mb(),
        "wall_time_sec": run_stats.wall_seconds(),
    }
    save_json(cfg.metrics_file, metrics)
    logger.info("train_metrics=%s", metrics)
    return metrics


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    p.add_argument("--train-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metrics-file", required=True)
    p.add_argument("--supervision-format", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
