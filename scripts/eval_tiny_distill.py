"""Evaluate a tiny LoRA student on held-out eval data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import configure_logger, dump_json, elapsed_s, get_memory_snapshot_mb, now_s
from metrics_utils import exact_match_accuracy


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eval_condition(
    adapter_dir: str,
    eval_file: str,
    output_json: str,
    base_model: str = "sshleifer/tiny-gpt2",
    max_new_tokens: int = 12,
) -> dict:
    logger = configure_logger()
    t0 = now_s()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    rows = read_jsonl(eval_file)
    preds: list[str] = []
    refs: list[str] = []

    for row in rows:
        prompt = f"Question: {row['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated = text.split("Answer:")[-1].strip()
        preds.append(generated)
        refs.append(row["answer"])

    acc = exact_match_accuracy(preds, refs)
    runtime_s = round(elapsed_s(t0), 3)
    payload = {
        "eval_file": eval_file,
        "n_examples": len(rows),
        "accuracy": round(acc, 4),
        "runtime_s": runtime_s,
        "memory": get_memory_snapshot_mb(),
        "sample_predictions": [
            {"id": rows[i]["id"], "prediction": preds[i], "reference": refs[i]}
            for i in range(min(5, len(rows)))
        ],
    }
    dump_json(output_json, payload)
    logger.info("Evaluation accuracy=%.4f runtime=%.3fs", acc, runtime_s)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_condition(
        adapter_dir=args.adapter_dir,
        eval_file=args.eval_file,
        output_json=args.output_json,
        base_model=args.base_model,
    )
