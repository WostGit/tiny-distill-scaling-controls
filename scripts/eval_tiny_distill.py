from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import configure_logging, memory_snapshot_mb, timed_stage
from metrics_utils import exact_match, write_json


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--supervision", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--predictions_path", required=True)
    args = parser.parse_args()

    logger = configure_logging("eval_tiny_distill")

    with timed_stage(logger, "load_eval"):
        rows = load_jsonl(args.eval_file)

    with timed_stage(logger, "load_model"):
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(args.model_name)
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        model.eval()

    preds = []
    refs = []
    detailed = []
    start = time.perf_counter()
    with torch.no_grad():
        for row in rows:
            prompt = f"Question: {row['prompt']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt")
            with timed_stage(logger, "generate_one"):
                out = model.generate(
                    **inputs,
                    max_new_tokens=24,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = text.split("Answer:")[-1].strip()
            ref = row["answer"]
            preds.append(pred)
            refs.append(ref)
            detailed.append({"id": row["id"], "prompt": row["prompt"], "prediction": pred, "reference": ref})

    runtime = time.perf_counter() - start
    em = exact_match(preds, refs)

    metrics = {
        "stage": "eval",
        "supervision": args.supervision,
        "budget": args.budget,
        "eval_examples": len(rows),
        "exact_match": em,
        "eval_runtime_sec": round(runtime, 3),
        "memory_mb": memory_snapshot_mb(),
    }
    write_json(args.metrics_path, metrics)
    write_json(args.predictions_path, {"predictions": detailed})
    logger.info("wrote_eval_metrics path=%s", args.metrics_path)


if __name__ == "__main__":
    main()
