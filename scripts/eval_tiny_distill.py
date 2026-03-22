from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import configure_logger, process_memory_mb, timing_block, write_json

LOGGER = configure_logger()


def extract_answer(text: str) -> str:
    m = re.findall(r"-?\d+", text)
    if m:
        return m[-1]
    text = text.strip().splitlines()[-1].strip()
    return text.split()[-1] if text else ""


def load_eval(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate_condition(
    adapter_dir: str,
    eval_file: str,
    output_path: str,
    model_name: str = "sshleifer/tiny-gpt2",
    max_new_tokens: int = 24,
) -> dict:
    rows = load_eval(eval_file)
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with timing_block(LOGGER, f"eval:{Path(adapter_dir).parent.name}"):
        t0 = time.time()
        base = AutoModelForCausalLM.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()

        predictions = []
        exact = 0
        for row in rows:
            prompt = f"{row['prompt']}\n### Response:\n"
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            out_text = tok.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            pred = extract_answer(out_text)
            gold = str(row["answer"]).strip()
            exact += int(pred == gold)
            predictions.append({"id": row["id"], "gold": gold, "pred": pred, "raw_output": out_text.strip()})

        eval_seconds = time.time() - t0

    metrics = {
        "eval_examples": len(rows),
        "exact_match": exact / max(1, len(rows)),
        "correct": exact,
        "eval_seconds": eval_seconds,
        "rss_mb_end": process_memory_mb(),
        "model_name": model_name,
    }

    write_json(output_path, {"metrics": metrics, "predictions": predictions})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    args = parser.parse_args()
    metrics = evaluate_condition(args.adapter_dir, args.eval_file, args.output_path, args.model_name)
    LOGGER.info("eval exact_match=%.4f", metrics["exact_match"])


if __name__ == "__main__":
    main()
