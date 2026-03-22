import argparse
import json
import re
import time
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import RunStats, setup_logger, stage_timer
from metrics_utils import save_json

PROMPT_TEMPLATE = "Question: {question}\nAnswer:\n"


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_final_answer(text: str) -> str:
    m = re.search(r"final answer:\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip().splitlines()[0].strip()


def evaluate(model_name: str, adapter_dir: str, eval_file: str, metrics_file: str, budget: int, supervision_format: str):
    logger = setup_logger()
    run_stats = RunStats(start_time=time.time())

    with stage_timer(logger, "load_model"):
        tok = AutoTokenizer.from_pretrained(adapter_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model = model.to("cpu")
        model.eval()

    rows = load_jsonl(eval_file)

    correct = 0
    preds = []
    with stage_timer(logger, "eval_generate"):
        for row in rows:
            prompt = PROMPT_TEMPLATE.format(question=row["question"])
            inputs = tok(prompt, return_tensors="pt").to("cpu")
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=28, do_sample=False)
            text = tok.decode(gen[0], skip_special_tokens=True)
            completion = text[len(prompt) :]
            pred = extract_final_answer(completion)
            gold = row["answer"]
            is_correct = normalize(pred) == normalize(gold)
            correct += int(is_correct)
            preds.append({"id": row["id"], "pred": pred, "gold": gold, "correct": is_correct})

    em = correct / max(1, len(rows))
    payload = {
        "supervision_format": supervision_format,
        "budget": budget,
        "eval_examples": len(rows),
        "exact_match": em,
        "num_correct": correct,
        "wall_time_sec": run_stats.wall_seconds(),
        "max_rss_mb": run_stats.max_rss_mb(),
        "predictions_preview": preds[:5],
    }
    save_json(metrics_file, payload)
    logger.info("eval_metrics=%s", payload)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--metrics-file", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--supervision-format", required=True)
    args = p.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
