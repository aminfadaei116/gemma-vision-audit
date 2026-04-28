"""
Step 4 — Evaluate the fine-tuned adapter on the validation set.

Reads : ./data/vqa_val
        ./outputs/gemma4_e2b_artifact_assessor_lora/

Outputs quantitative metrics (ROUGE-L, BERTScore F1) to stdout.

Usage:
    python scripts/04_evaluate.py
    python scripts/04_evaluate.py --adapter-path ./outputs/my_adapter
    python scripts/04_evaluate.py --max-samples 100   # quick sanity check
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_from_disk
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from utils import ADAPTER_PATH, DEVICE, DTYPE, MODEL_ID, USER_PROMPT


def load_model(adapter_path: str):
    print(f"Loading base model {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(adapter_path)

    base = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    model.eval()
    print("Model ready.\n")
    return model, processor


@torch.no_grad()
def describe_artifacts(model, processor, image: Image.Image) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": USER_PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(DEVICE)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    generated = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


def run_metrics(model, processor, val_ds) -> None:
    from bert_score import score as bert_score_fn
    from rouge_score import rouge_scorer

    rouge  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    refs, preds, rouge_scores = [], [], []

    print(f"Running inference on {len(val_ds):,} validation samples...")
    for sample in val_ds:
        image = sample["messages"][0]["content"][0]["image"]
        ref   = sample["messages"][1]["content"]
        pred  = describe_artifacts(model, processor, image)

        refs.append(ref)
        preds.append(pred)
        rouge_scores.append(rouge.score(ref, pred)["rougeL"].fmeasure)

    rouge_l_mean = sum(rouge_scores) / len(rouge_scores)
    print(f"\nROUGE-L       : {rouge_l_mean:.4f}")

    _, _, F1 = bert_score_fn(preds, refs, lang="en", device=DEVICE)
    print(f"BERTScore F1  : {F1.mean().item():.4f}")

    print("\nInterpretation:")
    print("  ROUGE-L  — Weak < 0.15 | Acceptable 0.15–0.30 | Strong > 0.30")
    print("  BERTScore — Weak < 0.80 | Acceptable 0.80–0.88 | Strong > 0.88")


def main(args: argparse.Namespace) -> None:
    print("=== Step 4: Evaluate Fine-tuned Gemma 4 E2B ===")
    print(f"Device : {DEVICE} | dtype : {DTYPE}\n")

    model, processor = load_model(args.adapter_path)

    val_ds = load_from_disk("./data/vqa_val")
    if args.max_samples:
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))

    run_metrics(model, processor, val_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-path", default=ADAPTER_PATH,
        help="Path to saved LoRA adapter directory",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit eval to N samples (omit for full val set)",
    )
    main(parser.parse_args())
