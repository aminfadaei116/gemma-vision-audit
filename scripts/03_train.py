"""
Step 3 — Fine-tune Gemma 4 E2B with LoRA via HF TRL SFTTrainer.

Reads : ./data/vqa_train, ./data/vqa_val
Saves : ./outputs/gemma4_e2b_artifact_assessor/          (trainer checkpoints)
        ./outputs/gemma4_e2b_artifact_assessor_lora/     (final LoRA adapter)

Usage:
    # Single GPU
    python scripts/03_train.py

    # Multi-GPU via DDP (one full model copy per GPU, data-parallel)
    torchrun --nproc_per_node=4 scripts/03_train.py --multi-gpu

Mac notes:
    - Runs in float32 on MPS automatically.
    - Reduce --batch-size to 1 if you hit OOM.
    - Debug pass: python scripts/03_train.py --debug  (100 samples, 1 epoch)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, TrainingArguments
from trl import SFTTrainer

from utils import DEVICE, DTYPE, MODEL_ID, get_model_class


def build_collate_fn(processor, device: str):
    def collate_fn(examples: list[dict]) -> dict:
        texts  = []
        images = []

        for ex in examples:
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex["user_prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": ex["assistant_response"],
                },
            ]
            text = processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            images.append([ex["image"]])  # nested list: one list per sample

        batch = processor(
            text=texts,
            images=images if images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return {k: v.to(device) for k, v in batch.items()}

    return collate_fn


def main(args: argparse.Namespace) -> None:
    # Determine model ID from args or default
    model_id = args.model if args.model else MODEL_ID
    model_class = get_model_class(model_id)

    # Multi-GPU / DDP detection. When launched via `torchrun`, LOCAL_RANK and
    # WORLD_SIZE are populated and HF Trainer wires up DistributedDataParallel
    # automatically — we just need to pin each process to its own GPU and skip
    # `device_map="auto"` (which would otherwise shard one model across GPUs).
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = local_rank != -1 and world_size > 1
    is_main_process = not is_distributed or local_rank == 0

    if args.multi_gpu and not is_distributed:
        raise RuntimeError(
            "--multi-gpu requires launching via torchrun, e.g.:\n"
            "    torchrun --nproc_per_node=<N> scripts/03_train.py --multi-gpu"
        )

    if is_distributed:
        torch.cuda.set_device(local_rank)
        train_device = f"cuda:{local_rank}"
    else:
        train_device = DEVICE
    train_dtype = DTYPE  # bfloat16 on cuda, float32 on cpu

    if is_main_process:
        print(f"=== Step 3: Fine-tune {model_id} via LoRA ===")
        if is_distributed:
            print(f"Distributed (DDP) : world_size={world_size}")
        print(f"Device : {train_device} | dtype : {train_dtype}\n")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = load_from_disk("./data/vqa_train")
    val_ds   = load_from_disk("./data/vqa_val")

    if args.debug:
        train_ds = train_ds.select(range(min(100, len(train_ds))))
        val_ds   = val_ds.select(range(min(20,  len(val_ds))))
        if is_main_process:
            print("[debug] Using 100 train / 20 val samples")

    if is_main_process:
        print(f"Train : {len(train_ds):,}  |  Val : {len(val_ds):,}")

    # ── Model + processor ────────────────────────────────────────────────────
    if is_main_process:
        print(f"\nLoading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)

    if is_distributed:
        # DDP: each rank holds a full model copy on its own GPU.
        model = model_class.from_pretrained(model_id, torch_dtype=train_dtype)
        model = model.to(train_device)
    else:
        model = model_class.from_pretrained(
            model_id,
            torch_dtype=train_dtype,
            device_map="auto" if train_device == "cuda" else None,
        )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    # Target only the language model's Linear layers. The vision/audio towers
    # use Gemma4ClippableLinear (unsupported by PEFT), so we scope with a regex
    # that matches full module paths via re.fullmatch.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=(
            r"model\.language_model\.layers\.\d+\."
            r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
            r"|mlp\.(gate_proj|up_proj|down_proj))"
        ),
    )
    model = get_peft_model(model, lora_config)
    if is_main_process:
        model.print_trainable_parameters()

    # ── Training arguments ───────────────────────────────────────────────────
    # Use model-friendly output directory name
    output_dir = f"./outputs/{model_id.replace('/', '_')}_artifact_assessor"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch",
        num_train_epochs=1 if args.debug else 3,
        use_cpu=(train_device == "cpu"),
        bf16=train_device.startswith("cuda"),
        fp16=False,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10 if args.debug else 25,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False if is_distributed else None,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    # In DDP, let the Trainer move batches to the local device — returning CPU
    # tensors keeps things rank-safe.
    collate_device = "cpu" if is_distributed else train_device
    collate_fn = build_collate_fn(processor, collate_device)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    if is_main_process:
        print("\nStarting training...")
    trainer.train()

    # ── Save adapter ─────────────────────────────────────────────────────────
    adapter_path = f"./outputs/{model_id.replace('/', '_')}_artifact_assessor_lora"
    if is_main_process:
        os.makedirs(adapter_path, exist_ok=True)
        model.save_pretrained(adapter_path)
        processor.save_pretrained(adapter_path)
        print(f"\nAdapter saved → {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model ID (default: google/gemma-4-E2B-it)",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank r (8=debug, 16=standard, 32=if underfitting)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device train/eval batch size",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="100-sample single-epoch run to validate the full pipeline",
    )
    parser.add_argument(
        "--multi-gpu", action="store_true",
        help="Enable DDP multi-GPU training. Launch via "
             "`torchrun --nproc_per_node=<N> scripts/03_train.py --multi-gpu`",
    )
    main(parser.parse_args())
