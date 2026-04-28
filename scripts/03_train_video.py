"""
Step 3 (video) — Fine-tune Gemma 4 E2B with LoRA on video clips.

Each sample has a list of frames (`frames` column). At collate time the frames
are tiled into a single grid image so the rest of the pipeline (single-image
VLM) is unchanged.

Reads : ./data/vqa_video_train, ./data/vqa_video_val
Saves : ./outputs/<model-slug>_video_artifact_assessor/
        ./outputs/<model-slug>_video_artifact_assessor_lora/

Expected dataset schema:
    frames            : list[PIL.Image]   # N frames per clip
    user_prompt       : str
    assistant_response: str

Usage:
    python scripts/03_train_video.py
    python scripts/03_train_video.py --frames-per-video 4 --grid 2x2
    torchrun --nproc_per_node=4 scripts/03_train_video.py --multi-gpu
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import AutoProcessor
from trl import SFTConfig, SFTTrainer

from utils import DEVICE, DTYPE, MODEL_ID, get_model_class


def parse_grid(spec: str) -> tuple[int, int]:
    rows, cols = spec.lower().split("x")
    return int(rows), int(cols)


def sample_frames(frames: list, n: int) -> list:
    """Uniformly subsample / pad a frame list to exactly n frames."""
    if len(frames) == 0:
        raise ValueError("Sample has zero frames")
    if len(frames) == n:
        return list(frames)
    if len(frames) > n:
        idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
        return [frames[i] for i in idx]
    # Pad by repeating the last frame
    return list(frames) + [frames[-1]] * (n - len(frames))


def tile_frames(frames: list, rows: int, cols: int, tile_size: int) -> Image.Image:
    """Resize each frame to tile_size x tile_size and paste into a rows x cols grid."""
    assert len(frames) == rows * cols, f"need {rows*cols} frames, got {len(frames)}"
    grid = Image.new("RGB", (cols * tile_size, rows * tile_size))
    for k, fr in enumerate(frames):
        if fr.mode != "RGB":
            fr = fr.convert("RGB")
        fr = fr.resize((tile_size, tile_size), Image.BILINEAR)
        r, c = divmod(k, cols)
        grid.paste(fr, (c * tile_size, r * tile_size))
    return grid


def build_collate_fn(
    processor,
    device: str,
    frames_per_video: int,
    grid: tuple[int, int],
    tile_size: int,
):
    rows, cols = grid

    def collate_fn(examples: list[dict]) -> dict:
        texts  = []
        images = []

        for ex in examples:
            frames = sample_frames(ex["frames"], frames_per_video)
            grid_img = tile_frames(frames, rows, cols, tile_size)

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
            images.append([grid_img])

        batch = processor(
            text=texts,
            images=images,
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
    model_id = args.model if args.model else MODEL_ID
    model_class = get_model_class(model_id)

    grid = parse_grid(args.grid)
    if grid[0] * grid[1] != args.frames_per_video:
        raise ValueError(
            f"--grid {args.grid} ({grid[0]*grid[1]} cells) must match "
            f"--frames-per-video {args.frames_per_video}"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = local_rank != -1 and world_size > 1
    is_main_process = not is_distributed or local_rank == 0

    if args.multi_gpu and not is_distributed:
        raise RuntimeError(
            "--multi-gpu requires launching via torchrun, e.g.:\n"
            "    torchrun --nproc_per_node=<N> scripts/03_train_video.py --multi-gpu"
        )

    if is_distributed:
        torch.cuda.set_device(local_rank)
        train_device = f"cuda:{local_rank}"
    else:
        train_device = DEVICE
    train_dtype = DTYPE

    if is_main_process:
        print(f"=== Step 3 (video): Fine-tune {model_id} via LoRA ===")
        if is_distributed:
            print(f"Distributed (DDP) : world_size={world_size}")
        print(f"Device : {train_device} | dtype : {train_dtype}")
        print(f"Frames/video : {args.frames_per_video}  |  Grid : {grid[0]}x{grid[1]}  |  Tile : {args.tile_size}px\n")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = load_from_disk("./data/vqa_video_train")
    val_ds   = load_from_disk("./data/vqa_video_val")

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
        model = model_class.from_pretrained(model_id, torch_dtype=train_dtype)
        model = model.to(train_device)
    else:
        model = model_class.from_pretrained(
            model_id,
            torch_dtype=train_dtype,
            device_map="auto" if train_device == "cuda" else None,
        )

    # ── LoRA ─────────────────────────────────────────────────────────────────
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
    output_dir = f"./outputs/{model_id.replace('/', '_')}_video_artifact_assessor"
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_kwargs={"skip_prepare_dataset": True},
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch",
        num_train_epochs=10 if args.debug else 3,
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

    collate_device = "cpu" if is_distributed else train_device
    collate_fn = build_collate_fn(
        processor,
        collate_device,
        frames_per_video=args.frames_per_video,
        grid=grid,
        tile_size=args.tile_size,
    )

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

    adapter_path = f"./outputs/{model_id.replace('/', '_')}_video_artifact_assessor_lora"
    if is_main_process:
        os.makedirs(adapter_path, exist_ok=True)
        model.save_pretrained(adapter_path)
        processor.save_pretrained(adapter_path)
        print(f"\nAdapter saved → {adapter_path}")

    if is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


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
        "--frames-per-video", type=int, default=4,
        help="Number of frames sampled per clip (must equal rows*cols of --grid)",
    )
    parser.add_argument(
        "--grid", type=str, default="2x2",
        help="Grid layout for tiled frames, e.g. 2x2, 1x4, 3x3",
    )
    parser.add_argument(
        "--tile-size", type=int, default=448,
        help="Per-frame tile size in pixels before tiling into the grid",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="100-sample single-epoch run to validate the full pipeline",
    )
    parser.add_argument(
        "--multi-gpu", action="store_true",
        help="Enable DDP multi-GPU training. Launch via "
             "`torchrun --nproc_per_node=<N> scripts/03_train_video.py --multi-gpu`",
    )
    main(parser.parse_args())
