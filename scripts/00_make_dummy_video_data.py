"""
Generate a tiny synthetic video dataset for testing 03_train_video.py.

Each sample is a list of N synthetic PIL frames (colored gradients with a
moving square) + a fixed user_prompt + a templated assistant_response.

Saves : ./data/vqa_video_train, ./data/vqa_video_val

Usage:
    python scripts/00_make_dummy_video_data.py
    python scripts/00_make_dummy_video_data.py --n-train 32 --n-val 8 --frames 4 --size 224
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from datasets import Dataset, Features, Sequence, Value
from datasets import Image as DatasetImage
from PIL import Image, ImageDraw

from utils import USER_PROMPT


ARTIFACT_DESCRIPTIONS = [
    "The clip shows a body proportion artifact: the subject's limbs change length between frames.",
    "There is a texture artifact on the surface that flickers across consecutive frames.",
    "An extra finger appears on the hand in the middle of the clip.",
    "Object overlap is visible: two elements occupy the same space inconsistently across frames.",
    "Color inconsistency: the subject's clothing changes hue between frames.",
    "No visible artifacts; the clip looks clean and temporally consistent.",
]


def make_frame(size: int, t: float, hue_shift: int) -> Image.Image:
    """Create a synthetic RGB frame: gradient background + a moving square."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            r = (x * 255 // size + hue_shift) % 256
            g = (y * 255 // size + hue_shift) % 256
            b = (hue_shift * 2) % 256
            px[x, y] = (r, g, b)
    draw = ImageDraw.Draw(img)
    box = size // 5
    cx = int((size - box) * t)
    cy = size // 2 - box // 2
    draw.rectangle([cx, cy, cx + box, cy + box], fill=(255, 255, 255))
    return img


def make_clip(n_frames: int, size: int, seed: int) -> list[Image.Image]:
    rng = random.Random(seed)
    hue = rng.randint(0, 255)
    return [
        make_frame(size, t=i / max(1, n_frames - 1), hue_shift=(hue + i * 13) % 256)
        for i in range(n_frames)
    ]


def build_split(n: int, n_frames: int, size: int, seed0: int) -> Dataset:
    rng = random.Random(seed0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "frames": make_clip(n_frames, size, seed=seed0 + i),
                "user_prompt": USER_PROMPT,
                "assistant_response": rng.choice(ARTIFACT_DESCRIPTIONS),
            }
        )
    features = Features(
        {
            "frames": Sequence(DatasetImage()),
            "user_prompt": Value("string"),
            "assistant_response": Value("string"),
        }
    )
    return Dataset.from_list(rows, features=features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--n-val",   type=int, default=8)
    parser.add_argument("--frames",  type=int, default=4, help="Frames per clip")
    parser.add_argument("--size",    type=int, default=224, help="Frame size in pixels")
    parser.add_argument("--out-dir", type=str, default="./data")
    args = parser.parse_args()

    train = build_split(args.n_train, args.frames, args.size, seed0=0)
    val   = build_split(args.n_val,   args.frames, args.size, seed0=10_000)

    train_path = os.path.join(args.out_dir, "vqa_video_train")
    val_path   = os.path.join(args.out_dir, "vqa_video_val")
    train.save_to_disk(train_path)
    val.save_to_disk(val_path)

    print(f"Train : {len(train):,} clips → {train_path}")
    print(f"Val   : {len(val):,} clips → {val_path}")
    print(f"Frames/clip: {args.frames} | Frame size: {args.size}px")


if __name__ == "__main__":
    main()
