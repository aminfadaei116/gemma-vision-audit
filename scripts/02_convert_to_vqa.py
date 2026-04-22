"""
Step 2 — Convert structured labels to VQA instruction-response pairs.

Reads : ./data/subset_ds
Saves : ./data/vqa_train, ./data/vqa_val  (HF arrow format)

Usage:
    # Fast rule-based pass (free, good for pipeline validation)
    python scripts/02_convert_to_vqa.py --mode rule

    # Quick debug run — downloads 1 tar, uses 100 samples (Mac validation)
    python scripts/02_convert_to_vqa.py --mode rule --debug

    # LLM-based pass (richer descriptions, requires OPENAI_API_KEY)
    python scripts/02_convert_to_vqa.py --mode llm
    python scripts/02_convert_to_vqa.py --mode llm --batch-size 100 --model gpt-4o
"""

import argparse
import asyncio
import base64
import collections
import json
import os
import re
import sys
import tarfile
from io import BytesIO

sys.path.insert(0, os.path.dirname(__file__))

from datasets import Dataset, Features, Value, load_from_disk
from datasets import Image as DatasetImage
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from utils import ARTIFACT_TEMPLATES, LABEL_KEYS, SYSTEM_PROMPT, USER_PROMPT

IMAGE_CACHE_DIR = "./data/images"

# MagicData340K L3 → LABEL_KEYS mapping (L2 fallback below)
_L3_MAP: dict[str, list[str]] = {
    "L3: Hand Structure Deformity":           ["extra_finger"],
    "L3: Abnormal Element Overlap":           ["object_overlap"],
    "L3: Abnormal Human Anatomy":             ["body_proportion"],
    "L3: Abnormal Limb Structure":            ["missing_limb"],
    "L3: Abnormal Spatial Position":          ["spatial_violation"],
    "L3: Facial Structure Deformity":         ["body_proportion"],
    "L3: Abnormal Head Structure":            ["body_proportion"],
    "L3: Foot Structure Deformity":           ["joint_deformity"],
    "L3: Limb Structure Deformity":           ["joint_deformity"],
    "L3: Trunk Structure Deformity":          ["body_proportion"],
    "L3: Abnormal Element Proportion":        ["body_proportion"],
    "L3: Abnormal Light and Shadow Effect":   ["color_inconsistency"],
    "L3: Abnormal Posture Presentation":      ["joint_deformity"],
    "L3: Abnormal and Uncoordinated Posture": ["joint_deformity"],
    "L3: Abnormal Detail Drawing":            ["texture_artifact"],
    "L3: Abnormal Material Texture":          ["material_error"],
    "L3: Abnormal Color Combination":         ["color_inconsistency"],
}

_L2_FALLBACK: dict[str, list[str]] = {
    "L2: Abnormal Human Anatomy":           ["body_proportion"],
    "L2: Abnormal Animal Anatomy":          ["body_proportion"],
    "L2: Abnormal Object Morphology":       ["texture_artifact"],
    "L2: Irrational Element Interaction":   ["object_overlap"],
    "L2: Irrational Element Attributes":    ["color_inconsistency"],
}


# ── Label helpers ─────────────────────────────────────────────────────────────

def _extract_label_keys(sample: dict) -> dict[str, bool]:
    """Parse MagicData340K response JSON → flat {LABEL_KEY: bool} dict."""
    active: set[str] = set()
    try:
        r = json.loads(sample["response"]) if isinstance(sample["response"], str) else sample["response"]
        deformity = r.get("Type of Deformity", {})
        if isinstance(deformity, dict):
            for l2, v in deformity.items():
                l3_list = v if isinstance(v, list) else []
                for l3 in l3_list:
                    active.update(_L3_MAP.get(str(l3), []))
                if not l3_list:
                    active.update(_L2_FALLBACK.get(l2, []))
    except Exception:
        pass
    return {k: (k in active) for k in LABEL_KEYS}


# ── Option A: Rule-based ─────────────────────────────────────────────────────

def _rule_based(sample: dict) -> str:
    labels = _extract_label_keys(sample)
    active = [ARTIFACT_TEMPLATES[k] for k in LABEL_KEYS if labels.get(k)]
    if not active:
        return (
            "This image does not contain any visible artifacts. "
            "The generation looks physically correct."
        )
    return " ".join(active)


# ── Option B: LLM-based ──────────────────────────────────────────────────────

async def _llm_single(client, model_name: str, image: Image.Image, labels: dict) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    resp = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Artifact labels: {json.dumps(labels)}\n\n"
                            "Write the description."
                        ),
                    },
                ],
            },
        ],
        max_tokens=250,
    )
    return resp.choices[0].message.content


async def _llm_all(samples: list, batch_size: int, model_name: str) -> list[str]:
    import openai  # imported here so the module is usable without openai installed

    client  = openai.AsyncOpenAI()
    results = []

    for i in tqdm(range(0, len(samples), batch_size), desc="LLM conversion"):
        batch = samples[i : i + batch_size]
        tasks = [
            _llm_single(
                client,
                model_name,
                _load_image(s["images"]),
                _extract_label_keys(s),
            )
            for s in batch
        ]
        results.extend(await asyncio.gather(*tasks))

    return results


# ── Image loading ─────────────────────────────────────────────────────────────

def _prefetch_images(samples: list[dict], cache_dir: str = IMAGE_CACHE_DIR) -> None:
    """Download and extract all images needed for the given samples.

    Images are stored in per-part .tar files on HuggingFace. hf_hub_download
    caches tars locally so repeated runs skip re-downloading.
    """
    needed: dict[str, set[str]] = collections.defaultdict(set)
    for s in samples:
        m = re.match(r"(images_part\d+)/(.+)", s["images"])
        if m:
            needed[m.group(1) + ".tar"].add(s["images"])  # full path as member name

    for tar_name, member_paths in tqdm(needed.items(), desc="Fetching image archives"):
        missing = {p for p in member_paths if not os.path.exists(os.path.join(cache_dir, p))}
        if not missing:
            continue

        part = tar_name.replace(".tar", "")
        os.makedirs(os.path.join(cache_dir, part), exist_ok=True)

        print(f"  Downloading {tar_name} via HF Hub ({len(missing)} images needed)…")
        tar_path = hf_hub_download(
            repo_id="wj-inf/MagicData340k",
            repo_type="dataset",
            filename=tar_name,
        )

        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if member.name in missing:
                    img_bytes = tf.extractfile(member)
                    if img_bytes is None:
                        continue
                    out_path = os.path.join(cache_dir, member.name)
                    with open(out_path, "wb") as f:
                        f.write(img_bytes.read())


def _load_image(images_path: str, cache_dir: str = IMAGE_CACHE_DIR) -> Image.Image:
    return Image.open(os.path.join(cache_dir, images_path)).convert("RGB")


# ── VQA builder ──────────────────────────────────────────────────────────────

_VQA_FEATURES = Features({
    "image":              DatasetImage(),
    "user_prompt":        Value("string"),
    "assistant_response": Value("string"),
})


def _build_vqa_sample(image: Image.Image, description: str) -> dict:
    return {
        "image":              image,
        "user_prompt":        USER_PROMPT,
        "assistant_response": description,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    print(f"=== Step 2: Label → VQA Conversion  (mode={args.mode}) ===\n")

    subset_ds = load_from_disk("./data/subset_ds")
    samples   = list(subset_ds)
    print(f"Loaded {len(samples):,} samples from ./data/subset_ds")

    if args.debug:
        # Limit to samples from one tar file for fast pipeline validation
        first_part = re.match(r"(images_part\d+)/", samples[0]["images"]).group(1)
        samples = [s for s in samples if s["images"].startswith(first_part + "/")][:100]
        print(f"[debug] Using {len(samples)} samples from {first_part}")
    elif args.max_samples:
        # Take the first N samples — they tend to cluster in fewer tars, minimising downloads
        samples = samples[: args.max_samples]
        print(f"[max-samples] Using {len(samples)} samples")

    _prefetch_images(samples)

    if args.mode == "rule":
        descriptions = [
            _rule_based(s) for s in tqdm(samples, desc="Rule-based conversion")
        ]
    else:
        descriptions = asyncio.run(
            _llm_all(samples, batch_size=args.batch_size, model_name=args.model)
        )

    vqa_samples = [_build_vqa_sample(_load_image(s["images"]), d)
                   for s, d in tqdm(zip(samples, descriptions), total=len(samples), desc="Building VQA pairs")]

    vqa_ds = Dataset.from_list(vqa_samples, features=_VQA_FEATURES)
    splits  = vqa_ds.train_test_split(test_size=0.1, seed=42)

    os.makedirs("./data", exist_ok=True)
    splits["train"].save_to_disk("./data/vqa_train")
    splits["test"].save_to_disk("./data/vqa_val")

    print(f"\nTrain : {len(splits['train']):,}")
    print(f"Val   : {len(splits['test']):,}")
    print("Saved → ./data/vqa_train  and  ./data/vqa_val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["rule", "llm"], default="rule",
        help="Conversion strategy: 'rule' (free) or 'llm' (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model for LLM mode (gpt-4o-mini recommended for cost)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Async batch size for LLM mode",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Quick validation: 100 samples from one tar only (avoids full 35-tar download)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the number of samples processed (e.g. --max-samples 500)",
    )
    main(parser.parse_args())
