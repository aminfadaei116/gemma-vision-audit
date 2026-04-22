"""
Step 1 — Download MagicData340K and create a stratified subset.

Saves: ./data/subset_ds  (HF arrow format)

Usage:
    python scripts/01_download_and_sample.py
    python scripts/01_download_and_sample.py --artifact-col has_artifact --n-artifact 15000 --n-clean 5000
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from datasets import Dataset, load_dataset


def main(args: argparse.Namespace) -> None:
    print("=== Step 1: Download and Sample MagicData340K ===\n")

    print("Loading dataset from HuggingFace (this may take a while)...")
    dataset = load_dataset(
        "json",
        data_files={"train": "hf://datasets/wj-inf/MagicData340k/magic_mirror_data_train.jsonl"},
        split="train",
    )
    print(f"Total samples : {len(dataset):,}")
    print(f"Columns       : {dataset.column_names}")
    print("\nSample [0]:")
    print(dataset[0])

    df = dataset.to_pandas()

    # Derive has_artifact from the nested response dict/string if needed.
    # response = '{"Whether Normal": true/false}'  → true = clean, false = has artifact
    if "has_artifact" not in df.columns and "response" in df.columns:
        def _whether_normal(r):
            if isinstance(r, str):
                r = json.loads(r)
            return r.get("Whether Normal", True) if isinstance(r, dict) else True

        df["has_artifact"] = df["response"].apply(lambda r: 0 if _whether_normal(r) else 1)
        print("Derived 'has_artifact' from response['Whether Normal']")

    artifact_col = args.artifact_col
    if artifact_col not in df.columns:
        print(
            f"\n[ERROR] Column '{artifact_col}' not found.\n"
            f"Available columns: {df.columns.tolist()}\n"
            "Re-run with --artifact-col set to the correct binary label column."
        )
        sys.exit(1)

    print(f"\nClass distribution for '{artifact_col}':")
    print(df[artifact_col].value_counts().to_string())

    n_artifact = min(args.n_artifact, int((df[artifact_col] == 1).sum()))
    n_clean    = min(args.n_clean,    int((df[artifact_col] == 0).sum()))

    artifact_df = df[df[artifact_col] == 1].sample(n=n_artifact, random_state=42)
    clean_df    = df[df[artifact_col] == 0].sample(n=n_clean,    random_state=42)

    subset_df = pd.concat([artifact_df, clean_df]).sample(frac=1, random_state=42)
    subset_ds = Dataset.from_pandas(subset_df.reset_index(drop=True))

    print(f"\nSubset: {len(subset_ds):,} samples  ({n_artifact} artifact / {n_clean} clean)")

    os.makedirs("./data", exist_ok=True)
    subset_ds.save_to_disk("./data/subset_ds")
    print("Saved → ./data/subset_ds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-col", default="has_artifact",
        help="Binary label column name (1 = has artifact, 0 = clean)"
    )
    parser.add_argument("--n-artifact", type=int, default=15000)
    parser.add_argument("--n-clean",    type=int, default=5000)
    main(parser.parse_args())
