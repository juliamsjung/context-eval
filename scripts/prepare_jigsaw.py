#!/usr/bin/env python3
"""
Prepare the Jigsaw Toxic Comment Classification dataset for the toxicity bench.

This script reads the Kaggle CSV, extracts text and multi-label targets, and emits arrays
plus dataset metadata inside src/benchmarks/jigsaw/workspace/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# The 6 toxicity label columns
LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _load_dataframe(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    required_cols = ["id", "comment_text"] + LABEL_COLUMNS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {train_path}: {missing}")
    return df


def _compute_context(df: pd.DataFrame) -> dict:
    """Compute dataset context for LLM prompt."""
    text_lengths = df["comment_text"].str.len()
    label_stats = {col: int(df[col].sum()) for col in LABEL_COLUMNS}
    return {
        "task": "multi-label toxicity classification",
        "target": "6 binary toxicity labels",
        "num_rows": int(len(df)),
        "num_labels": len(LABEL_COLUMNS),
        "label_names": LABEL_COLUMNS,
        "label_distribution": label_stats,
        "text_stats": {
            "mean_length": float(text_lengths.mean()),
            "min_length": int(text_lengths.min()),
            "max_length": int(text_lengths.max()),
            "median_length": float(text_lengths.median()),
        },
    }


RAW_REQUIRED = ("train.csv",)


def _verify_raw_dir(raw_dir: Path) -> None:
    missing = [name for name in RAW_REQUIRED if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw files: "
            + ", ".join(missing)
            + f" in {raw_dir}. Place the Kaggle CSVs there (manual download or scripts/fetch_jigsaw.py --local-source)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Jigsaw Toxic Comment dataset arrays.")
    parser.add_argument(
        "--raw-dir",
        default="kaggle-data/jigsaw/raw",
        help="Directory containing train.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="src/benchmarks/jigsaw/workspace",
        help="Directory to store workspace-ready artifacts",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for testing/debugging)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _verify_raw_dir(raw_dir)
    train_path = raw_dir / "train.csv"

    df = _load_dataframe(train_path)

    # Optionally limit samples
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"[prepare-jigsaw] Limited to {args.max_samples} samples")

    # Extract IDs, texts, and labels
    ids = df["id"].tolist()
    texts = df["comment_text"].tolist()
    labels = df[LABEL_COLUMNS].to_numpy(dtype=np.int32)

    # Save texts and IDs as JSON (text data doesn't compress well as numpy)
    (output_dir / "ids.json").write_text(json.dumps(ids, ensure_ascii=False))
    (output_dir / "texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=None))

    # Save labels as numpy array (shape: num_samples x 6)
    np.save(output_dir / "labels.npy", labels)

    # Compute and save context
    context = _compute_context(df)
    (output_dir / "dataset_context.json").write_text(json.dumps(context, indent=2))

    # Save label column names
    label_mapping = {name: idx for idx, name in enumerate(LABEL_COLUMNS)}
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2))

    # Save metadata
    meta = {
        "ids_path": str(output_dir / "ids.json"),
        "texts_path": str(output_dir / "texts.json"),
        "labels_path": str(output_dir / "labels.npy"),
        "num_rows": int(len(df)),
        "num_labels": len(LABEL_COLUMNS),
        "task": "multi-label toxicity classification",
    }
    (output_dir / "prepared_meta.json").write_text(json.dumps(meta, indent=2))

    # Print summary
    print(f"[prepare-jigsaw] Saved arrays and metadata to {output_dir}")
    print(f"[prepare-jigsaw] {len(df)} samples, {len(LABEL_COLUMNS)} labels")
    print(f"[prepare-jigsaw] Label counts: {dict(zip(LABEL_COLUMNS, labels.sum(axis=0)))}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[prepare-jigsaw] Error: {exc}", file=sys.stderr)
        sys.exit(1)
