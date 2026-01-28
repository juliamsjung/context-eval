#!/usr/bin/env python3
"""
Prepare the Mercor AI Detection dataset for the AI text detection bench.

This script reads the Kaggle CSV, extracts text and binary labels,
and emits arrays plus dataset metadata inside src/benchmarks/mercor/workspace/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_dataframe(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    required = ["id", "topic", "answer", "is_cheating"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {train_path}: {missing}")
    return df


def _compute_context(df: pd.DataFrame) -> dict:
    # Text length statistics
    answer_lengths = df["answer"].str.len()
    topic_lengths = df["topic"].str.len()
    
    # Label distribution
    positive_count = int(df["is_cheating"].sum())
    negative_count = int(len(df) - positive_count)
    
    return {
        "task": "binary classification",
        "target": "is_cheating",
        "target_description": "0 = authentic human writing, 1 = AI-assisted/generated",
        "num_rows": int(len(df)),
        "num_topics": int(df["topic"].nunique()),
        "label_distribution": {
            "authentic (0)": negative_count,
            "ai_assisted (1)": positive_count,
            "positive_rate": float(df["is_cheating"].mean()),
        },
        "answer_length_stats": {
            "mean": float(answer_lengths.mean()),
            "std": float(answer_lengths.std()),
            "min": int(answer_lengths.min()),
            "max": int(answer_lengths.max()),
            "median": float(answer_lengths.median()),
        },
        "topic_length_stats": {
            "mean": float(topic_lengths.mean()),
            "std": float(topic_lengths.std()),
            "min": int(topic_lengths.min()),
            "max": int(topic_lengths.max()),
        },
    }


RAW_REQUIRED = ("train.csv",)


def _verify_raw_dir(raw_dir: Path) -> None:
    missing = [name for name in RAW_REQUIRED if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw files: "
            + ", ".join(missing)
            + f" in {raw_dir}. Place the Kaggle CSVs there (manual download or scripts/fetch_mercor.py --local-source)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mercor AI Detection dataset arrays.")
    parser.add_argument(
        "--raw-dir",
        default="kaggle-data/mercor/raw",
        help="Directory containing train.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="src/benchmarks/mercor/workspace",
        help="Directory to store workspace-ready artifacts",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for development/testing)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _verify_raw_dir(raw_dir)
    train_path = raw_dir / "train.csv"

    df = _load_dataframe(train_path)

    if args.max_samples and args.max_samples < len(df):
        df = df.head(args.max_samples)
        print(f"[prepare-mercor] Limited to {args.max_samples} samples")

    # Extract text columns as object arrays (variable length strings)
    ids = df["id"].values
    topics = df["topic"].values
    answers = df["answer"].values

    # Extract binary labels
    labels = df["is_cheating"].to_numpy(dtype=np.int8)

    np.save(output_dir / "ids.npy", ids, allow_pickle=True)
    np.save(output_dir / "topics.npy", topics, allow_pickle=True)
    np.save(output_dir / "answers.npy", answers, allow_pickle=True)
    np.save(output_dir / "labels.npy", labels)

    context = _compute_context(df)
    (output_dir / "dataset_context.json").write_text(json.dumps(context, indent=2))

    meta = {
        "ids_path": str(output_dir / "ids.npy"),
        "topics_path": str(output_dir / "topics.npy"),
        "answers_path": str(output_dir / "answers.npy"),
        "labels_path": str(output_dir / "labels.npy"),
        "num_rows": int(len(df)),
        "task": "binary classification",
        "target": "is_cheating",
    }
    (output_dir / "prepared_meta.json").write_text(json.dumps(meta, indent=2))

    positive_rate = df["is_cheating"].mean() * 100
    print(f"[prepare-mercor] Saved arrays and metadata to {output_dir}")
    print(f"[prepare-mercor] {len(df)} samples, {positive_rate:.1f}% AI-assisted")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[prepare-mercor] Error: {exc}", file=sys.stderr)
        sys.exit(1)
