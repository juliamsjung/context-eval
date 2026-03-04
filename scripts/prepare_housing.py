#!/usr/bin/env python3
"""
Prepare the California Housing dataset for the regression bench.

This script reads the Kaggle CSV, extracts features and targets, and emits NumPy arrays
plus dataset metadata inside src/benchmarks/housing/workspace/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# California Housing dataset has 8 numeric feature columns
FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET_COLUMN = "MedHouseVal"

RAW_REQUIRED = ("train.csv",)


def _verify_raw_dir(raw_dir: Path) -> None:
    missing = [name for name in RAW_REQUIRED if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw files: "
            + ", ".join(missing)
            + f" in {raw_dir}. Place the Kaggle CSVs there (manual download or scripts/fetch_housing.py --local-source)."
        )


def _compute_context(df: pd.DataFrame) -> dict:
    """Compute dataset context for the context layer."""
    feature_stats = {
        col: {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
        for col in FEATURE_COLUMNS
    }
    target_series = df[TARGET_COLUMN]
    return {
        "task": "regression – predict California median house values",
        "target": TARGET_COLUMN,
        "num_rows": int(len(df)),
        "num_features": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "target_stats": {
            "mean": float(target_series.mean()),
            "std": float(target_series.std()),
            "min": float(target_series.min()),
            "max": float(target_series.max()),
        },
        "feature_stats": feature_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare California Housing dataset arrays.")
    parser.add_argument(
        "--raw-dir",
        default="kaggle-data/housing/raw",
        help="Directory containing train.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="src/benchmarks/housing/workspace",
        help="Directory to store workspace-ready artifacts",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Store arrays as float32 (default: float64)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _verify_raw_dir(raw_dir)
    train_path = raw_dir / "train.csv"

    df = pd.read_csv(train_path)

    # Validate columns exist
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {train_path}: {missing}")

    # Extract features and targets
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32 if args.float32 else None)
    targets = df[TARGET_COLUMN].to_numpy(dtype=np.float32 if args.float32 else None)

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "targets.npy", targets)

    # Dataset context for the context layer
    context = _compute_context(df)
    (output_dir / "dataset_context.json").write_text(json.dumps(context, indent=2))

    # Preparation metadata
    meta = {
        "features_path": str(output_dir / "features.npy"),
        "targets_path": str(output_dir / "targets.npy"),
        "num_rows": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "dtype": "float32" if args.float32 else "float64",
        "target": TARGET_COLUMN,
    }
    (output_dir / "prepared_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[prepare-housing] Saved arrays and metadata to {output_dir}")
    print(f"[prepare-housing] {features.shape[0]} samples, {features.shape[1]} features")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[prepare-housing] Error: {exc}", file=sys.stderr)
        sys.exit(1)
