#!/usr/bin/env python3
"""
Prepare the Forest Cover Type dataset for the classification bench.

This script reads the Kaggle CSV, extracts features and labels, and emits NumPy arrays
plus dataset metadata inside src/benchmarks/forest/workspace/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Forest Cover Type dataset has 54 feature columns:
# - 10 quantitative (Elevation, Aspect, Slope, distances, hillshade)
# - 4 binary wilderness area indicators
# - 40 binary soil type indicators
QUANTITATIVE_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

WILDERNESS_COLUMNS = [f"Wilderness_Area{i}" for i in range(1, 5)]

SOIL_COLUMNS = [f"Soil_Type{i}" for i in range(1, 41)]

FEATURE_COLUMNS = QUANTITATIVE_COLUMNS + WILDERNESS_COLUMNS + SOIL_COLUMNS

TARGET_COLUMN = "Cover_Type"

RAW_REQUIRED = ("train.csv",)


def _verify_raw_dir(raw_dir: Path) -> None:
    missing = [name for name in RAW_REQUIRED if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw files: "
            + ", ".join(missing)
            + f" in {raw_dir}. Place the Kaggle CSVs there (manual download or scripts/fetch_forest.py --local-source)."
        )


def _compute_context(df: pd.DataFrame) -> dict:
    """Compute dataset context for the context layer."""
    quant_stats = {
        col: {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
        for col in QUANTITATIVE_COLUMNS
    }

    cover_type_counts = df[TARGET_COLUMN].value_counts().sort_index().to_dict()
    cover_type_counts = {str(k): int(v) for k, v in cover_type_counts.items()}

    return {
        "task": "multi-class forest cover type classification",
        "target": TARGET_COLUMN,
        "num_rows": int(len(df)),
        "num_features": len(FEATURE_COLUMNS),
        "num_classes": int(df[TARGET_COLUMN].nunique()),
        "feature_groups": {
            "quantitative": QUANTITATIVE_COLUMNS,
            "wilderness_area": WILDERNESS_COLUMNS,
            "soil_type": SOIL_COLUMNS,
        },
        "quantitative_stats": quant_stats,
        "cover_type_distribution": cover_type_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Forest Cover Type dataset arrays.")
    parser.add_argument(
        "--raw-dir",
        default="kaggle-data/forest/raw",
        help="Directory containing train.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="src/benchmarks/forest/workspace",
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

    # Extract features and labels
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32 if args.float32 else None)
    labels = df[TARGET_COLUMN].to_numpy(dtype=np.int64)

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "labels.npy", labels)

    # Dataset context for the context layer
    context = _compute_context(df)
    (output_dir / "dataset_context.json").write_text(json.dumps(context, indent=2))

    # Preparation metadata
    meta = {
        "features_path": str(output_dir / "features.npy"),
        "labels_path": str(output_dir / "labels.npy"),
        "num_rows": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "num_classes": int(len(np.unique(labels))),
        "dtype": "float32" if args.float32 else "float64",
        "target": TARGET_COLUMN,
    }
    (output_dir / "prepared_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[prepare-forest] Saved arrays and metadata to {output_dir}")
    print(f"[prepare-forest] {features.shape[0]} samples, {features.shape[1]} features, {meta['num_classes']} classes")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[prepare-forest] Error: {exc}", file=sys.stderr)
        sys.exit(1)
