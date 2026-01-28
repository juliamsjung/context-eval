#!/usr/bin/env python3
"""
Prepare the Leaf Classification dataset for the species classification bench.

This script reads the Kaggle CSV, extracts features, and emits NumPy arrays
plus dataset metadata inside src/benchmarks/leaf/workspace/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Leaf dataset has 192 numeric feature columns (margin, shape, texture features)
# We'll extract all numeric columns except 'id' and 'species'
EXCLUDE_COLUMNS: Sequence[str] = ("id", "species")


def _load_dataframe(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    if "species" not in df.columns:
        raise ValueError(f"Column 'species' missing from {train_path}")
    return df


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all numeric feature columns, excluding id and species."""
    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            feature_cols.append(col)
    return feature_cols


def _compute_context(df: pd.DataFrame, feature_cols: list[str], label_names: list[str]) -> dict:
    feature_stats = {
        col: {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
        for col in feature_cols
    }
    return {
        "task": "multi-class classification",
        "target": "species",
        "num_rows": int(len(df)),
        "num_features": len(feature_cols),
        "num_classes": len(label_names),
        "feature_names": feature_cols,
        "class_names": label_names,
        "feature_stats": feature_stats,
    }


RAW_REQUIRED = ("train.csv",)


def _verify_raw_dir(raw_dir: Path) -> None:
    missing = [name for name in RAW_REQUIRED if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw files: "
            + ", ".join(missing)
            + f" in {raw_dir}. Place the Kaggle CSVs there (manual download or scripts/fetch_leaf.py --local-source)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Leaf Classification dataset arrays.")
    parser.add_argument(
        "--raw-dir",
        default="kaggle-data/leaf/raw",
        help="Directory containing train.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="src/benchmarks/leaf/workspace",
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

    df = _load_dataframe(train_path)
    feature_cols = _get_feature_columns(df)

    if not feature_cols:
        raise ValueError("No numeric feature columns found in the dataset.")

    # Extract features
    dtype = np.float32 if args.float32 else np.float64
    features = df.loc[:, feature_cols].to_numpy(dtype=dtype)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["species"])
    label_names = list(label_encoder.classes_)

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "labels.npy", labels)

    # Check for images and create image path mapping
    images_dir = raw_dir / "images"
    image_paths = []
    has_images = images_dir.exists() and images_dir.is_dir()
    
    if has_images:
        for sample_id in df["id"].values:
            img_path = images_dir / f"{sample_id}.jpg"
            if img_path.exists():
                image_paths.append(str(img_path))
            else:
                image_paths.append("")  # Missing image
        np.save(output_dir / "image_paths.npy", np.array(image_paths, dtype=object), allow_pickle=True)
        print(f"[prepare-leaf] Found {sum(1 for p in image_paths if p)} images")

    context = _compute_context(df, feature_cols, label_names)
    context["has_images"] = has_images
    if has_images:
        context["images_dir"] = str(images_dir)
        context["num_images"] = sum(1 for p in image_paths if p)
    (output_dir / "dataset_context.json").write_text(json.dumps(context, indent=2))

    # Save label encoder mapping
    label_mapping = {name: int(idx) for idx, name in enumerate(label_names)}
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2))

    meta = {
        "features_path": str(output_dir / "features.npy"),
        "labels_path": str(output_dir / "labels.npy"),
        "num_rows": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "num_classes": len(label_names),
        "dtype": "float32" if args.float32 else "float64",
        "task": "multi-class classification",
        "has_images": has_images,
    }
    if has_images:
        meta["image_paths_path"] = str(output_dir / "image_paths.npy")
    (output_dir / "prepared_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[prepare-leaf] Saved arrays and metadata to {output_dir}")
    print(f"[prepare-leaf] {features.shape[0]} samples, {features.shape[1]} features, {len(label_names)} classes")



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[prepare-leaf] Error: {exc}", file=sys.stderr)
        sys.exit(1)
