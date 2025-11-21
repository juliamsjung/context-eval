#!/usr/bin/env python3
"""
Prepare the NOMAD dataset for the bandgap regression bench.

This script reads the Kaggle CSV, filters/encodes columns, and emits NumPy arrays
plus dataset metadata inside benchmarks/nomad/workspace/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


FEATURE_COLUMNS: Sequence[str] = (
    "spacegroup",
    "number_of_total_atoms",
    "percent_atom_al",
    "percent_atom_ga",
    "percent_atom_in",
    "lattice_vector_1_ang",
    "lattice_vector_2_ang",
    "lattice_vector_3_ang",
    "lattice_angle_alpha_degree",
    "lattice_angle_beta_degree",
    "lattice_angle_gamma_degree",
    "formation_energy_ev_natom",
)


def _load_dataframe(train_path: Path, target: str, drop_cols: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    missing = [col for col in FEATURE_COLUMNS + (target,) if col not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {train_path}: {missing}")
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    df = df.dropna(subset=[target])
    return df


def _compute_context(df: pd.DataFrame, target: str) -> dict:
    feature_stats = {
        col: {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
        for col in FEATURE_COLUMNS
    }
    target_series = df[target]
    return {
        "target": target,
        "num_rows": int(len(df)),
        "num_features": len(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "target_stats": {
            "mean": float(target_series.mean()),
            "std": float(target_series.std()),
            "min": float(target_series.min()),
            "max": float(target_series.max()),
        },
        "feature_stats": feature_stats,
    }


RAW_REQUIRED = ("train.csv", "test.csv")


def _verify_raw_dir(raw_dir: Path) -> None:
    missing = [name for name in RAW_REQUIRED if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw files: "
            + ", ".join(missing)
            + f" in {raw_dir}. Place the Kaggle CSVs there (manual download or scripts/fetch_nomad.py --local-source)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NOMAD dataset arrays.")
    parser.add_argument(
        "--raw-dir",
        default="kaggle-data/nomad/raw",
        help="Directory containing train.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/nomad/workspace",
        help="Directory to store workspace-ready artifacts",
    )
    parser.add_argument(
        "--target",
        default="bandgap_energy_ev",
        help="Regression target column",
    )
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=["id"],
        help="Columns to drop prior to modeling",
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

    df = _load_dataframe(train_path, args.target, args.drop_cols)
    features = df.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32 if args.float32 else None)
    targets = df[args.target].to_numpy(dtype=np.float32 if args.float32 else None)

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "targets.npy", targets)

    context = _compute_context(df, args.target)
    (output_dir / "dataset_context.json").write_text(json.dumps(context, indent=2))

    meta = {
        "features_path": str(output_dir / "features.npy"),
        "targets_path": str(output_dir / "targets.npy"),
        "num_rows": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "dtype": "float32" if args.float32 else "float64",
        "target": args.target,
    }
    (output_dir / "prepared_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[prepare-nomad] Saved arrays and metadata to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[prepare-nomad] Error: {exc}", file=sys.stderr)
        sys.exit(1)

