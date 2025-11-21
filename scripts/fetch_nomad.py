#!/usr/bin/env python3
"""
Download the NOMAD Kaggle dataset into kaggle-data/nomad/raw.

Usage:
    python scripts/fetch_nomad.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from kaggle import KaggleApi  # type: ignore[import-not-found]
except ImportError:
    KaggleApi = None  # type: ignore[assignment]


def _write_stamp(dest: Path, dataset: str) -> None:
    stamp = {
        "dataset": dataset,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    (dest / ".download_meta.json").write_text(json.dumps(stamp, indent=2))


def _local_exists(path: Path) -> bool:
    train = path / "train.csv"
    test = path / "test.csv"
    return train.exists() and test.exists()


def _copy_from_source(source: Path, dest: Path) -> None:
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        src = source / name
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}; populate your local mirror first.")
        shutil.copy2(src, dest / name)
    _write_stamp(dest, f"local:{source}")
    print(f"[fetch-nomad] Copied local files from {source} into {dest}")


def _download_from_kaggle(dataset: str, target_dir: Path, filename: str | None, unzip: bool) -> None:
    if KaggleApi is None:
        raise RuntimeError("kaggle package not installed; install it or supply --local-source.")
    api = KaggleApi()
    api.authenticate()

    is_competition = dataset.startswith("competitions/")
    if is_competition:
        slug = dataset.split("/", 1)[1]
        if filename:
            print(f"[fetch-nomad] Downloading competition file {filename} from {slug}")
            api.competition_download_file(slug, filename, path=str(target_dir), quiet=False)
        else:
            print(f"[fetch-nomad] Downloading competition bundle {slug} to {target_dir}")
            api.competition_download_files(slug, path=str(target_dir), quiet=False)
    else:
        if filename:
            print(f"[fetch-nomad] Downloading dataset file {filename} from {dataset}")
            api.dataset_download_file(dataset, filename, path=str(target_dir), quiet=False)
        else:
            print(f"[fetch-nomad] Downloading dataset {dataset} to {target_dir}")
            api.dataset_download_files(dataset, path=str(target_dir), quiet=False, unzip=unzip)

    if not unzip and not is_competition:
        print("[fetch-nomad] Skipped auto-unzip per --no-unzip flag.")

    _write_stamp(target_dir, dataset)
    print(f"[fetch-nomad] Dataset ready in {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or stage the NOMAD dataset.")
    parser.add_argument(
        "--dataset",
        default="competitions/nomad2018-predict-transparent-conductors",
        help="Kaggle slug (use competitions/<slug> for competition files).",
    )
    parser.add_argument(
        "--target-dir",
        default="kaggle-data/nomad/raw",
        help="Directory where raw files should reside.",
    )
    parser.add_argument(
        "--local-source",
        help="Optional path containing manually downloaded train/test CSVs to copy instead of hitting Kaggle.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--filename",
        help="Download a single file (only relevant when using Kaggle API).",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Skip unzip when using dataset download endpoint.",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if _local_exists(target_dir) and not args.force:
        print(f"[fetch-nomad] train.csv/test.csv already present in {target_dir}. Use --force to overwrite.")
        return

    if args.local_source:
        source = Path(args.local_source).expanduser().resolve()
        if not _local_exists(source):
            raise FileNotFoundError(
                f"Local source {source} missing train.csv/test.csv. Populate it first."
            )
        _copy_from_source(source, target_dir)
        return

    _download_from_kaggle(args.dataset, target_dir, args.filename, not args.no_unzip)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[fetch-nomad] Error: {exc}", file=sys.stderr)
        sys.exit(1)

