#!/usr/bin/env python3
"""
Download the Forest Cover Type Prediction Kaggle dataset into kaggle-data/forest/raw.

Usage:
    python scripts/fetch_forest.py
    python scripts/fetch_forest.py --local-source /path/to/local/csvs
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _copy_local(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)
    print(f"[fetch-forest] Copied local files from {source} into {dest}")


def _download_kaggle(
    slug: str,
    target_dir: Path,
    filename: str | None,
    dataset: str | None,
    unzip: bool,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        if slug.startswith("competitions/"):
            comp_name = slug.split("/", 1)[1]
            if filename:
                print(f"[fetch-forest] Downloading competition file {filename} from {slug}")
                api.competition_download_file(comp_name, filename, path=str(target_dir))
            else:
                print(f"[fetch-forest] Downloading competition bundle {slug} to {target_dir}")
                api.competition_download_files(comp_name, path=str(target_dir))
        elif dataset:
            if filename:
                print(f"[fetch-forest] Downloading dataset file {filename} from {dataset}")
                api.dataset_download_file(dataset, filename, path=str(target_dir))
            else:
                print(f"[fetch-forest] Downloading dataset {dataset} to {target_dir}")
                api.dataset_download_files(dataset, path=str(target_dir))

        if not unzip:
            print("[fetch-forest] Skipped auto-unzip per --no-unzip flag.")

    except Exception as exc:
        raise RuntimeError(f"Kaggle download failed: {exc}") from exc
    print(f"[fetch-forest] Dataset ready in {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or stage the Forest Cover Type dataset.")
    parser.add_argument(
        "--slug",
        default="competitions/forest-cover-type-prediction",
        help="Kaggle competition or dataset slug.",
    )
    parser.add_argument(
        "--target-dir",
        default="kaggle-data/forest/raw",
        help="Directory to store raw data.",
    )
    parser.add_argument("--filename", default=None, help="Specific file to download.")
    parser.add_argument("--dataset", default=None, help="Kaggle dataset slug (for datasets API).")
    parser.add_argument(
        "--local-source",
        default=None,
        help="Path to local copy of extracted CSVs (skips Kaggle download).",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Skip auto-unzip after download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir).expanduser().resolve()

    if args.local_source:
        _copy_local(Path(args.local_source).expanduser().resolve(), target_dir)
        return

    # Check if already downloaded
    if not args.force and (target_dir / "train.csv").exists():
        print(f"[fetch-forest] train.csv already present in {target_dir}. Use --force to overwrite.")
        return

    _download_kaggle(
        slug=args.slug,
        target_dir=target_dir,
        filename=args.filename,
        dataset=args.dataset,
        unzip=not args.no_unzip,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fetch-forest] Error: {exc}", file=sys.stderr)
        sys.exit(1)
