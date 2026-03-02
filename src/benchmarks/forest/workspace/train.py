#!/usr/bin/env python3
"""
Multi-class classification trainer for the Forest Cover Type bench.
Uses RandomForestClassifier for fast iteration during experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

WORKSPACE = Path(__file__).resolve().parent
FEATURES_PATH = WORKSPACE / "features.npy"
LABELS_PATH = WORKSPACE / "labels.npy"
CONFIG_PATH = WORKSPACE / "run_config.json"
RESULTS_PATH = WORKSPACE / "results.json"


def _load_arrays() -> tuple[np.ndarray, np.ndarray]:
    if not FEATURES_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Prepared arrays not found. Run scripts/prepare_forest.py first."
        )
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    if len(X) != len(y):
        raise ValueError("Feature and label array lengths do not match.")
    return X, y


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    random_seed = int(cfg.get("random_seed", 0))
    test_size = float(cfg.get("test_size", 0.2))

    # RandomForest hyperparameters (tunable by LLM)
    n_estimators = int(cfg.get("n_estimators", 100))
    max_depth = int(cfg.get("max_depth", 10))
    min_samples_split = int(cfg.get("min_samples_split", 5))
    min_samples_leaf = int(cfg.get("min_samples_leaf", 2))
    max_features = float(cfg.get("max_features", 0.5))

    X, y = _load_arrays()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_seed,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)

    accuracy = float(accuracy_score(y_test, preds))
    f1_weighted = float(f1_score(y_test, preds, average="weighted"))

    metrics = {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
    }
    metric_name = "accuracy"  # Kaggle evaluation metric (higher is better)
    metric_value = metrics[metric_name]

    results = {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metrics": metrics,
        "config": cfg,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
