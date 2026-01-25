#!/usr/bin/env python3
"""
Tiny logistic-regression trainer for the toy tabular bench.
Generates (and caches) a synthetic dataset, trains, and emits results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

WORKSPACE = Path(__file__).resolve().parent
DATA_PATH = WORKSPACE / "data.npy"
LABELS_PATH = WORKSPACE / "labels.npy"
CONFIG_PATH = WORKSPACE / "run_config.json"
RESULTS_PATH = WORKSPACE / "results.json"
ALL_RESULTS_PATH = WORKSPACE / "all_results.json"


def _load_or_create_dataset(random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if DATA_PATH.exists() and LABELS_PATH.exists():
        return np.load(DATA_PATH), np.load(LABELS_PATH)

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        n_classes=2,
        class_sep=1.5,
        random_state=random_state,
    )
    np.save(DATA_PATH, X)
    np.save(LABELS_PATH, y)
    return X, y


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    seed = int(cfg.get("random_seed", 0))
    test_size = float(cfg.get("test_size", 0.2))
    C = float(cfg.get("C", 1.0))
    max_iter = int(cfg.get("max_iter", 100))

    X, y = _load_or_create_dataset(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    if ALL_RESULTS_PATH.exists():
        with ALL_RESULTS_PATH.open("r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    new_result = {
        "accuracy": float(accuracy),
        "C": C,
        "max_iter": max_iter,
        "test_size": test_size,
        "random_seed": seed,
    }
    all_results.append(new_result)

    with ALL_RESULTS_PATH.open("w") as f:
        json.dump(all_results, f, indent=2)
    
    RESULTS_PATH.write_text(json.dumps(new_result, indent=2))
    print(json.dumps(new_result))


if __name__ == "__main__":
    main()


