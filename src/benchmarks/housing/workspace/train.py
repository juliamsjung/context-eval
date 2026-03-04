#!/usr/bin/env python3
"""
Regression trainer for the California Housing bench.
Uses GradientBoostingRegressor for varied model exploration across benchmarks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

WORKSPACE = Path(__file__).resolve().parent
FEATURES_PATH = WORKSPACE / "features.npy"
TARGETS_PATH = WORKSPACE / "targets.npy"
CONFIG_PATH = WORKSPACE / "run_config.json"
RESULTS_PATH = WORKSPACE / "results.json"


def _load_arrays() -> tuple[np.ndarray, np.ndarray]:
    if not FEATURES_PATH.exists() or not TARGETS_PATH.exists():
        raise FileNotFoundError(
            "Prepared arrays not found. Run scripts/prepare_housing.py first."
        )
    X = np.load(FEATURES_PATH)
    y = np.load(TARGETS_PATH)
    if len(X) != len(y):
        raise ValueError("Feature and target array lengths do not match.")
    return X, y


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    random_seed = int(cfg.get("random_seed", 0))
    test_size = float(cfg.get("test_size", 0.2))

    # GradientBoostingRegressor hyperparameters (tunable by LLM)
    n_estimators = int(cfg.get("n_estimators", 100))
    max_depth = int(cfg.get("max_depth", 5))
    learning_rate = float(cfg.get("learning_rate", 0.1))
    subsample = float(cfg.get("subsample", 1.0))
    min_samples_split = int(cfg.get("min_samples_split", 5))
    min_samples_leaf = int(cfg.get("min_samples_leaf", 5))

    X, y = _load_arrays()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
    )

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    mse = float(mean_squared_error(y_test, preds))
    rmse = math.sqrt(mse)
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2_score(y_test, preds)),
    }
    metric_name = "rmse"  # Kaggle evaluation metric (lower is better)
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
