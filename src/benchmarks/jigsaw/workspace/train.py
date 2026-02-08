#!/usr/bin/env python3
"""
Multi-label text classification trainer for the Jigsaw Toxic Comment bench.
Uses TF-IDF + Logistic Regression for fast iteration during experiments.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

WORKSPACE = Path(__file__).resolve().parent
TEXTS_PATH = WORKSPACE / "texts.json"
LABELS_PATH = WORKSPACE / "labels.npy"
CONFIG_PATH = WORKSPACE / "run_config.json"
RESULTS_PATH = WORKSPACE / "results.json"


def _load_data() -> tuple[list[str], np.ndarray]:
    if not TEXTS_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Prepared data not found. Run scripts/prepare_jigsaw.py first."
        )
    texts = json.loads(TEXTS_PATH.read_text())
    labels = np.load(LABELS_PATH)
    if len(texts) != len(labels):
        raise ValueError("Text and label counts do not match.")
    return texts, labels


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    random_seed = int(cfg.get("random_seed", 42))
    test_size = float(cfg.get("test_size", 0.2))

    # TF-IDF hyperparameters (tunable by LLM)
    max_features = int(cfg.get("max_features", 10000))
    ngram_max = int(cfg.get("ngram_max", 2))
    min_df = int(cfg.get("min_df", 5))

    # Logistic Regression hyperparameters (tunable by LLM)
    C = float(cfg.get("C", 1.0))
    max_iter = int(cfg.get("max_iter", 100))

    texts, labels = _load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_seed,
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        strip_accents="unicode",
        lowercase=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Multi-label classification with OneVsRest
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_seed,
            solver="lbfgs",
        )
    )
    classifier.fit(X_train_tfidf, y_train)

    # Predict probabilities for ROC AUC
    y_pred_proba = classifier.predict_proba(X_test_tfidf)

    # Compute per-column AUC and mean (Jigsaw evaluation metric)
    auc_scores = []
    for i in range(labels.shape[1]):
        try:
            auc = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
            auc_scores.append(auc)
        except ValueError:
            # Skip if only one class present in test set
            pass

    mean_auc = float(np.mean(auc_scores)) if auc_scores else 0.0

    metrics = {
        "mean_auc": mean_auc,
        "num_labels_scored": len(auc_scores),
    }
    metric_name = "mean_auc"
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
