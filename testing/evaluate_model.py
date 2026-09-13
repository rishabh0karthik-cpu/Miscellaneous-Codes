"""Run an objective, chronological diagnostic against the dummy current data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from quadratic_current_model import QuadraticCurrentPredictor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(__file__).with_name("dummy_current_patterns_2006_2025.csv")
REPORT_PATH = Path(__file__).with_name("model_diagnostic_report.json")
ROC_PATH = Path(__file__).with_name("roc_curve_points.csv")


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dates = np.array([row["date"] for row in rows])
    features = np.array([[float(row[f"feature_{index:02d}"]) for index in range(30)] for row in rows])
    targets = np.array([
        [float(row[f"current_{location}_{component}"]) for location in range(1, 6) for component in ("u", "v")]
        for row in rows
    ])
    return dates, features, targets


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2_uniform_average": float(r2_score(actual, predicted, multioutput="uniform_average")),
    }


def main() -> None:
    dates, features, targets = load_data()
    train = dates < "2022-01-01"
    validation = (dates >= "2022-01-01") & (dates < "2024-01-01")
    test = dates >= "2024-01-01"

    predictor = QuadraticCurrentPredictor()
    predictor.fit(features[train], targets[train])
    validation_predictions = predictor.predict(features[validation])
    predictions = predictor.predict(features[test])

    train_mean = np.broadcast_to(targets[train].mean(axis=0), targets[test].shape)
    persistence = np.vstack([targets[np.flatnonzero(test)[0] - 1], targets[test][:-1]])
    report = {
        "data": {
            "path": str(DATA_PATH.relative_to(ROOT)),
            "rows": int(len(dates)),
            "period": [dates[0], dates[-1]],
            "feature_count": int(features.shape[1]),
            "target_count": int(targets.shape[1]),
            "synthetic": True,
        },
        "split": {
            "train": ["2006-01-01", "2021-12-01"],
            "validation": ["2022-01-01", "2023-12-01"],
            "test": ["2024-01-01", "2025-12-01"],
            "leakage_check": bool(np.max(np.flatnonzero(train)) < np.min(np.flatnonzero(test))),
        },
        "regression": {
            "validation": regression_metrics(targets[validation], validation_predictions),
            "quadratic_ridge": regression_metrics(targets[test], predictions),
            "mean_baseline": regression_metrics(targets[test], train_mean),
            "persistence_baseline": regression_metrics(targets[test], persistence),
        },
        "classification_note": "Secondary diagnostic only: continuous targets thresholded at each training-target median; not the native regression task.",
    }

    threshold = np.median(targets[train], axis=0)
    actual_binary = (targets[test] >= threshold).ravel()
    score = predictions.ravel()
    predicted_binary = (score >= np.tile(threshold, int(test.sum()))).ravel()
    fpr, tpr, thresholds = roc_curve(actual_binary, score)
    report["classification_thresholded"] = {
        "threshold": "per-target training median",
        "accuracy": float(accuracy_score(actual_binary, predicted_binary)),
        "precision": float(precision_score(actual_binary, predicted_binary, zero_division=0)),
        "recall": float(recall_score(actual_binary, predicted_binary, zero_division=0)),
        "f1": float(f1_score(actual_binary, predicted_binary, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual_binary, score)),
    }
    report["quality_gate"] = {
        "target": 0.90,
        "all_requested_metrics_at_least_target": all(
            report["classification_thresholded"][metric] >= 0.90
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
        ),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with ROC_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["false_positive_rate", "true_positive_rate", "threshold"])
        writer.writerows(zip(fpr, tpr, thresholds))

    print(json.dumps(report, indent=2))
    print(f"Wrote {REPORT_PATH} and {ROC_PATH}")


if __name__ == "__main__":
    main()
