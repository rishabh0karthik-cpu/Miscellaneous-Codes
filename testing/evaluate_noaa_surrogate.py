"""Evaluate the forest on the NOAA-fitted surrogate, separately from real data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = Path(__file__).with_name("noaa_surrogate_patterns.csv")
REPORT_PATH = Path(__file__).with_name("noaa_surrogate_diagnostic_report.json")


def main() -> None:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    feature_count = sum(key.startswith("feature_") for key in rows[0])
    features = np.asarray([[float(row[f"feature_{index}"]) for index in range(feature_count)] for row in rows])
    targets = np.asarray([[float(row[f"target_current_{index}"]) for index in range(10)] for row in rows])
    split = int(len(rows) * 0.8)

    model = RandomForestRegressor(n_estimators=500, max_features=1.0, random_state=42)
    model.fit(features[:split], targets[:split])
    prediction = model.predict(features[split:])
    actual = targets[split:]
    threshold = np.median(targets[:split], axis=0)
    actual_binary = (actual >= threshold).ravel()
    scores = prediction.ravel()
    predicted_binary = (scores >= np.tile(threshold, len(actual))).ravel()

    report = {
        "synthetic": True,
        "warning": "This measures potential on NOAA-fitted surrogate behavior, not real-world accuracy.",
        "split": {"train_rows": split, "test_rows": len(rows) - split, "chronological": True},
        "feature_count": feature_count,
        "regression": {
            "mae": float(mean_absolute_error(actual, prediction)),
            "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
            "r2_uniform_average": float(r2_score(actual, prediction, multioutput="uniform_average")),
        },
        "classification_thresholded": {
            "accuracy": float(accuracy_score(actual_binary, predicted_binary)),
            "precision": float(precision_score(actual_binary, predicted_binary, zero_division=0)),
            "recall": float(recall_score(actual_binary, predicted_binary, zero_division=0)),
            "f1": float(f1_score(actual_binary, predicted_binary, zero_division=0)),
            "roc_auc": float(roc_auc_score(actual_binary, scores)),
        },
    }
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
