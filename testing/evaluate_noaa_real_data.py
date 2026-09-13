"""Evaluate the existing predictor on a real NOAA monthly current dataset.

This is a one-step forecast: three months of observed currents (30 features)
predict the following month's currents at five fixed grid locations (10 targets).
The source is public NOAA CoastWatch ERDDAP data, not the synthetic fixture.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from importlib.machinery import SourceFileLoader
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

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
try:
    from .current_features import build_lagged_features
    from .quadratic_current_model import QuadraticCurrentPredictor
except ImportError:
    from current_features import build_lagged_features
    from quadratic_current_model import QuadraticCurrentPredictor

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = Path(__file__).with_name("noaa_real_diagnostic_report.json")
ROC_PATH = Path(__file__).with_name("noaa_real_roc_curve_points.csv")
MODEL_PATH = ROOT / "Code-2:Del_Niño"
BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdTAgeomday.csv?"
# Five ocean grid points, represented by source array indexes.
LOCATIONS = [(300, 720), (300, 900), (300, 1080), (320, 720), (280, 900)]


def fetch_location(latitude_index: int, longitude_index: int) -> list[dict[str, str]]:
    selection = (
        f"u_current[0:1:200][0][{latitude_index}][{longitude_index}],"
        f"v_current[0:1:200][0][{latitude_index}][{longitude_index}]"
    )
    with urlopen(BASE_URL + quote(selection, safe="[],:"), timeout=30) as response:
        rows = list(csv.DictReader(response.read().decode("utf-8").splitlines()))
    return [row for row in rows if row["time"] != "UTC"]


def load_real_targets() -> tuple[np.ndarray, np.ndarray]:
    location_rows = [fetch_location(*location) for location in LOCATIONS]
    dates = np.array([row["time"][:10] for row in location_rows[0]])
    targets = []
    for row_index in range(len(dates)):
        values = []
        for rows in location_rows:
            values.extend((float(rows[row_index]["u_current"]), float(rows[row_index]["v_current"])))
        targets.append(values)
    targets_array = np.asarray(targets, dtype=float)
    valid = np.isfinite(targets_array).all(axis=1)
    return dates[valid], targets_array[valid]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_m_per_s": float(mean_absolute_error(actual, predicted)),
        "rmse_m_per_s": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2_uniform_average": float(r2_score(actual, predicted, multioutput="uniform_average")),
    }


def thresholded_metrics(actual: np.ndarray, predicted: np.ndarray, threshold: np.ndarray) -> dict[str, float]:
    actual_binary = (actual >= threshold).ravel()
    scores = predicted.ravel()
    predicted_binary = (scores >= np.tile(threshold, len(actual))).ravel()
    return {
        "accuracy": float(accuracy_score(actual_binary, predicted_binary)),
        "precision": float(precision_score(actual_binary, predicted_binary, zero_division=0)),
        "recall": float(recall_score(actual_binary, predicted_binary, zero_division=0)),
        "f1": float(f1_score(actual_binary, predicted_binary, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual_binary, scores)),
    }


def load_predictor():
    with tempfile.TemporaryDirectory() as workdir:
        previous_directory = os.getcwd()
        os.chdir(workdir)
        try:
            return SourceFileLoader("ocean_model", str(MODEL_PATH)).load_module().OceanCurrentPredictor
        finally:
            os.chdir(previous_directory)


def main() -> None:
    dates, current = load_real_targets()
    if len(current) < 20:
        raise RuntimeError("Too few complete NOAA observations for a meaningful split")

    # Multiple lags and seasonal phase predict the next month (10 values).
    feature_dates, features, targets = build_lagged_features(dates, current)
    train = feature_dates < "2006-01-01"
    validation = (feature_dates >= "2006-01-01") & (feature_dates < "2008-01-01")
    test = feature_dates >= "2008-01-01"

    predictor = load_predictor()()
    predictor.fit(features[train], targets[train])
    validation_prediction = predictor.predict(features[validation])
    prediction = predictor.predict(features[test])
    candidate_alphas = [100.0, 300.0, 1000.0, 3000.0, 10000.0]
    alpha_scores = []
    for alpha in candidate_alphas:
        candidate = QuadraticCurrentPredictor(alpha=alpha)
        candidate.fit(features[train], targets[train])
        validation_score = r2_score(
            targets[validation],
            candidate.predict(features[validation]),
            multioutput="uniform_average",
        )
        alpha_scores.append((validation_score, alpha))
    selected_alpha = max(alpha_scores)[1]
    quadratic = QuadraticCurrentPredictor(alpha=selected_alpha)
    quadratic.fit(features[train], targets[train])
    quadratic_validation_prediction = quadratic.predict(features[validation])
    quadratic_prediction = quadratic.predict(features[test])
    test_target_indices = np.flatnonzero(test) + max((1, 2, 3, 6, 12))
    persistence = current[test_target_indices - 1]
    train_mean = np.broadcast_to(targets[train].mean(axis=0), targets[test].shape)

    threshold = np.median(targets[train], axis=0)
    actual_binary = (targets[test] >= threshold).ravel()
    score = prediction.ravel()
    predicted_binary = (score >= np.tile(threshold, int(test.sum()))).ravel()
    fpr, tpr, roc_thresholds = roc_curve(actual_binary, score)

    report = {
        "source": {
            "provider": "NOAA NMFS SWFSC ERD",
            "dataset_id": "erdTAgeomday",
            "url": "https://coastwatch.pfeg.noaa.gov/erddap/info/erdTAgeomday/index.csv",
            "observations_after_missing_value_filter": int(len(current)),
            "locations": LOCATIONS,
            "target_units": "m s-1",
        },
        "task": {
            "type": "one-step multi-output regression",
            "features": "1, 2, 3, 6, and 12-month current lags plus speed, spatial summaries, recent change, and seasonal phase (89 values)",
            "targets": "next monthly u/v velocity at five locations (10 values)",
        },
        "split": {
            "train": ["1993-10", "2005-12"],
            "validation": ["2006-01", "2007-12"],
            "test": ["2008-01", "2009-06"],
            "chronological": True,
        },
        "regression": {
            "random_forest": regression_metrics(targets[test], prediction),
            "quadratic_ridge": {
                "selected_alpha": selected_alpha,
                "validation": regression_metrics(targets[validation], quadratic_validation_prediction),
                "test": regression_metrics(targets[test], quadratic_prediction),
            },
            "persistence_baseline": regression_metrics(targets[test], persistence),
            "mean_baseline": regression_metrics(targets[test], train_mean),
        },
        "classification_note": "Secondary threshold diagnostic only; continuous current prediction is the native task.",
        "classification_thresholded": {
            "threshold": "per-target training median",
            "accuracy": float(accuracy_score(actual_binary, predicted_binary)),
            "precision": float(precision_score(actual_binary, predicted_binary, zero_division=0)),
            "recall": float(recall_score(actual_binary, predicted_binary, zero_division=0)),
            "f1": float(f1_score(actual_binary, predicted_binary, zero_division=0)),
            "roc_auc": float(roc_auc_score(actual_binary, score)),
        },
        "classification_thresholded_quadratic_ridge": thresholded_metrics(
            targets[test], quadratic_prediction, threshold
        ),
    }
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with ROC_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["false_positive_rate", "true_positive_rate", "threshold"])
        writer.writerows(zip(fpr, tpr, roc_thresholds))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
