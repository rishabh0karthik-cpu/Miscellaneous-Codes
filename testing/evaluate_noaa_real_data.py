"""Evaluate the existing predictor on a real NOAA monthly current dataset.

This is a one-step forecast: lagged and rolling features from observed currents
predict the following month's currents at five fixed grid locations (10 targets).
The source is public NOAA CoastWatch ERDDAP data, not the synthetic fixture.
"""
from __future__ import annotations

import csv
import json
from typing import Any
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
    from .backtesting import (
        anomaly_persistence_forecast,
        climatology_forecast,
        interval_metrics,
        persistence_forecast,
        rolling_origin_splits,
        split_conformal_radius,
    )
    from .current_features import build_lagged_features
    from .quadratic_current_model import QuadraticCurrentPredictor
except ImportError:
    from backtesting import (
        anomaly_persistence_forecast,
        climatology_forecast,
        interval_metrics,
        persistence_forecast,
        rolling_origin_splits,
        split_conformal_radius,
    )
    from current_features import build_lagged_features
    from quadratic_current_model import QuadraticCurrentPredictor

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = Path(__file__).with_name("noaa_real_diagnostic_report.json")
ROC_PATH = Path(__file__).with_name("noaa_real_roc_curve_points.csv")
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


def load_real_targets_with_quality() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    location_rows = [fetch_location(*location) for location in LOCATIONS]
    dates = np.array([f"{row['time'][:7]}-01" for row in location_rows[0]])
    for rows in location_rows[1:]:
        candidate_dates = np.array([f"{row['time'][:7]}-01" for row in rows])
        if not np.array_equal(candidate_dates, dates):
            raise RuntimeError("NOAA locations do not share identical monthly timestamps")
    targets = []
    for row_index in range(len(dates)):
        values = []
        for rows in location_rows:
            values.extend((float(rows[row_index]["u_current"]), float(rows[row_index]["v_current"])))
        targets.append(values)
    targets_array = np.asarray(targets, dtype=float)
    valid = np.isfinite(targets_array).all(axis=1)
    quality = {
        "raw_observations": int(len(dates)),
        "complete_observations": int(valid.sum()),
        "removed_incomplete_observations": int((~valid).sum()),
        "raw_period": [dates[0], dates[-1]] if len(dates) else [],
        "locations": [list(location) for location in LOCATIONS],
    }
    return dates[valid], targets_array[valid], quality


def load_real_targets() -> tuple[np.ndarray, np.ndarray]:
    dates, targets, _ = load_real_targets_with_quality()
    return dates, targets


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_m_per_s": float(mean_absolute_error(actual, predicted)),
        "rmse_m_per_s": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2_uniform_average": float(r2_score(actual, predicted, multioutput="uniform_average")),
    }


def rolling_origin_validation(
    dates: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> list[dict[str, object]]:
    splits = rolling_origin_splits(dates, initial_train_size=60, horizon=12, step=12)
    results = []
    for train_indices, test_indices in splits:
        calibration_size = 12
        fit_indices = train_indices[:-calibration_size]
        calibration_indices = train_indices[-calibration_size:]
        calibration_model = QuadraticCurrentPredictor(alpha=alpha)
        calibration_model.fit(features[fit_indices], targets[fit_indices])
        calibration_prediction = calibration_model.predict(features[calibration_indices])
        model = QuadraticCurrentPredictor(alpha=alpha)
        model.fit(features[train_indices], targets[train_indices])
        predicted = model.predict(features[test_indices])
        interval_radius = split_conformal_radius(
            targets[calibration_indices], calibration_prediction, coverage=0.9
        )
        climatology = climatology_forecast(
            dates[train_indices], targets[train_indices], dates[test_indices]
        )
        anomaly_persistence = anomaly_persistence_forecast(
            dates[train_indices], targets[train_indices], dates[test_indices]
        )
        persistence = persistence_forecast(targets[train_indices], len(test_indices))
        results.append({
            "period": [dates[test_indices[0]], dates[test_indices[-1]]],
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "quadratic_ridge": regression_metrics(targets[test_indices], predicted),
            "climatology": regression_metrics(targets[test_indices], climatology),
            "anomaly_persistence": regression_metrics(targets[test_indices], anomaly_persistence),
            "persistence": regression_metrics(targets[test_indices], persistence),
            "quadratic_ridge_interval_90": interval_metrics(
                targets[test_indices], predicted, interval_radius
            ),
        })
    return results


def summarize_rolling_validation(folds: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    model_names = ("quadratic_ridge", "climatology", "anomaly_persistence", "persistence")
    metric_names = ("mae_m_per_s", "rmse_m_per_s", "r2_uniform_average")
    return {
        model_name: {
            metric_name: float(np.mean([
                float(fold[model_name][metric_name])
                for fold in folds
            ]))
            for metric_name in metric_names
        }
        for model_name in model_names
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


def main() -> None:
    dates, current, quality = load_real_targets_with_quality()
    if len(current) < 20:
        raise RuntimeError("Too few complete NOAA observations for a meaningful split")

    feature_dates, features, targets = build_lagged_features(dates, current)
    train = feature_dates < "2006-01-01"
    validation = (feature_dates >= "2006-01-01") & (feature_dates < "2008-01-01")
    test = feature_dates >= "2008-01-01"

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
    backtest_mask = feature_dates < "2008-01-01"
    rolling_validation = rolling_origin_validation(
        feature_dates[backtest_mask],
        features[backtest_mask],
        targets[backtest_mask],
        selected_alpha,
    )
    quadratic = QuadraticCurrentPredictor(alpha=selected_alpha)
    quadratic.fit(features[train], targets[train])
    quadratic_validation_prediction = quadratic.predict(features[validation])
    quadratic_prediction = quadratic.predict(features[test])
    test_target_indices = np.flatnonzero(test) + (len(current) - len(targets))
    persistence = current[test_target_indices - 1]
    train_mean = np.broadcast_to(targets[train].mean(axis=0), targets[test].shape)

    threshold = np.median(targets[train], axis=0)
    actual_binary = (targets[test] >= threshold).ravel()
    score = quadratic_prediction.ravel()
    predicted_binary = (score >= np.tile(threshold, int(test.sum()))).ravel()
    fpr, tpr, roc_thresholds = roc_curve(actual_binary, score)

    report = {
        "source": {
            "provider": "NOAA NMFS SWFSC ERD",
            "dataset_id": "erdTAgeomday",
            "url": "https://coastwatch.pfeg.noaa.gov/erddap/info/erdTAgeomday/index.csv",
            "observations_after_missing_value_filter": int(len(current)),
            **quality,
            "target_units": "m s-1",
        },
        "task": {
            "type": "one-step multi-output regression",
            "features": "1, 2, 3, 6, and 12-month current lags plus speed, spatial summaries, recent change, seasonal phase, and 3/6/12/24-month rolling statistics (312 values)",
            "targets": "next monthly u/v velocity at five locations (10 values)",
        },
        "split": {
            "train": ["1993-10", "2005-12"],
            "validation": ["2006-01", "2007-12"],
            "test": ["2008-01", "2009-06"],
            "chronological": True,
        },
        "regression": {
            "quadratic_ridge": {
                "selected_alpha": selected_alpha,
                "validation": regression_metrics(targets[validation], quadratic_validation_prediction),
                "test": regression_metrics(targets[test], quadratic_prediction),
            },
            "persistence_baseline": regression_metrics(targets[test], persistence),
            "mean_baseline": regression_metrics(targets[test], train_mean),
        },
        "rolling_origin_validation": {
            "purpose": "Expanding training windows before the untouched 2008-2009 test period",
            "uncertainty_note": "Quadratic-ridge intervals use a 12-month historical calibration tail; coverage is empirical and not guaranteed under serial dependence.",
            "folds": rolling_validation,
            "mean_metrics": summarize_rolling_validation(rolling_validation),
        },
        "classification_note": "Secondary threshold diagnostic only; continuous current prediction is the native task.",
        "classification_thresholded_quadratic_ridge": {
            "threshold": "per-target training median",
            "accuracy": float(accuracy_score(actual_binary, predicted_binary)),
            "precision": float(precision_score(actual_binary, predicted_binary, zero_division=0)),
            "recall": float(recall_score(actual_binary, predicted_binary, zero_division=0)),
            "f1": float(f1_score(actual_binary, predicted_binary, zero_division=0)),
            "roc_auc": float(roc_auc_score(actual_binary, score)),
        },
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
