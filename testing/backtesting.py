"""Leakage-safe utilities for chronological current-forecast evaluation."""
from __future__ import annotations

from datetime import date

import numpy as np


def _parse_dates(dates: np.ndarray) -> list[date]:
    parsed = [date.fromisoformat(str(value)[:10]) for value in np.asarray(dates)]
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError("dates must be strictly increasing")
    return parsed


def rolling_origin_splits(
    dates: np.ndarray,
    initial_train_size: int,
    horizon: int,
    step: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding-training, forward-only train/test index pairs."""
    parsed_dates = _parse_dates(dates)
    if initial_train_size < 1 or horizon < 1:
        raise ValueError("initial_train_size and horizon must be positive")
    step = horizon if step is None else step
    if step < 1:
        raise ValueError("step must be positive")
    splits = []
    test_start = initial_train_size
    while test_start + horizon <= len(parsed_dates):
        splits.append((
            np.arange(test_start),
            np.arange(test_start, test_start + horizon),
        ))
        test_start += step
    if not splits:
        raise ValueError("dates do not contain a complete rolling-origin split")
    return splits


def persistence_forecast(train_targets: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed target for every forecast step."""
    train_targets = np.asarray(train_targets, dtype=float)
    if train_targets.ndim != 2 or len(train_targets) == 0 or horizon < 1:
        raise ValueError("train_targets must be non-empty two-dimensional data")
    return np.repeat(train_targets[-1][None, :], horizon, axis=0)


def climatology_forecast(
    train_dates: np.ndarray,
    train_targets: np.ndarray,
    forecast_dates: np.ndarray,
) -> np.ndarray:
    """Forecast each month using means computed only from the training period."""
    parsed_train_dates = _parse_dates(train_dates)
    parsed_forecast_dates = _parse_dates(forecast_dates)
    train_targets = np.asarray(train_targets, dtype=float)
    if train_targets.ndim != 2 or len(parsed_train_dates) != len(train_targets):
        raise ValueError("training dates and targets must have matching two-dimensional data")
    month_means = {}
    for month in range(1, 13):
        month_values = train_targets[[value.month == month for value in parsed_train_dates]]
        if len(month_values) == 0:
            raise ValueError(f"training data does not contain month {month}")
        month_means[month] = month_values.mean(axis=0)
    return np.asarray([month_means[value.month] for value in parsed_forecast_dates])


def anomaly_persistence_forecast(
    train_dates: np.ndarray,
    train_targets: np.ndarray,
    forecast_dates: np.ndarray,
) -> np.ndarray:
    """Persist the final training anomaly around the monthly climatology."""
    train_dates = np.asarray(train_dates)
    train_targets = np.asarray(train_targets, dtype=float)
    forecast_dates = np.asarray(forecast_dates)
    if len(train_dates) == 0:
        raise ValueError("training data must not be empty")
    climatology = climatology_forecast(train_dates, train_targets, train_dates)
    final_anomaly = train_targets[-1] - climatology[-1]
    forecast_climatology = climatology_forecast(train_dates, train_targets, forecast_dates)
    return forecast_climatology + final_anomaly


def split_conformal_radius(
    calibration_actual: np.ndarray,
    calibration_predicted: np.ndarray,
    coverage: float = 0.9,
) -> np.ndarray:
    """Estimate per-target absolute-error radii from a held-out calibration tail."""
    calibration_actual = np.asarray(calibration_actual, dtype=float)
    calibration_predicted = np.asarray(calibration_predicted, dtype=float)
    if calibration_actual.shape != calibration_predicted.shape or calibration_actual.ndim != 2:
        raise ValueError("calibration arrays must have matching two-dimensional shapes")
    if len(calibration_actual) == 0 or not 0 < coverage < 1:
        raise ValueError("calibration data must be non-empty and coverage must be between zero and one")
    errors = np.abs(calibration_actual - calibration_predicted)
    quantile = min(1.0, np.ceil((len(errors) + 1) * coverage) / len(errors))
    return np.quantile(errors, quantile, axis=0, method="higher")


def interval_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    radius: np.ndarray,
) -> dict[str, float]:
    """Return empirical marginal coverage and mean interval width."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    radius = np.asarray(radius, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 2 or radius.shape != (actual.shape[1],):
        raise ValueError("interval inputs have incompatible shapes")
    covered = np.abs(actual - predicted) <= radius
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(2 * radius.mean()),
    }