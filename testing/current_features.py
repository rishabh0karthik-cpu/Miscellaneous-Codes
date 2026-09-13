"""Shared feature construction for chronological current forecasting."""
from __future__ import annotations

from datetime import date

import numpy as np

DEFAULT_LAGS = (1, 2, 3, 6, 12)


def _location_matrix(values: np.ndarray) -> np.ndarray:
    """Reshape a monthly current vector into (locations, 2) pairs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size % 2 != 0:
        raise ValueError("current vectors must be one-dimensional and paired u/v components")
    return values.reshape(-1, 2)


def _rolling_window_features(history: np.ndarray, windows: tuple[int, ...]) -> np.ndarray:
    """Add causal rolling statistics from historical month-by-month current data."""
    if history.ndim != 2 or history.shape[1] == 0:
        return np.asarray([], dtype=float)
    if history.shape[1] % 2 != 0:
        raise ValueError("history rows must contain paired u/v current components")
    max_window = max(windows)
    padded_history = history
    if len(history) < max_window:
        padded_history = np.pad(history, ((max_window - len(history), 0), (0, 0)), mode="constant")
    monthly_vectors = padded_history.reshape(len(padded_history), -1, 2)
    features: list[float] = []
    for window in windows:
        recent = monthly_vectors[-window:]
        u = recent[:, :, 0]
        v = recent[:, :, 1]
        speed = np.linalg.norm(recent, axis=2)
        for location_index in range(u.shape[1]):
            loc_u = u[:, location_index]
            loc_v = v[:, location_index]
            loc_speed = speed[:, location_index]
            features.extend([
                float(np.mean(loc_speed)),
                float(np.std(loc_speed)),
                float(np.mean(loc_u)),
                float(np.std(loc_u)),
                float(np.mean(loc_v)),
                float(np.std(loc_v)),
                float(loc_speed[-1] - loc_speed[0]),
                float(np.arctan2(loc_v[-1], loc_u[-1]) - np.arctan2(loc_v[0], loc_u[0])),
            ])
    return np.asarray(features, dtype=float)


def build_feature_row(
    history: np.ndarray,
    target_date: date,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> np.ndarray:
    """Build one prediction row from observations strictly before target_date."""
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or len(history) < max(lags):
        raise ValueError("history must contain at least the largest requested lag")
    lag_values = [history[-lag].reshape(-1, 2) for lag in lags]
    raw_values = np.concatenate([value.ravel() for value in lag_values])
    derived_values: list[float] = []
    for lag_value in lag_values:
        derived_values.extend(np.linalg.norm(lag_value, axis=1))
        derived_values.extend(lag_value.mean(axis=0))
        derived_values.extend(lag_value.std(axis=0))
    recent_change = lag_values[0].ravel() - lag_values[-1].ravel()
    month_angle = 2 * np.pi * (target_date.month - 1) / 12
    rolling_features = _rolling_window_features(history, (3, 6, 12, 24))
    return np.asarray([
        *raw_values,
        *derived_values,
        *recent_change,
        float(np.sin(month_angle)),
        float(np.cos(month_angle)),
        *rolling_features,
    ])


def build_lagged_features(
    dates: np.ndarray,
    current: np.ndarray,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build lagged current and seasonal features without future information."""
    dates = np.asarray(dates)
    current = np.asarray(current, dtype=float)
    if dates.ndim != 1 or current.ndim != 2 or len(dates) != len(current):
        raise ValueError("dates and current must have matching row counts")
    if not lags or min(lags) < 1 or tuple(sorted(lags)) != lags:
        raise ValueError("lags must be a sorted sequence of positive integers")
    maximum_lag = max(lags)
    feature_rows = []
    target_rows = []
    feature_dates = []
    for target_index in range(maximum_lag, len(current)):
        target_date = date.fromisoformat(str(dates[target_index])[:10])
        feature_rows.append(build_feature_row(current[:target_index], target_date, lags))
        target_rows.append(current[target_index])
        feature_dates.append(str(dates[target_index])[:10])
    return np.asarray(feature_dates), np.asarray(feature_rows), np.asarray(target_rows)
