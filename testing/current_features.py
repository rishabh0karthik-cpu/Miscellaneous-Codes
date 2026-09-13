"""Shared feature construction for chronological current forecasting."""
from __future__ import annotations

from datetime import date

import numpy as np

DEFAULT_LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12, 24)


def _validate_lags(lags: tuple[int, ...]) -> None:
    if (
        not lags
        or any(not isinstance(lag, (int, np.integer)) for lag in lags)
        or min(lags) < 1
        or tuple(sorted(lags)) != lags
        or len(set(lags)) != len(lags)
    ):
        raise ValueError("lags must be a sorted sequence of unique positive integers")


def _validate_monthly_dates(dates: np.ndarray) -> list[date]:
    parsed = [date.fromisoformat(str(value)[:10]) for value in dates]
    if any(value.day != 1 for value in parsed):
        raise ValueError("dates must be month-start timestamps")
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError("dates must be strictly increasing")
    for left, right in zip(parsed, parsed[1:]):
        expected_year = left.year + (left.month == 12)
        expected_month = 1 if left.month == 12 else left.month + 1
        if (right.year, right.month) != (expected_year, expected_month):
            raise ValueError("dates must contain consecutive monthly observations")
    return parsed


def _location_matrix(values: np.ndarray) -> np.ndarray:
    """Reshape a monthly current vector into (locations, 2) pairs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size % 2 != 0:
        raise ValueError("current vectors must be one-dimensional and paired u/v components")
    if not np.isfinite(values).all():
        raise ValueError("current vectors must contain only finite values")
    return values.reshape(-1, 2)


def _rolling_window_features(history: np.ndarray, windows: tuple[int, ...]) -> np.ndarray:
    """Add causal rolling statistics from historical month-by-month current data."""
    if history.ndim != 2 or history.shape[1] == 0:
        return np.asarray([], dtype=float)
    if history.shape[1] % 2 != 0:
        raise ValueError("history rows must contain paired u/v current components")
    if len(history) < max(windows):
        raise ValueError("history must contain enough observations for rolling windows")
    monthly_vectors = history.reshape(len(history), -1, 2)
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
                float(np.angle(np.exp(1j * (np.arctan2(loc_v[-1], loc_u[-1]) - np.arctan2(loc_v[0], loc_u[0]))))),
            ])
    return np.asarray(features, dtype=float)


def build_feature_row(
    history: np.ndarray,
    target_date: date,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> np.ndarray:
    """Build one prediction row from observations strictly before target_date."""
    history = np.asarray(history, dtype=float)
    _validate_lags(lags)
    if history.ndim != 2 or len(history) < max(max(lags), max(ROLLING_WINDOWS)):
        raise ValueError("history must contain the requested lags and rolling windows")
    if not np.isfinite(history).all():
        raise ValueError("history must contain only finite values")
    lag_values = [history[-lag].reshape(-1, 2) for lag in lags]
    raw_values = np.concatenate([value.ravel() for value in lag_values])
    derived_values: list[float] = []
    for lag_value in lag_values:
        derived_values.extend(np.linalg.norm(lag_value, axis=1))
        derived_values.extend(lag_value.mean(axis=0))
        derived_values.extend(lag_value.std(axis=0))
    recent_change = lag_values[0].ravel() - lag_values[-1].ravel()
    month_angle = 2 * np.pi * (target_date.month - 1) / 12
    rolling_features = _rolling_window_features(history, ROLLING_WINDOWS)
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
    if not np.isfinite(current).all():
        raise ValueError("current must contain only finite values")
    _validate_lags(lags)
    parsed_dates = _validate_monthly_dates(dates)
    maximum_lag = max(max(lags), max(ROLLING_WINDOWS))
    feature_rows = []
    target_rows = []
    feature_dates = []
    for target_index in range(maximum_lag, len(current)):
        target_date = parsed_dates[target_index]
        feature_rows.append(build_feature_row(current[:target_index], target_date, lags))
        target_rows.append(current[target_index])
        feature_dates.append(target_date.isoformat())
    return np.asarray(feature_dates), np.asarray(feature_rows), np.asarray(target_rows)
