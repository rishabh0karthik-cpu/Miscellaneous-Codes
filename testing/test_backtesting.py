import numpy as np
from unittest import TestCase

from backtesting import (
    anomaly_persistence_forecast,
    climatology_forecast,
    interval_metrics,
    persistence_forecast,
    rolling_origin_splits,
    split_conformal_radius,
)


def monthly_dates(start_year: int, count: int) -> np.ndarray:
    return np.asarray([
        f"{start_year + index // 12:04d}-{index % 12 + 1:02d}-01"
        for index in range(count)
    ])


def test_rolling_origin_splits_are_forward_only_and_expanding():
    dates = monthly_dates(2000, 12)

    splits = rolling_origin_splits(dates, initial_train_size=6, horizon=2, step=2)

    assert len(splits) == 3
    for train_indices, test_indices in splits:
        assert train_indices[-1] < test_indices[0]
        assert len(test_indices) == 2
    assert len(splits[1][0]) > len(splits[0][0])


def test_persistence_uses_only_last_training_observation():
    targets = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    forecast = persistence_forecast(targets, horizon=2)

    np.testing.assert_array_equal(forecast, [[5.0, 6.0], [5.0, 6.0]])


def test_climatology_uses_training_month_means():
    dates = monthly_dates(2000, 24)
    targets = np.asarray([[float(date[5:7])] for date in dates])
    forecast_dates = np.asarray(["2002-01-01", "2002-02-01"])

    forecast = climatology_forecast(dates, targets, forecast_dates)

    np.testing.assert_array_equal(forecast.ravel(), [1.0, 2.0])


def test_anomaly_persistence_preserves_final_training_anomaly():
    dates = monthly_dates(2000, 24)
    targets = np.asarray([[float(date[5:7]) + (2.0 if index == 23 else 0.0)] for index, date in enumerate(dates)])
    forecast_dates = np.asarray(["2002-01-01", "2002-02-01"])

    forecast = anomaly_persistence_forecast(dates, targets, forecast_dates)

    np.testing.assert_array_equal(forecast.ravel(), [2.0, 3.0])


def test_rolling_origin_rejects_duplicate_dates():
    dates = np.asarray(["2000-01-01", "2000-01-01", "2000-02-01"])

    with TestCase().assertRaisesRegex(ValueError, "strictly increasing"):
        rolling_origin_splits(dates, initial_train_size=1, horizon=1)


def test_split_conformal_interval_reports_empirical_coverage():
    calibration_actual = np.asarray([[0.0], [1.0], [2.0]])
    calibration_predicted = np.asarray([[0.0], [0.5], [1.0]])
    actual = np.asarray([[0.0], [2.0]])
    predicted = np.asarray([[0.5], [1.5]])

    radius = split_conformal_radius(calibration_actual, calibration_predicted, coverage=0.9)
    metrics = interval_metrics(actual, predicted, radius)

    np.testing.assert_array_equal(radius, [1.0])
    assert metrics == {"coverage": 1.0, "mean_width": 2.0}