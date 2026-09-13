from datetime import date
from unittest import TestCase

import numpy as np

from current_features import build_feature_row, build_lagged_features


def test_feature_row_has_richer_causal_signal():
    rng = np.random.default_rng(42)
    history = rng.normal(size=(48, 10))
    row = build_feature_row(history, date(2006, 1, 1))
    assert row.shape[0] >= 150, f"expected richer feature set, got {row.shape[0]}"


def test_feature_rows_are_leakage_safe_and_compatible():
    dates = np.asarray([
        f"{year:04d}-{month:02d}-01"
        for year in range(2000, 2004)
        for month in range(1, 13)
    ])
    current = np.random.default_rng(7).normal(size=(48, 10))
    feature_dates, features, targets = build_lagged_features(dates, current)
    assert feature_dates.shape[0] == features.shape[0] == targets.shape[0]
    assert feature_dates[0] == "2002-01-01"
    assert features.shape[1] >= 150
    assert targets.shape[1] == 10


def test_feature_builder_rejects_calendar_gaps():
    dates = np.asarray([
        "2000-01-01", "2000-02-01", "2000-04-01",
    ] + [
        *(f"2000-{month:02d}-01" for month in range(5, 13)),
        *(f"2001-{month:02d}-01" for month in range(1, 13)),
        "2002-01-01",
    ])
    current = np.zeros((len(dates), 10))

    with TestCase().assertRaisesRegex(ValueError, "consecutive monthly"):
        build_lagged_features(dates, current)


def test_feature_builder_does_not_use_target_date_information():
    dates = np.asarray([
        f"{year:04d}-{month:02d}-01"
        for year in range(2000, 2004)
        for month in range(1, 13)
    ])
    current = np.random.default_rng(8).normal(size=(48, 10))
    _, first_features, _ = build_lagged_features(dates, current)
    changed = current.copy()
    changed[25] = 1_000_000
    _, changed_features, _ = build_lagged_features(dates, changed)

    assert np.array_equal(first_features[0], changed_features[0])


def test_rolling_direction_change_is_wrapped():
    history = np.zeros((24, 10))
    history[:, 0] = -1.0
    history[-1, 0] = 1.0

    row = build_feature_row(history, date(2002, 1, 1))

    assert np.max(np.abs(row)) < 10.0


def test_feature_builder_rejects_non_finite_history():
    history = np.ones((24, 10))
    history[0, 0] = np.nan

    with TestCase().assertRaisesRegex(ValueError, "finite"):
        build_feature_row(history, date(2002, 1, 1))


def test_feature_builder_rejects_unsorted_lags():
    history = np.ones((24, 10))

    with TestCase().assertRaisesRegex(ValueError, "sorted"):
        build_feature_row(history, date(2002, 1, 1), lags=(2, 1))


def test_feature_builder_rejects_non_finite_targets():
    dates = np.asarray([
        f"{year:04d}-{month:02d}-01"
        for year in range(2000, 2004)
        for month in range(1, 13)
    ])
    current = np.ones((48, 10))
    current[-1, 0] = np.inf

    with TestCase().assertRaisesRegex(ValueError, "finite"):
        build_lagged_features(dates, current)
