from datetime import date

import numpy as np

from current_features import build_feature_row, build_lagged_features


def test_feature_row_has_richer_causal_signal():
    rng = np.random.default_rng(42)
    history = rng.normal(size=(48, 10))
    row = build_feature_row(history, date(2006, 1, 1))
    assert row.shape[0] >= 150, f"expected richer feature set, got {row.shape[0]}"


def test_feature_rows_are_leakage_safe_and_compatible():
    dates = np.asarray([f"2000-{month:02d}-01" for month in range(1, 13) for _ in range(2)])[:24]
    current = np.random.default_rng(7).normal(size=(24, 10))
    feature_dates, features, targets = build_lagged_features(dates, current)
    assert feature_dates.shape[0] == features.shape[0] == targets.shape[0]
    assert features.shape[1] >= 150
    assert targets.shape[1] == 10
