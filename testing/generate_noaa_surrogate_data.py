"""Generate plausible NOAA-like current patterns from the NOAA training period.

This is a surrogate-data generator for potential testing. It is not a source
of new observations and must not be described as measured ocean data.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from current_features import ROLLING_WINDOWS, build_feature_row, build_lagged_features
from evaluate_noaa_real_data import load_real_targets

OUTPUT = Path(__file__).with_name("noaa_surrogate_patterns.csv")
METADATA = Path(__file__).with_name("noaa_surrogate_metadata.json")
RANDOM_SEED = 20260911


def fit_surrogate(dates: np.ndarray, current: np.ndarray, alpha: float = 10.0) -> tuple[Ridge, np.ndarray]:
    _, features, targets = build_lagged_features(dates, current)
    model = Ridge(alpha=alpha).fit(features, targets)
    residuals = targets - model.predict(features)
    covariance = np.atleast_2d(np.cov(residuals, rowvar=False))
    return model, covariance


def next_month(value: date) -> date:
    return date(value.year + (value.month == 12), 1 if value.month == 12 else value.month + 1, 1)


def simulate(
    model: Ridge,
    covariance: np.ndarray,
    initial_dates: np.ndarray,
    initial: np.ndarray,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dates = [date.fromisoformat(str(value)[:10]) for value in initial_dates]
    current = [row.copy() for row in initial]
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-8
    for _ in range(steps):
        target_date = next_month(dates[-1])
        features = build_feature_row(np.asarray(current), target_date)
        expected = model.predict(features.reshape(1, -1))[0]
        noise = rng.multivariate_normal(np.zeros(expected.shape[0]), covariance)
        dates.append(target_date)
        current.append(expected + noise)
    return np.asarray(dates), np.asarray(current)


def main() -> None:
    dates, observed = load_real_targets()
    training = dates < "2006-01-01"
    training_current = observed[training]
    training_dates = dates[training]
    model, covariance = fit_surrogate(training_dates, training_current)
    rng = np.random.default_rng(RANDOM_SEED)
    initial_index = int(rng.integers(0, len(training_current) - max(ROLLING_WINDOWS)))
    initial_end = initial_index + max(ROLLING_WINDOWS)
    simulated_dates, simulated = simulate(
        model,
        covariance,
        training_dates[initial_index:initial_end],
        training_current[initial_index:initial_end],
        2000,
        rng,
    )

    _, feature_rows, target_rows = build_lagged_features(simulated_dates, simulated)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample",
            *[f"feature_{index}" for index in range(feature_rows.shape[1])],
            *[f"target_current_{component}" for component in range(10)],
        ])
        for row_index, (features, target) in enumerate(zip(feature_rows, target_rows)):
            writer.writerow([row_index, *features, *target])

    metadata = {
        "synthetic": True,
        "purpose": "Potential and robustness testing only; not measured observations",
        "source": "NOAA CoastWatch ERDDAP erdTAgeomday",
        "source_training_period": ["1992-10", "2005-12"],
        "source_locations": "The five locations configured in evaluate_noaa_real_data.py",
        "generator": "multivariate Ridge autoregression with 1/2/3/6/12-month lags, kinematic/spatial summaries, seasonal phase, and sampled residual covariance",
        "rows": int(len(target_rows)),
        "random_seed": RANDOM_SEED,
        "warning": "Synthetic rows inherit assumptions from the training period and cannot establish real-world accuracy.",
    }
    with METADATA.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote {len(target_rows)} surrogate rows to {OUTPUT}")


if __name__ == "__main__":
    main()
