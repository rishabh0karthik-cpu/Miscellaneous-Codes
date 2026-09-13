"""Generate a deterministic, synthetic ocean-current test document.

This is a functional test fixture, not an observation database. The column
names mirror common oceanographic variables while the values are simulated.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np

OUTPUT = Path(__file__).with_name("dummy_current_patterns_2006_2025.csv")


def main() -> None:
    rng = np.random.default_rng(20260911)
    dates = [date(year, month, 1) for year in range(2006, 2026) for month in range(1, 13)]
    feature_names = [f"feature_{index:02d}" for index in range(30)]
    target_names = [f"current_{location}_{component}" for location in range(1, 6) for component in ("u", "v")]

    rows = []
    target_coefficients = np.array([
        [0.30 + 0.03 * index, 0.18 - 0.01 * index, 0.12 + 0.01 * index, 0.08]
        for index in range(10)
    ])
    for row_index, observation_date in enumerate(dates):
        month_angle = 2 * np.pi * observation_date.month / 12
        year_fraction = row_index / 12
        seasonal = np.sin(month_angle)
        slow_cycle = np.sin(2 * np.pi * year_fraction / 7)
        features = np.array([
            seasonal,
            np.cos(month_angle),
            slow_cycle,
            np.cos(2 * np.pi * year_fraction / 7),
            row_index / len(dates),
            *rng.normal(0, 0.05, 25),
        ])
        pattern = np.array([
            coefficients @ features[:4] + rng.normal(0, 0.005)
            for coefficients in target_coefficients
        ])
        rows.append([observation_date.isoformat(), *features, *pattern])

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", *feature_names, *target_names])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
