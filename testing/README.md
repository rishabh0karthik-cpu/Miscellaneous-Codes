# Current-pattern diagnostic fixture

`dummy_current_patterns_2006_2025.csv` is a deterministic, deliberately learnable synthetic test document with 240 monthly rows, 30 feature columns, and 10 current-pattern targets (eastward and northward components for five locations). It is designed as a 90% pipeline quality gate, not as a claim about real-data performance.

The schema is oceanography-inspired, but the values are simulated. It is not an import from NOAA, Copernicus, NASA, or another scientific archive and must not be used as evidence about real currents. It exists to validate loading, chronological splitting, metrics, baseline comparisons, threshold handling, and report generation.

`scientific_source_manifest.json` records a successful NOAA CoastWatch ERDDAP metadata lookup for a public current dataset (`erdTAgeomday`) and explains why its full grid is not bundled into this deterministic fixture.

For a real-data check, run:

```bash
python testing/evaluate_noaa_real_data.py
```

This downloads 1992-2009 NOAA monthly observations at five fixed grid points. It uses the previous three months (30 velocity values) to predict the next month (10 values), with a chronological 2008-2009 test period. The resulting report is `testing/noaa_real_diagnostic_report.json`. This real-data check currently fails the 90% target, which is an honest indication that the present feature set and random forest are not pitch-ready for scientific claims.

To create a NOAA-fitted surrogate expansion for potential and robustness testing, run:

```bash
python testing/generate_noaa_surrogate_data.py
python testing/evaluate_noaa_surrogate.py
```

This fits a multivariate autoregression with 1/2/3/6/12-month lags, speed/spatial summaries, and seasonal phase to the NOAA training period, then samples its learned residual covariance. The output is plausible synthetic behavior, not additional observations. The current chronological forest check reaches 61.25% accuracy, 59.88% precision, 61.91% recall, and 60.88% F1 on later surrogate rows, but only 52.64% ROC-AUC; this demonstrates partial potential, not scientific validation.

Run from the repository root:

```bash
python testing/generate_dummy_current_data.py
python testing/evaluate_model.py
```

The evaluator trains only on 2006-2021, uses 2022-2023 for validation, and reports final metrics on 2024-2025. It writes `testing/model_diagnostic_report.json` and `testing/roc_curve_points.csv`, including a pass/fail check requiring accuracy, precision, recall, F1, and ROC-AUC to each be at least 0.90.
