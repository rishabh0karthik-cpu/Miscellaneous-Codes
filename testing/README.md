# Current-pattern diagnostic fixture

`dummy_current_patterns_2006_2025.csv` is a deterministic, deliberately learnable synthetic test document with 240 monthly rows, 30 feature columns, and 10 current-pattern targets (eastward and northward components for five locations). It is designed as a 90% pipeline quality gate, not as a claim about real-data performance.

The schema is oceanography-inspired, but the values are simulated. It is not an import from NOAA, Copernicus, NASA, or another scientific archive and must not be used as evidence about real currents. It exists to validate loading, chronological splitting, metrics, baseline comparisons, threshold handling, and report generation.

`scientific_source_manifest.json` records a successful NOAA CoastWatch ERDDAP metadata lookup for a public current dataset (`erdTAgeomday`) and explains why its full grid is not bundled into this deterministic fixture.

For a real-data check, run:

```bash
python testing/evaluate_noaa_real_data.py
```

This downloads 1992-2009 NOAA monthly observations at five fixed grid points. It uses 1, 2, 3, 6, and 12-month lags plus causal rolling statistics through 24 months to predict the next month (10 values), with a chronological 2008-2009 test period. The evaluator uses the canonical quadratic ridge model and reports its result explicitly. This real-data check currently fails the 90% target, which is an honest indication that the present feature set is not pitch-ready for scientific claims.

The report also includes expanding rolling-origin validation folds before 2008, comparing quadratic ridge with monthly climatology, anomaly persistence, and raw persistence. These folds are diagnostic evidence, not a replacement for independent spatial or in-situ validation.

Each rolling fold also reports empirical 90% quadratic-ridge interval coverage and mean width. Intervals use a held-out 12-month calibration tail; serial dependence means this is a diagnostic, not a formal coverage guarantee. The source section records raw rows, complete rows, removed rows, period, and queried locations for provenance.

The repository does not contain SST, sea-level anomaly, wind-stress, ENSO, bathymetry, or independent in-situ observations. Those inputs must be supplied and versioned by the research partner before claims about seasonal or operational skill can be made.

To create a NOAA-fitted surrogate expansion for potential and robustness testing, run:

```bash
python testing/generate_noaa_surrogate_data.py
python testing/evaluate_noaa_surrogate.py
```

This fits a multivariate autoregression with 1/2/3/6/12-month lags, speed/spatial summaries, seasonal phase, and rolling statistics to the NOAA training period, then samples its learned residual covariance. The output is plausible synthetic behavior, not additional observations. The current chronological check reaches 61.25% accuracy, 59.88% precision, 61.91% recall, and 60.88% F1 on later surrogate rows, but only 52.64% ROC-AUC; this demonstrates partial potential, not scientific validation.

Run from the repository root:

```bash
python testing/generate_dummy_current_data.py
python testing/evaluate_model.py
```

The evaluator uses the canonical quadratic ridge model, trains only on 2006-2021, uses 2022-2023 for validation, and reports final metrics on 2024-2025. It writes `testing/model_diagnostic_report.json` and `testing/roc_curve_points.csv`, including a synthetic-only pass/fail check requiring accuracy, precision, recall, F1, and ROC-AUC to each be at least 0.90. The canonical model currently fails this gate, which is expected and prevents the old legacy-forest result from being mistaken for evidence.
