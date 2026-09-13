"""Regularized quadratic forecaster for lagged ocean-current variables."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class QuadraticCurrentPredictor:
    """Predict continuous u/v current targets with squared and interaction terms.

    The current feature builder produces 312 inputs before the quadratic
    expansion. Ridge regularization is required because the real-data sample
    is much smaller than that expanded feature space.
    """

    def __init__(self, alpha: float = 3000.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.feature_count = None
        self.target_count = None
        self.model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=alpha),
        )

    def fit(self, feature_data: np.ndarray, current_patterns: np.ndarray):
        feature_data = np.asarray(feature_data, dtype=float)
        current_patterns = np.asarray(current_patterns, dtype=float)
        if feature_data.ndim != 2 or current_patterns.ndim != 2:
            raise ValueError("features and current_patterns must be two-dimensional")
        if feature_data.shape[0] == 0 or current_patterns.shape[1] == 0:
            raise ValueError("features and current_patterns must not be empty")
        if feature_data.shape[0] != current_patterns.shape[0]:
            raise ValueError("features and current_patterns must have equal row counts")
        if not np.isfinite(feature_data).all() or not np.isfinite(current_patterns).all():
            raise ValueError("features and current_patterns must contain only finite values")
        self.model.fit(feature_data, current_patterns)
        self.feature_count = feature_data.shape[1]
        self.target_count = current_patterns.shape[1]
        return self

    def predict(self, feature_data: np.ndarray) -> np.ndarray:
        feature_data = np.asarray(feature_data, dtype=float)
        if feature_data.ndim != 2 or not np.isfinite(feature_data).all():
            raise ValueError("features must be a finite two-dimensional array")
        if self.feature_count is None:
            raise RuntimeError("fit must be called before predict")
        if feature_data.shape[1] != self.feature_count:
            raise ValueError("prediction features have a different column count than training features")
        predictions = np.asarray(self.model.predict(feature_data), dtype=float)
        return predictions.reshape(feature_data.shape[0], self.target_count)
