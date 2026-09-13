"""Regularized quadratic forecaster for lagged ocean-current variables."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class QuadraticCurrentPredictor:
    """Predict continuous u/v current targets with squared and interaction terms.

    For 30 lagged current inputs, degree-two expansion produces 495 polynomial
    features. Ridge regularization is required because the real-data sample is
    much smaller than that expanded feature space.
    """

    def __init__(self, alpha: float = 3000.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
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
        if feature_data.shape[0] != current_patterns.shape[0]:
            raise ValueError("features and current_patterns must have equal row counts")
        self.model.fit(feature_data, current_patterns)
        return self

    def predict(self, feature_data: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(feature_data, dtype=float))
