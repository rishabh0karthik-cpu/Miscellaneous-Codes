import numpy as np
from unittest import TestCase

from quadratic_current_model import QuadraticCurrentPredictor


def test_quadratic_model_rejects_non_finite_training_data():
    features = np.ones((4, 2))
    targets = np.ones((4, 1))
    features[0, 0] = np.inf

    with TestCase().assertRaisesRegex(ValueError, "finite"):
        QuadraticCurrentPredictor().fit(features, targets)


def test_quadratic_model_rejects_non_finite_prediction_data():
    model = QuadraticCurrentPredictor().fit(np.ones((4, 2)), np.ones((4, 1)))

    with TestCase().assertRaisesRegex(ValueError, "finite"):
        model.predict(np.array([[np.nan, 1.0]]))


def test_quadratic_model_rejects_empty_training_data():
    with TestCase().assertRaisesRegex(ValueError, "empty"):
        QuadraticCurrentPredictor().fit(np.empty((0, 2)), np.empty((0, 1)))


def test_quadratic_model_rejects_prediction_dimension_changes():
    model = QuadraticCurrentPredictor().fit(np.ones((4, 2)), np.ones((4, 1)))

    with TestCase().assertRaisesRegex(ValueError, "column count"):
        model.predict(np.ones((1, 3)))