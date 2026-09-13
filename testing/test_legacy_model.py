import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import TestCase

import numpy as np


LEGACY_PATH = Path(__file__).resolve().parents[1] / "Code-2:Del_Niño"


def load_legacy_module():
    loader = SourceFileLoader("legacy_model_for_tests", str(LEGACY_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_legacy_predictor_delegates_to_canonical_model():
    module = load_legacy_module()
    predictor = module.OceanCurrentPredictor()

    fitted = predictor.fit(np.ones((4, 2)), np.ones((4, 1)))

    assert fitted is predictor
    assert type(predictor.model).__name__ == "QuadraticCurrentPredictor"
    assert predictor.predict(np.ones((1, 2))).shape == (1, 1)


def test_legacy_path_plot_converts_velocity_units():
    module = load_legacy_module()
    figure, axes = module.plot_current_paths(
        np.array([[0.0, 0.0]]),
        np.array([[[1.0, 0.0]]]),
    )

    longitude_delta = float(axes.lines[0].get_xdata()[-1])
    assert 23.0 < longitude_delta < 23.6
    figure.clf()


def test_legacy_dynamics_rejects_wrong_prediction_dimension():
    module = load_legacy_module()
    dynamics = module.LinearCurrentDynamics().fit(np.ones((3, 2)))

    with TestCase().assertRaisesRegex(ValueError, "feature dimension"):
        dynamics.predict_next(np.ones(3))
