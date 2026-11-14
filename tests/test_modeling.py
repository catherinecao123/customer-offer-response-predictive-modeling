"""Tests for offer_response.modeling."""

import numpy as np
import pandas as pd
import pytest

from offer_response import modeling
from offer_response.config import FEATURE_COLUMNS


@pytest.fixture
def toy_frame():
    """A small frame where the target is (mostly) a function of age."""
    rng = np.random.RandomState(0)
    n = 200
    age = rng.randint(18, 80, size=n)
    frame = pd.DataFrame(
        {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    )
    frame["age"] = age
    y = pd.Series((age > 50).astype(int), name="completed")
    return frame, y


def test_train_and_evaluate_returns_both_models(toy_frame):
    X, y = toy_frame
    models, metrics, split = modeling.train_and_evaluate(X, y)
    assert set(models) == {"logistic_regression", "decision_tree"}
    assert set(metrics) == {"logistic_regression", "decision_tree"}
    assert split["train_rows"] == 160
    assert split["test_rows"] == 40
    assert split["n_features"] == len(FEATURE_COLUMNS)


def test_metrics_are_in_valid_ranges(toy_frame):
    X, y = toy_frame
    _models, metrics, _split = modeling.train_and_evaluate(X, y)
    for name, m in metrics.items():
        assert set(m) == {"jaccard", "f1_weighted", "log_loss", "accuracy", "roc_auc"}
        for key in ("jaccard", "f1_weighted", "accuracy", "roc_auc"):
            assert 0.0 <= m[key] <= 1.0, f"{name}.{key} out of range"
        assert m["log_loss"] >= 0.0


def test_training_is_deterministic(toy_frame):
    X, y = toy_frame
    _m1, metrics1, _s1 = modeling.train_and_evaluate(X, y)
    _m2, metrics2, _s2 = modeling.train_and_evaluate(X, y)
    assert metrics1 == metrics2


def test_pipeline_scales_inside_fit(toy_frame):
    """The scaler must be part of the fitted pipeline, not fit beforehand."""
    X, y = toy_frame
    models, _metrics, _split = modeling.train_and_evaluate(X, y)
    for model in models.values():
        assert "scaler" in model.named_steps
        assert hasattr(model.named_steps["scaler"], "mean_")
