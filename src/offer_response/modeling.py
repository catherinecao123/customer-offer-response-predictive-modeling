"""Train and evaluate the offer-completion classifiers.

Models (kept from the original project):

- Logistic regression (``C=0.01``, liblinear solver).
- Decision tree (entropy criterion, ``max_depth=4``).

Two methodology fixes relative to the original notebook:

- Features are scaled inside an sklearn ``Pipeline``, so the scaler is fit
  on the training split only. The original notebook scaled the full data
  before splitting, leaking test-set statistics into training.
- The train/test split is stratified on the target.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE, TEST_SIZE


def make_logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(C=0.01, solver="liblinear", random_state=RANDOM_STATE),
            ),
        ]
    )


def make_decision_tree() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                DecisionTreeClassifier(
                    criterion="entropy", max_depth=4, random_state=RANDOM_STATE
                ),
            ),
        ]
    )


MODELS = {
    "logistic_regression": make_logistic_regression,
    "decision_tree": make_decision_tree,
}


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Score a fitted model: Jaccard, weighted F1, log loss, accuracy, ROC AUC."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "jaccard": round(float(jaccard_score(y_test, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "log_loss": round(float(log_loss(y_test, y_proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Split the data, train both models, and evaluate them.

    Returns ``(models, metrics, split_info)``.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    models: Dict[str, Pipeline] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for name, factory in MODELS.items():
        model = factory()
        model.fit(X_train, y_train)
        models[name] = model
        metrics[name] = evaluate_model(model, X_test, y_test)
    split_info = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "positive_rate": round(float(y.mean()), 4),
    }
    return models, metrics, split_info
