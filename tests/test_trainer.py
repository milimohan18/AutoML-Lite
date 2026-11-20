import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from automl_lite.trainer import train_and_evaluate


def make_split(X, y, stratify=None):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)


def test_train_classification_with_word_labels():
    # Word labels like the breast-cancer sample, which used to crash precision_score
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"size": rng.normal(size=120), "texture": rng.normal(size=120)})
    y = pd.Series(np.where(X["size"] + 0.3 * rng.normal(size=120) > 0, "malignant", "benign"))

    X_train, X_test, y_train, y_test = make_split(X, y, stratify=y)
    _, leaderboard, best = train_and_evaluate(X_train, y_train, X_test, y_test, "classification")

    assert len(leaderboard) == 3
    assert set(best["test_metrics"]) == {"accuracy", "precision", "recall", "f1_score"}
    assert all(0.0 <= v <= 1.0 for v in best["test_metrics"].values())


def test_train_regression():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    y = pd.Series(3 * X["a"] - 2 * X["b"] + rng.normal(scale=0.1, size=120))

    X_train, X_test, y_train, y_test = make_split(X, y)
    _, leaderboard, best = train_and_evaluate(X_train, y_train, X_test, y_test, "regression")

    assert len(leaderboard) == 3
    assert set(best["test_metrics"]) == {"r2", "mae", "rmse"}
    assert best["test_metrics"]["rmse"] >= 0
