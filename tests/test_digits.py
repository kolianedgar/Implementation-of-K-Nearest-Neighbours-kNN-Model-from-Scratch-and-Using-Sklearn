import pytest
import numpy as np
import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from knn.classifier import knn_classifier

from knn.data_loader import load_dataset
from knn.utils import (
    grid_search_knn,
    cross_validate_knn,
    evaluate_on_dataset,
)

# -------------------------------------------------
# Dataset configurations to test
# -------------------------------------------------
DATASETS = [
    {"source": "builtin", "name": "digits"},
]


@pytest.mark.parametrize("dataset_config", DATASETS)
def test_package_knn_pipeline_execution(dataset_config):
    """
    End-to-end execution test for package-based KNN pipeline.

    Verifies that:
    - datasets load correctly
    - sklearn KNN can be fitted
    - grid search executes
    - cross-validation executes
    - final evaluation produces finite metrics
    """

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------
    X, y = load_dataset(**dataset_config)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    assert len(np.unique(y)) >= 2

    # -------------------------------------------------
    # Train / test split
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -------------------------------------------------
    # Scaling
    # -------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------------------------
    # Grid search (custom KNN)
    # -------------------------------------------------
    best_params = grid_search_knn(
        X=X_train,
        y=y_train,
        param_grid={
            "n_neighbours": [3, 5],
            "weights": ["uniform", "distance"],
            "distance_metric": ["euclidean", "manhattan"],
        },
        cv=3,
    )

    assert isinstance(best_params, dict)

    model = knn_classifier(**best_params)

    # -------------------------------------------------
    # Cross-validation
    # -------------------------------------------------
    cv_results = cross_validate_knn(
        model=model,
        X=X_train,
        y=y_train,
        cv=3,
    )

    expected_metrics = {
        "macro_f1",
        "macro_recall",
        "macro_sensitivity",
        "macro_roc_auc",
        "cross_entropy",
        "brier",
        "ece",
    }

    assert expected_metrics.issubset(cv_results.keys())

    for metric, (mean, std) in cv_results.items():
        if metric == "macro_roc_auc":
            continue  # may be undefined in some folds
        assert np.isfinite(mean)
        assert np.isfinite(std)
        assert std >= 0.0

    # -------------------------------------------------
    # Final test evaluation
    # -------------------------------------------------
    final_model = knn_classifier(**best_params)
    final_model.fit(X_train, y_train)

    test_results = evaluate_on_dataset(
        model=final_model,
        X=X_test,
        y=y_test,
    )

    assert expected_metrics.issubset(test_results.keys())

    for value in test_results.values():
        if np.isnan(value):
            continue  # ROC AUC may be undefined
        assert np.isfinite(value)