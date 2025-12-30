import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    label_binarize,
    LabelEncoder
)
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    log_loss,
    make_scorer
)

# =========================
# Dataset loading helpers
# =========================

def load_tabular_dataset(
    path,
    target_column,
    drop_columns=None
):
    """
    Loads a tabular dataset from disk and returns X, y.

    Parameters
    ----------
    path : str
        Path to CSV / Parquet file
    target_column : str
        Name of target column
    drop_columns : list[str] or None
        Optional columns to drop

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format")

    if drop_columns is not None:
        df = df.drop(columns=drop_columns)

    y = df[target_column].values
    X = df.drop(columns=[target_column]).values

    # Encode labels if needed
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    return X, y


# =========================
# Probability-based metrics
# =========================

def multiclass_brier_score(y_true, y_proba):
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)
    return np.mean(np.sum((y_proba - y_true_bin) ** 2, axis=1))


def expected_calibration_error(y_true, y_proba, n_bins=10):
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if np.any(mask):
            ece += np.abs(
                np.mean(accuracies[mask]) - np.mean(confidences[mask])
            ) * np.mean(mask)

    return ece


# =========================
# Scoring helpers
# =========================

def build_scoring_dict():
    return {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "recall_macro": make_scorer(recall_score, average="macro"),
        "sensitivity_macro": make_scorer(recall_score, average="macro"),

        "roc_auc_macro": make_scorer(
            roc_auc_score,
            response_method="predict_proba",
            multi_class="ovr",
            average="macro"
        ),

        "log_loss": make_scorer(
            log_loss,
            response_method="predict_proba",
            greater_is_better=False
        ),

        "brier_multiclass": make_scorer(
            multiclass_brier_score,
            response_method="predict_proba",
            greater_is_better=False
        ),

        "ece": make_scorer(
            expected_calibration_error,
            response_method="predict_proba",
            greater_is_better=False
        )
    }


def summarize_cv_results(cv_results):
    def _mean_std(values, higher_is_better=True):
        mean = np.mean(values)
        std = np.std(values)
        if not higher_is_better:
            mean = -mean
        return mean, std

    return {
        "f1_macro": _mean_std(cv_results["test_f1_macro"]),
        "recall_macro": _mean_std(cv_results["test_recall_macro"]),
        "sensitivity_macro": _mean_std(cv_results["test_sensitivity_macro"]),
        "roc_auc_macro": _mean_std(cv_results["test_roc_auc_macro"]),
        "log_loss": _mean_std(cv_results["test_log_loss"], False),
        "brier_multiclass": _mean_std(cv_results["test_brier_multiclass"], False),
        "ece": _mean_std(cv_results["test_ece"], False),
    }