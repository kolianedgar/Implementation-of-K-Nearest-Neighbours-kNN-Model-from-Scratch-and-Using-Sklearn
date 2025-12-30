import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
)

def load_builtin_dataset(name):
    name = name.lower()

    if name == "iris":
        X, y = load_iris(return_X_y=True)

    elif name == "wine":
        X, y = load_wine(return_X_y=True)

    elif name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)

    elif name == "digits":
        X, y = load_digits(return_X_y=True)

    else:
        raise ValueError(f"Unknown built-in dataset: {name}")

    return np.asarray(X), np.asarray(y)

def load_csv_dataset(
    filepath,
    target_column,
    drop_columns=None,
    encoding=True,
):
    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if drop_columns is not None:
        X = X.drop(columns=drop_columns)

    # Convert to numpy
    X = X.values

    # Encode target if needed
    if encoding and not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.values

    return X, y

def load_dataset(
    source,
    *,
    name=None,
    filepath=None,
    target_column=None,
    **kwargs
):
    if source == "builtin":
        if name is None:
            raise ValueError("Built-in dataset requires 'name'")
        return load_builtin_dataset(name)

    elif source == "csv":
        if filepath is None or target_column is None:
            raise ValueError("CSV dataset requires filepath and target_column")
        return load_csv_dataset(filepath, target_column, **kwargs)

    else:
        raise ValueError(f"Unknown dataset source: {source}")
    
def macro_recall(y_true, y_pred):
    classes = np.unique(y_true)
    recalls = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recalls.append(tp / (tp + fn + 1e-12))

    return np.mean(recalls)

macro_sensitivity = macro_recall

def macro_f1(y_true, y_pred):
    classes = np.unique(y_true)
    f1s = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        f1s.append(2 * precision * recall / (precision + recall + 1e-12))

    return np.mean(f1s)

def macro_roc_auc(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_true_oh = np.eye(n_classes)[y_true]
    return roc_auc_score(
        y_true_oh,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

def categorical_cross_entropy(y_true, y_prob):
    eps = 1e-12
    return -np.mean(np.log(y_prob[np.arange(len(y_true)), y_true] + eps))

def multiclass_brier_score(y_true, y_prob):
    y_true_oh = np.eye(y_prob.shape[1])[y_true]
    return np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1))

def expected_calibration_error(y_true, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.any(mask):
            acc = np.mean(accuracies[mask])
            conf = np.mean(confidences[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(y_true)

    return ece

def print_cv_results(results, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for metric, (mean, std) in results.items():
        print(f"{metric:<30}: {mean:.6f} ± {std:.6f}")

def print_test_results(results):
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)

    for metric, value in results.items():
        print(f"{metric:<30}: {value:.6f}")

def grid_search_knn(
    X,
    y,
    param_grid,
    cv=5,
    random_state=42,
):
    """
    Performs sklearn GridSearchCV for KNN.

    Parameters
    ----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Training labels
    param_grid : dict
        Hyperparameter grid with keys:
        - n_neighbours
        - weights
        - distance_metric
    cv : int
        Number of CV folds
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    grid : GridSearchCV
        Fitted GridSearchCV object
    """

    knn = KNeighborsClassifier()

    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    sklearn_param_grid = {
        "n_neighbors": param_grid["n_neighbours"],
        "weights": param_grid["weights"],
        "metric": param_grid["distance_metric"],
    }

    grid = GridSearchCV(
        estimator=knn,
        param_grid=sklearn_param_grid,
        scoring="f1_macro",
        cv=cv_strategy,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    grid.fit(X, y)

    return grid

def cross_validate_knn(model, X, y, cv=5, random_state=42):
    """
    Cross-validate a sklearn KNN model using custom metrics.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted KNeighborsClassifier instance
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    cv : int
        Number of CV folds
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Metric name -> (mean, std)
    """
    metrics = {
        "macro_f1": [],
        "macro_recall": [],
        "macro_sensitivity": [],
        "macro_roc_auc": [],
        "cross_entropy": [],
        "brier": [],
        "ece": [],
    }

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    for train_idx, val_idx in skf.split(X, y):
        # Refit model on each fold
        model.fit(X[train_idx], y[train_idx])

        y_pred = model.predict(X[val_idx])
        y_prob = model.predict_proba(X[val_idx])

        metrics["macro_f1"].append(macro_f1(y[val_idx], y_pred))
        metrics["macro_recall"].append(macro_recall(y[val_idx], y_pred))
        metrics["macro_sensitivity"].append(macro_sensitivity(y[val_idx], y_pred))
        metrics["macro_roc_auc"].append(macro_roc_auc(y[val_idx], y_prob))
        metrics["cross_entropy"].append(
            categorical_cross_entropy(y[val_idx], y_prob)
        )
        metrics["brier"].append(
            multiclass_brier_score(y[val_idx], y_prob)
        )
        metrics["ece"].append(
            expected_calibration_error(y[val_idx], y_prob)
        )

    return {
        metric: (np.mean(values), np.std(values))
        for metric, values in metrics.items()
    }

def evaluate_on_dataset(model, X, y):
    """
    Evaluate a fitted sklearn classifier on a dataset using custom metrics.

    Parameters
    ----------
    model : sklearn classifier
        Must implement predict and predict_proba
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        True labels

    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    return {
        "macro_f1": macro_f1(y, y_pred),
        "macro_recall": macro_recall(y, y_pred),
        "macro_sensitivity": macro_sensitivity(y, y_pred),
        "macro_roc_auc": macro_roc_auc(y, y_prob),
        "cross_entropy": categorical_cross_entropy(y, y_prob),
        "brier": multiclass_brier_score(y, y_prob),
        "ece": expected_calibration_error(y, y_prob),
    }
