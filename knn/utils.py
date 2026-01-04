import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from .metrics import *

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
