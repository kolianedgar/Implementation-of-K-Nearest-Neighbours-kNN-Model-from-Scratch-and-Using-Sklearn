import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from knn.metrics import *
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import copy

def grid_search_knn(X, y, param_grid, cv=5):
    """
        Perform grid search with cross-validation to select optimal KNN hyperparameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        param_grid : dict
            Dictionary specifying hyperparameter search space. Expected keys are
            'n_neighbours', 'weights', and 'distance_metric'.
        cv : int, default=5
            Number of stratified cross-validation folds.

        Returns
        -------
        dict
            Dictionary containing the best hyperparameters found.
    """

    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42,
    )

    gs = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid={
            "n_neighbors": param_grid["n_neighbours"],
            "weights": param_grid["weights"],
            "metric": param_grid["distance_metric"],
        },
        scoring="f1_macro",
        cv=cv_strategy,
        n_jobs=-1,
    )

    gs.fit(X, y)

    # 🔑 THIS IS THE FIX
    return {
        "n_neighbours": gs.best_params_["n_neighbors"],
        "weights": gs.best_params_["weights"],
        "distance_metric": gs.best_params_["metric"],
    }

def cross_validate_knn(
    model,
    X,
    y,
    cv=5,
):
    """
        Perform stratified cross-validation for a KNN-based classifier using custom metrics.

        Parameters
        ----------
        model : sklearn-compatible classifier
            A fitted or unfitted classifier implementing fit, predict, and predict_proba.
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        cv : int, default=5
            Number of stratified cross-validation folds.

        Returns
        -------
        dict
            Dictionary mapping metric names to tuples of (mean, standard deviation)
            computed across folds.
    """
    
    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42,
    )

    metrics = {
        "macro_f1": [],
        "macro_recall": [],
        "macro_sensitivity": [],
        "macro_roc_auc": [],
        "cross_entropy": [],
        "brier": [],
        "ece": [],
    }

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone model safely (custom estimator)
        clf = copy.deepcopy(model)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)

        # -------------------------------
        # Metrics per fold
        # -------------------------------
        metrics["macro_f1"].append(macro_f1(y_val, y_pred))
        metrics["macro_recall"].append(macro_recall(y_val, y_pred))
        metrics["macro_sensitivity"].append(macro_sensitivity(y_val, y_pred))

        # Suppress sklearn ROC-AUC warnings locally
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UndefinedMetricWarning
            )

            metrics["macro_roc_auc"].append(
                macro_roc_auc(y_val, y_prob)
            )

        metrics["cross_entropy"].append(
            categorical_cross_entropy(y_val, y_prob)
        )
        metrics["brier"].append(
            multiclass_brier_score(y_val, y_prob)
        )
        metrics["ece"].append(
            expected_calibration_error(y_val, y_prob)
        )

    # -------------------------------
    # Aggregate mean ± std
    # -------------------------------
    results = {}

    for metric, values in metrics.items():
        values = np.asarray(values, dtype=float)

        if metric == "macro_roc_auc":
            # STRICT POLICY:
            # if any fold is invalid → drop metric entirely
            if np.any(np.isnan(values)):
                results[metric] = (np.nan, np.nan)
            else:
                results[metric] = (values.mean(), values.std())
        else:
            results[metric] = (np.nanmean(values), np.nanstd(values))

    return results

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
