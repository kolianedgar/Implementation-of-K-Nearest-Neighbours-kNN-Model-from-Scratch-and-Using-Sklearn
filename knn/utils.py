import numpy as np
from itertools import product
from .classifier import *
from .metrics import *
import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

def k_fold_split(X, y, n_splits=5, shuffle=True, random_state=42):
    n = len(X)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    folds = np.array_split(indices, n_splits)

    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i])
        yield train_idx, val_idx

def grid_search_knn(X, y, param_grid, cv=5):
    best_score = -np.inf
    best_params = None

    for n_neighbors, weights, metric in product(
        param_grid["n_neighbours"],
        param_grid["weights"],
        param_grid["distance_metric"]
    ):
        fold_scores = []

        for train_idx, val_idx in k_fold_split(X, y, cv):
            model = knn_classifier(
                n_neighbours=n_neighbors,
                weights=weights,
                distance_metric=metric,
            )

            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            fold_scores.append(macro_f1(y[val_idx], y_pred))

        mean_score = np.mean(fold_scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                "n_neighbours": n_neighbors,
                "weights": weights,
                "distance_metric": metric,
            }

    return best_params

def cross_validate_knn(X, y, model_factory, cv=5):
    metrics = {
        "macro_f1": [],
        "macro_recall": [],
        "macro_sensitivity": [],
        "macro_roc_auc": [],
        "cross_entropy": [],
        "brier": [],
        "ece": [],
    }

    for train_idx, val_idx in k_fold_split(X, y, cv):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])

        y_pred = model.predict(X[val_idx])
        y_prob = model.predict_prob(X[val_idx])

        metrics["macro_f1"].append(macro_f1(y[val_idx], y_pred))
        metrics["macro_recall"].append(macro_recall(y[val_idx], y_pred))
        metrics["macro_sensitivity"].append(macro_sensitivity(y[val_idx], y_pred))
        value = macro_roc_auc(y[val_idx], y_prob)
        if not np.isnan(value):
            metrics["macro_roc_auc"].append(value)
        metrics["cross_entropy"].append(categorical_cross_entropy(y[val_idx], y_prob))
        metrics["brier"].append(multiclass_brier_score(y[val_idx], y_prob))
        metrics["ece"].append(expected_calibration_error(y[val_idx], y_prob))

    results = {}

    for name, vals in metrics.items():
        if len(vals) == 0:
            results[name] = (np.nan, np.nan)
        else:
            results[name] = (float(np.mean(vals)), float(np.std(vals)))

    return results

def evaluate_on_dataset(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_prob(X)

    return {
        "macro_f1": macro_f1(y, y_pred),
        "macro_recall": macro_recall(y, y_pred),
        "macro_sensitivity": macro_sensitivity(y, y_pred),
        "macro_roc_auc": macro_roc_auc(y, y_prob),
        "cross_entropy": categorical_cross_entropy(y, y_prob),
        "brier": multiclass_brier_score(y, y_prob),
        "ece": expected_calibration_error(y, y_prob),
    }
