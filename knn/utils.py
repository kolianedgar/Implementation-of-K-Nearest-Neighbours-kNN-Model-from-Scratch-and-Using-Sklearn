import numpy as np              #
from itertools import product   #   Import packages and functions from different libraries
import warnings                 #

from .classifier import *       # Import public functions related to the knn classifier 
from .metrics import *          # Import public functions related to performance/loss metrics

warnings.filterwarnings("ignore") #   Handle and hide warnings that can arise from usage of functions of other libraries.
                                  #   Source: https://stackoverflow.com/a/14463362

def _k_fold_split(X, y, n_splits=5, shuffle=True, random_state=42):
    """
        Generate train/validation indices for k-fold cross-validation.

        Splits the data into ``n_splits`` consecutive folds and yields
        indices for training and validation sets at each iteration.
        Optionally shuffles the data before splitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target labels.

        n_splits : int, default=5
            Number of folds.

        shuffle : bool, default=True
            Whether to shuffle the data before splitting into folds.

        random_state : int, default=42
            Seed for the random number generator used when shuffling.

        Yields
        ------
        train_idx : ndarray
            Indices of the training samples.

        val_idx : ndarray
            Indices of the validation samples.
    """
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
    """
        Perform grid search with cross-validation for a KNN classifier.

        Evaluates all parameter combinations specified in ``param_grid``
        using k-fold cross-validation and selects the parameters that
        maximize the mean macro-averaged F1 score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        y : array-like of shape (n_samples,)
            Target labels.

        param_grid : dict
            Dictionary with parameters names as keys and lists of parameter
            values to try. Expected keys are ``'n_neighbours'``, ``'weights'``,
            and ``'distance_metric'``.

        cv : int, default=5
            Number of folds used in cross-validation.

        Returns
        -------
        best_params : dict
            Parameter set that achieved the highest mean cross-validation
            score.
    """
    best_score = -np.inf
    best_params = None

    for n_neighbors, weights, metric in product(
        param_grid["n_neighbours"],
        param_grid["weights"],
        param_grid["distance_metric"]
    ):
        fold_scores = []

        for train_idx, val_idx in _k_fold_split(X, y, cv):
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
    """
        Evaluate a KNN classifier using cross-validation.

        Performs k-fold cross-validation and computes multiple
        performance and calibration metrics for each fold. The
        final result reports the mean and standard deviation of
        each metric across folds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        y : array-like of shape (n_samples,)
            Target labels.

        model_factory : callable
            Callable that returns a new, unfitted KNN classifier
            instance for each fold.

        cv : int, default=5
            Number of folds used in cross-validation.

        Returns
        -------
        results : dict
            Dictionary mapping metric names to ``(mean, std)``
            tuples computed across folds.
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

    for train_idx, val_idx in _k_fold_split(X, y, cv):
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
    """
        Evaluate a fitted model on a dataset using multiple metrics.

        Computes classification performance and calibration metrics
        based on predicted labels and class probabilities.

        Parameters
        ----------
        model : estimator
            Fitted classifier implementing ``predict`` and ``predict_prob``.

        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        y : array-like of shape (n_samples,)
            True target labels.

        Returns
        -------
        scores : dict
            Dictionary mapping metric names to their computed values
            on the given dataset.
    """
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