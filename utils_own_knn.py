import numpy as np
from itertools import product
from sklearn.metrics import roc_auc_score
from knn_classifier import *

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
        metrics["macro_roc_auc"].append(macro_roc_auc(y[val_idx], y_prob))
        metrics["cross_entropy"].append(categorical_cross_entropy(y[val_idx], y_prob))
        metrics["brier"].append(multiclass_brier_score(y[val_idx], y_prob))
        metrics["ece"].append(expected_calibration_error(y[val_idx], y_prob))

    return {
        name: (np.mean(vals), np.std(vals))
        for name, vals in metrics.items()
    }

def print_cv_results(results, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for metric, (mean, std) in results.items():
        print(f"{metric:<30}: {mean:.6f} ± {std:.6f}")

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

def print_test_results(results):
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)

    for metric, value in results.items():
        print(f"{metric:<30}: {value:.6f}")
