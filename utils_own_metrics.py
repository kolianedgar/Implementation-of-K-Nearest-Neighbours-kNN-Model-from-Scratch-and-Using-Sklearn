import numpy as np
from sklearn.metrics import roc_auc_score
from knn_classifier import *

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