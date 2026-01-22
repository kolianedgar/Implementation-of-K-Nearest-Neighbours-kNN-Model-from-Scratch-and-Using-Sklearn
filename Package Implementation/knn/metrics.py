import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def macro_recall(y_true, y_pred):
    """
        Compute the macro-averaged recall score.

        Recall is computed independently for each class and then averaged,
        giving equal weight to all classes.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        score : float
            Macro-averaged recall.
    """
    classes = np.unique(y_true)
    recalls = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recalls.append(tp / (tp + fn + 1e-12))

    return np.mean(recalls)

macro_sensitivity = macro_recall

def macro_f1(y_true, y_pred):
    """
        Compute the macro-averaged F1 score.

        The F1 score is computed independently for each class and then averaged,
        giving equal weight to all classes.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        score : float
            Macro-averaged F1 score.
    """
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
    """
        Compute the macro-averaged ROC AUC score.

        Supports binary and multiclass classification using a one-vs-rest
        strategy. Returns NaN if the score is undefined.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_prob : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.

        Returns
        -------
        score : float
            Macro-averaged ROC AUC score, or NaN if undefined.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # If only one class present, ROC AUC is undefined
    unique_classes = np.unique(y_true)
    if unique_classes.size < 2:
        return np.nan

    n_classes = y_prob.shape[1]

    # Ensure labels are in [0, n_classes - 1]
    if y_true.max() >= n_classes:
        return np.nan

    # One-hot encode only valid classes
    y_true_oh = np.zeros((y_true.size, n_classes))
    y_true_oh[np.arange(y_true.size), y_true] = 1

    try:
        return roc_auc_score(
            y_true_oh,
            y_prob,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        return np.nan

def categorical_cross_entropy(y_true, y_prob):
    """
    Compute the categorical cross-entropy loss.

    Samples whose true class is not present in the probability output
    are ignored.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted class probabilities.

    Returns
    -------
    loss : float
        Mean categorical cross-entropy, or NaN if undefined.
    """

    eps = 1e-12
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_classes = y_prob.shape[1]

    # Keep only samples whose true class exists in this fold
    valid_mask = y_true < n_classes

    if not np.any(valid_mask):
        return np.nan

    return -np.mean(
        np.log(
            y_prob[np.arange(len(y_true))[valid_mask], y_true[valid_mask]] + eps
        )
    )

def multiclass_brier_score(y_true, y_prob):
    """
        Compute the multiclass Brier score.

        The score measures the mean squared difference between predicted
        probabilities and the one-hot encoded true labels. Samples whose
        true class is not present in the probability output are ignored.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_prob : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.

        Returns
        -------
        score : float
            Multiclass Brier score, or NaN if undefined.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_classes = y_prob.shape[1]

    # Keep only samples whose true class exists in this fold
    valid_mask = y_true < n_classes

    if not np.any(valid_mask):
        return np.nan

    y_true_valid = y_true[valid_mask]
    y_prob_valid = y_prob[valid_mask]

    # One-hot encode only valid labels
    y_true_oh = np.zeros_like(y_prob_valid)
    y_true_oh[np.arange(len(y_true_valid)), y_true_valid] = 1.0

    return np.mean(np.sum((y_prob_valid - y_true_oh) ** 2, axis=1))

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
        Compute the Expected Calibration Error (ECE).

        The ECE measures the difference between confidence and accuracy
        by partitioning predictions into confidence bins.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_prob : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        n_bins : int, default=10
            Number of confidence bins.

        Returns
        -------
        ece : float
            Expected calibration error.
    """
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