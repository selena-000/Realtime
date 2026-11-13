"""Metrics shared by training and evaluation."""

import numpy as np


def weighted_f1(y_true, y_pred, num_classes=7):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Targets and predictions must have the same shape.")
    if not y_true.size:
        return 0.0
    if ((y_true < 0) | (y_true >= num_classes)).any():
        raise ValueError("Target outside the label vocabulary.")
    if ((y_pred < 0) | (y_pred >= num_classes)).any():
        raise ValueError("Prediction outside the label vocabulary.")
    score = 0.0
    for label in range(num_classes):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        denom = 2 * tp + fp + fn
        score += (2 * tp / denom if denom else 0.0) * np.sum(y_true == label)
    return float(score / y_true.size)
