"""Error analysis utilities for classification tasks."""

from collections import defaultdict

import numpy as np

from rapidfit.types import ClassMetrics, ErrorSample, TaskAnalysis


def compute_task_analysis(
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
    labels: list[str],
) -> TaskAnalysis:
    """Compute complete analysis for a single task."""
    n_labels = len(labels)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Confusion matrix
    matrix = [[0] * n_labels for _ in range(n_labels)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    # Per-class metrics
    class_metrics = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n_labels) if j != i)
        fn = sum(matrix[i][j] for j in range(n_labels) if j != i)
        support = sum(matrix[i])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[label] = ClassMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            support=support,
        )

    # Collect errors
    errors = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            errors.append(ErrorSample(
                text=texts[i],
                true_label=labels[t],
                predicted_label=labels[p],
                confidence=round(confidences[i], 4),
            ))

    # Sort by confidence descending (high-confidence errors first)
    errors.sort(key=lambda e: e["confidence"], reverse=True)

    return TaskAnalysis(
        accuracy=round(float((y_true_arr == y_pred_arr).mean()), 4),
        class_metrics=class_metrics,
        confusion_matrix=matrix,
        labels=labels,
        errors=errors,
    )
