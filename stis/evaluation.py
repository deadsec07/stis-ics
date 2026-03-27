from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def contiguous_positive_windows(labels: pd.Series) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(labels.astype(int).tolist()):
        if value == 1 and start is None:
            start = idx
        if value == 0 and start is not None:
            windows.append((start, idx - 1))
            start = None
    if start is not None:
        windows.append((start, len(labels) - 1))
    return windows


def detection_delay(labels: pd.Series, predictions: pd.Series) -> float | None:
    windows = contiguous_positive_windows(labels)
    if not windows:
        return None
    delays = []
    pred = predictions.astype(int).tolist()
    for start, end in windows:
        hit = next((idx for idx in range(start, end + 1) if pred[idx] == 1), None)
        if hit is not None:
            delays.append(hit - start)
    if not delays:
        return None
    return float(np.mean(delays))


def classification_metrics(
    labels: pd.Series | None,
    predictions: pd.Series,
    scores: pd.Series,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"num_samples": int(len(scores))}
    if labels is None:
        return metrics
    valid_mask = labels.notna()
    metrics["num_labeled_samples"] = int(valid_mask.sum())
    if not valid_mask.any():
        return metrics
    y_true = labels.loc[valid_mask].astype(int)
    y_pred = predictions.loc[valid_mask].astype(int)
    scores = scores.loc[valid_mask]
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["detection_delay"] = detection_delay(y_true, y_pred)
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["roc_auc"] = None
    return metrics
