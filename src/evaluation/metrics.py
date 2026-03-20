"""
Evaluation metrics for hit retrieval.
Primary metric: Enrichment Factor (EF) at various top-k cutoffs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.01) -> float:
    """
    Enrichment Factor at `fraction` of the ranked list.
    EF = (hits in top X%) / (expected hits by random at X%)
    """
    n = len(y_true)
    n_top = max(1, int(n * fraction))
    top_indices = np.argsort(y_score)[::-1][:n_top]
    hits_in_top = y_true[top_indices].sum()
    total_hits = y_true.sum()
    if total_hits == 0:
        return 0.0
    random_expected = total_hits * fraction
    return hits_in_top / random_expected


def hit_rate(y_true: np.ndarray, y_score: np.ndarray, n_top: int = 1000) -> float:
    """Precision in the top-n_top predictions."""
    top_indices = np.argsort(y_score)[::-1][:n_top]
    return y_true[top_indices].mean()


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute all relevant metrics."""
    return {
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
        "ef_1pct": enrichment_factor(y_true, y_score, fraction=0.01),
        "ef_5pct": enrichment_factor(y_true, y_score, fraction=0.05),
        "hit_rate_top100": hit_rate(y_true, y_score, n_top=100),
        "hit_rate_top1000": hit_rate(y_true, y_score, n_top=1000),
    }
