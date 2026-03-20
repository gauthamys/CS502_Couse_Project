"""
Imbalance strategy comparison for WDR91 DEL hit retrieval.

Strategies compared (all on XGBoost + ECFP6 unless noted):
  1. baseline        — no imbalance handling
  2. scale_pos_weight — built-in XGBoost upweighting (ratio=12.05)
  3. undersample_3   — random undersampling to 1:3 hit:non-hit ratio
  4. undersample_5   — random undersampling to 1:5
  5. lgbm_unbalance  — LightGBM is_unbalance=True
  6. rf_balanced     — Random Forest class_weight='balanced'

CV: StratifiedGroupKFold(n_splits=5) grouped by lib_prefix
    → prevents same DEL library appearing in both train and test.

Primary metric: Enrichment Factor @ 1% (EF1)
Secondary: AUPRC, AUROC, EF5%
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parents[2] / "outputs" / "predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Metrics ──────────────────────────────────────────────────────────────────

def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    n = len(y_true)
    n_top = max(1, int(n * frac))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    hits_in_top = y_true[top_idx].sum()
    total_hits = y_true.sum()
    if total_hits == 0:
        return 0.0
    return (hits_in_top / n_top) / (total_hits / n)


def score(y_true, y_prob) -> dict:
    return {
        "auroc":  roc_auc_score(y_true, y_prob),
        "auprc":  average_precision_score(y_true, y_prob),
        "ef1":    enrichment_factor(y_true, y_prob, 0.01),
        "ef5":    enrichment_factor(y_true, y_prob, 0.05),
    }


# ── Strategy runners ─────────────────────────────────────────────────────────

def _xgb_base_params(scale_pos_weight: float = 1.0) -> dict:
    return dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False,
    )


STRATEGIES = {
    "baseline": lambda X_tr, y_tr: (
        xgb.XGBClassifier(**_xgb_base_params(1.0)).fit(X_tr, y_tr), X_tr
    ),
    "scale_pos_weight": lambda X_tr, y_tr: (
        xgb.XGBClassifier(**_xgb_base_params(
            (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        )).fit(X_tr, y_tr), X_tr
    ),
    "undersample_3": lambda X_tr, y_tr: _xgb_with_undersample(X_tr, y_tr, ratio=1/3),
    "undersample_5": lambda X_tr, y_tr: _xgb_with_undersample(X_tr, y_tr, ratio=1/5),
    "lgbm_unbalance": lambda X_tr, y_tr: (
        lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
        ).fit(X_tr, y_tr), X_tr
    ),
    "rf_balanced": lambda X_tr, y_tr: (
        RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_features="sqrt", n_jobs=-1, random_state=42,
        ).fit(X_tr, y_tr), X_tr
    ),
}


def _xgb_with_undersample(X_tr, y_tr, ratio: float):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
    X_res, y_res = rus.fit_resample(X_tr, y_tr)
    model = xgb.XGBClassifier(**_xgb_base_params(1.0)).fit(X_res, y_res)
    return model, X_tr  # predict on full original train (unused — we pass X_te externally)


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_comparison(X: sp.csr_matrix, y: np.ndarray, groups: np.ndarray,
                   n_splits: int = 5) -> pd.DataFrame:
    """
    Run library-aware 5-fold CV for each strategy.
    Returns a DataFrame with per-fold metrics.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        n_hits_tr = y_tr.sum()
        n_hits_te = y_te.sum()
        print(f"\n  Fold {fold+1}/{n_splits} | "
              f"train={len(y_tr):,} ({n_hits_tr} hits) | "
              f"test={len(y_te):,} ({n_hits_te} hits)")

        if n_hits_te == 0:
            print("    [skip] no hits in test fold")
            continue

        for name, strategy_fn in STRATEGIES.items():
            print(f"    {name}...", end=" ", flush=True)
            try:
                model, _ = strategy_fn(X_tr, y_tr)
                y_prob = model.predict_proba(X_te)[:, 1]
                metrics = score(y_te, y_prob)
                records.append({"fold": fold + 1, "strategy": name, **metrics})
                print(f"EF1={metrics['ef1']:.2f}  AUPRC={metrics['auprc']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")

    return pd.DataFrame(records)


def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby("strategy")[["auroc", "auprc", "ef1", "ef5"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.sort_values("ef1_mean", ascending=False)
    return summary
