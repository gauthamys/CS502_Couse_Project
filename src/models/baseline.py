"""
Baseline models: XGBoost and Random Forest on ECFP fingerprints.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib
from pathlib import Path


CHECKPOINT_DIR = Path(__file__).parents[2] / "outputs" / "checkpoints"


def train_random_forest(X: np.ndarray, y: np.ndarray, **kwargs) -> RandomForestClassifier:
    defaults = dict(n_estimators=500, max_depth=None, n_jobs=-1, random_state=42)
    defaults.update(kwargs)
    clf = RandomForestClassifier(**defaults)
    clf.fit(X, y)
    return clf


def train_xgboost(X: np.ndarray, y: np.ndarray, **kwargs) -> xgb.XGBClassifier:
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    defaults = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    clf = xgb.XGBClassifier(**defaults)
    clf.fit(X, y)
    return clf


def save_model(model, name: str):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(name: str):
    path = CHECKPOINT_DIR / f"{name}.pkl"
    return joblib.load(path)
