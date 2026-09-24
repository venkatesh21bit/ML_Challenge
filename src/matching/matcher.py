"""
matcher.py — Pairwise matching model (LightGBM / CatBoost).

Takes candidate pairs + feature matrix → produces match / no-match probabilities.
Threshold is tuned to maximize F₀.₅ on validation.

Training:
  - Positive pairs: ground truth matches (in candidate set)
  - Negative pairs: non-matching candidates (hard negatives from blocking)
  - Class imbalance handled via scale_pos_weight

F₀.₅ formula: (1 + 0.25) * P * R / (0.25 * P + R)
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Set, List, Tuple, Optional

from src.matching.features import (
    build_pair_feature_matrix,
    FEATURE_NAMES,
)
from src.evaluation.evaluate_blocking import parse_ground_truth


# ─────────────────────────────────────────────────────────────────────────────
# F₀.₅ scorer
# ─────────────────────────────────────────────────────────────────────────────

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def compute_f05_macro(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, float, float]:
    """Macro-average F₀.₅ — same formula as the leaderboard."""
    f05_scores = []
    for s1_eid, true_set in ground_truth.items():
        pred_set = predictions.get(s1_eid, set())
        # Singleton: correctly predict empty → score=1.0; any match → score=0.0
        if not true_set:
            f05_scores.append(1.0 if not pred_set else 0.0)
            continue
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05_scores.append(f_beta(p, r, beta=0.5))
    macro_f05 = float(np.mean(f05_scores))
    return macro_f05, float(np.mean([p for p in f05_scores])), float(np.mean(f05_scores))


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimizer
# ─────────────────────────────────────────────────────────────────────────────

def tune_threshold(
    probs: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Sweep thresholds and return (best_threshold, best_f05).
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.02).tolist()

    best_t, best_f05 = 0.5, 0.0
    for t in thresholds:
        preds = build_predictions_from_probs(probs, pair_ids, threshold=t)
        f05, _, _ = compute_f05_macro(preds, ground_truth)
        if f05 > best_f05:
            best_f05, best_t = f05, t

    print(f"  Best threshold: {best_t:.3f} → F₀.₅ = {best_f05:.4f}")
    return best_t, best_f05


def build_predictions_from_probs(
    probs: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    threshold: float = 0.5,
) -> Dict[str, Set[str]]:
    """Convert probability array + pair ids → final prediction dict."""
    preds: Dict[str, Set[str]] = {}
    for (s1_eid, cand_eid), prob in zip(pair_ids, probs):
        if prob >= threshold:
            preds.setdefault(s1_eid, set()).add(cand_eid)
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM matcher
# ─────────────────────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    feature_names: Optional[List[str]] = None,
    model_path: Optional[str] = None,
):
    import lightgbm as lgb

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  LightGBM: {X_train.shape[0]:,} pairs, "
          f"pos_rate={y_train.mean():.4f}, scale_pos_weight={pos_weight:.1f}")

    params = {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "n_estimators":     n_estimators,
        "learning_rate":    learning_rate,
        "num_leaves":       num_leaves,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "scale_pos_weight": pos_weight,
        "verbosity":        -1,
        "n_jobs":           -1,
    }

    model = lgb.LGBMClassifier(**params)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]

    if X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
            feature_name=feature_names or FEATURE_NAMES,
        )
    else:
        model.fit(X_train, y_train, feature_name=feature_names or FEATURE_NAMES)

    if model_path:
        model.booster_.save_model(model_path)
        print(f"  Model saved to {model_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# CatBoost matcher (alternative)
# ─────────────────────────────────────────────────────────────────────────────

def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    iterations: int = 1000,
    learning_rate: float = 0.05,
    depth: int = 8,
    model_path: Optional[str] = None,
):
    from catboost import CatBoostClassifier, Pool

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  CatBoost: {X_train.shape[0]:,} pairs, scale_pos_weight={pos_weight:.1f}")

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        eval_metric="F1",
        scale_pos_weight=pos_weight,
        verbose=100,
        early_stopping_rounds=50,
        random_seed=42,
    )

    if X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    if model_path:
        model.save_model(model_path)
        print(f"  Model saved to {model_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Return probability of match (positive class)."""
    try:
        return model.predict_proba(X)[:, 1]
    except AttributeError:
        import lightgbm as lgb
        # booster loaded directly
        return model.predict(X)
