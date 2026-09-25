"""
ensemble.py — Weighted ensemble of CatBoost + DeBERTa + BGE reranker scores.

The ensemble combines three independent probability signals:
  - P_catboost:   from LightGBM/CatBoost on 56 engineered features
  - P_deberta:    from DeBERTa cross-encoder fine-tuned on entity pairs
  - P_bge:        from BGE reranker (logit → sigmoid)

Final score = w1 * P_catboost + w2 * P_deberta + w3 * P_bge

Weights are tuned by optimizing F0.5 on the validation set using OOF predictions.

Key design decisions:
  - Each model sees the same candidate pairs (same blocking output)
  - Weights can be learned via Ridge regression on OOF scores
  - Threshold is always tuned on validation (never on test)
  - Singletons: if max score < SINGLETON_THRESHOLD → predict no match
"""

import numpy as np
from typing import Dict, Set, List, Tuple, Optional
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────────────
# F0.5 metric
# ─────────────────────────────────────────────────────────────────────────────

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def compute_f05_macro(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    scores = []
    for s1_eid, true_set in ground_truth.items():
        pred_set = predictions.get(s1_eid, set())
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(f_beta(p, r))
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble scoring
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleScorer:
    """
    Weighted linear ensemble of multiple model scores.

    Usage:
        scorer = EnsembleScorer(w_catboost=0.4, w_deberta=0.35, w_bge=0.25)
        final_scores = scorer.combine(catboost_scores, deberta_scores, bge_scores)
        predictions = scorer.predict(final_scores, pair_ids, threshold=0.55)
    """

    DEFAULT_WEIGHTS = {
        "catboost": 0.40,
        "deberta":  0.35,
        "bge":      0.25,
    }

    SINGLETON_THRESHOLD = 0.50   # if max(scores) < this → predict singleton

    def __init__(
        self,
        w_catboost: float = 0.40,
        w_deberta: float = 0.35,
        w_bge: float = 0.25,
        singleton_threshold: float = 0.50,
    ):
        total = w_catboost + w_deberta + w_bge
        self.w_catboost = w_catboost / total
        self.w_deberta = w_deberta / total
        self.w_bge = w_bge / total
        self.singleton_threshold = singleton_threshold

    def combine(
        self,
        catboost_scores: Optional[np.ndarray] = None,
        deberta_scores: Optional[np.ndarray] = None,
        bge_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Combine available model scores into a final probability.
        Missing models (None) are skipped with weight renormalized.
        """
        scores = []
        weights = []

        if catboost_scores is not None:
            scores.append(catboost_scores)
            weights.append(self.w_catboost)
        if deberta_scores is not None:
            scores.append(deberta_scores)
            weights.append(self.w_deberta)
        if bge_scores is not None:
            scores.append(bge_scores)
            weights.append(self.w_bge)

        if not scores:
            raise ValueError("At least one model score must be provided")

        total_w = sum(weights)
        result = np.zeros_like(scores[0], dtype=np.float32)
        for s, w in zip(scores, weights):
            result += s * (w / total_w)
        return result

    def predict(
        self,
        ensemble_scores: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        threshold: float = 0.55,
    ) -> Dict[str, Set[str]]:
        """Convert ensemble scores → {s1_eid → set of matched eids}."""
        preds: Dict[str, Set[str]] = {}
        for (s1_eid, cand_eid), score in zip(pair_ids, ensemble_scores):
            if score >= threshold:
                preds.setdefault(s1_eid, set()).add(cand_eid)
        return preds

    def predict_with_singleton_rule(
        self,
        ensemble_scores: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        threshold: float = 0.55,
    ) -> Dict[str, Set[str]]:
        """
        Same as predict() but applies the singleton rule:
        If max score for an S1 entity < singleton_threshold → predict no match.

        This improves precision for entities that barely exceeded the threshold.
        Critical for F0.5 (precision-weighted metric).
        """
        # Group by s1_eid
        entity_scores: Dict[str, List[Tuple[str, float]]] = {}
        for (s1_eid, cand_eid), score in zip(pair_ids, ensemble_scores):
            entity_scores.setdefault(s1_eid, []).append((cand_eid, float(score)))

        preds: Dict[str, Set[str]] = {}
        for s1_eid, cand_score_list in entity_scores.items():
            max_score = max(s for _, s in cand_score_list)
            if max_score < self.singleton_threshold:
                continue  # predict singleton
            for cand_eid, score in cand_score_list:
                if score >= threshold:
                    preds.setdefault(s1_eid, set()).add(cand_eid)
        return preds


# ─────────────────────────────────────────────────────────────────────────────
# Weight optimization via grid search
# ─────────────────────────────────────────────────────────────────────────────

def optimize_weights_grid(
    catboost_scores: np.ndarray,
    deberta_scores: Optional[np.ndarray],
    bge_scores: Optional[np.ndarray],
    pair_ids: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    threshold_range: Tuple[float, float] = (0.40, 0.90),
    threshold_steps: int = 25,
    weight_steps: int = 5,
) -> Tuple[float, float, float, float, float]:
    """
    Grid search over (w_catboost, w_deberta, w_bge) and threshold.

    Returns: (best_w_catboost, best_w_deberta, best_w_bge, best_threshold, best_f05)
    """
    best_f05 = 0.0
    best_params = (0.4, 0.35, 0.25, 0.55)

    w_range = np.linspace(0, 1, weight_steps)
    t_range = np.linspace(threshold_range[0], threshold_range[1], threshold_steps)

    print("  Optimizing ensemble weights...")

    for w1 in w_range:
        for w2 in w_range:
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            scorer = EnsembleScorer(w_catboost=w1 + 1e-9, w_deberta=w2 + 1e-9, w_bge=w3 + 1e-9)
            scores = scorer.combine(catboost_scores, deberta_scores, bge_scores)

            for t in t_range:
                preds = scorer.predict_with_singleton_rule(scores, pair_ids, threshold=t)
                f05 = compute_f05_macro(preds, ground_truth)
                if f05 > best_f05:
                    best_f05 = f05
                    best_params = (w1, w2, w3, t)

    w1, w2, w3, t = best_params
    print(f"  Best: w_catboost={w1:.2f} w_deberta={w2:.2f} w_bge={w3:.2f} "
          f"threshold={t:.3f} → F0.5={best_f05:.4f}")
    return w1, w2, w3, t, best_f05


# ─────────────────────────────────────────────────────────────────────────────
# Threshold tuning (single model)
# ─────────────────────────────────────────────────────────────────────────────

def tune_threshold(
    scores: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    n_steps: int = 50,
    use_singleton_rule: bool = True,
    singleton_threshold: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Sweep threshold from 0.05 to 0.95 and return (best_threshold, best_f05).
    """
    scorer = EnsembleScorer()
    if singleton_threshold is not None:
        scorer.singleton_threshold = singleton_threshold

    best_t, best_f05 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, n_steps):
        if use_singleton_rule:
            preds = scorer.predict_with_singleton_rule(scores, pair_ids, threshold=t)
        else:
            preds = scorer.predict(scores, pair_ids, threshold=t)
        f05 = compute_f05_macro(preds, ground_truth)
        if f05 > best_f05:
            best_f05, best_t = f05, t

    print(f"  Best threshold: {best_t:.3f} → F0.5 = {best_f05:.4f}")
    return best_t, best_f05


# ─────────────────────────────────────────────────────────────────────────────
# Score aggregation utilities
# ─────────────────────────────────────────────────────────────────────────────

def scores_dict_to_array(
    scores_dict: Dict[Tuple[str, str], float],
    pair_ids: List[Tuple[str, str]],
    default: float = 0.0,
) -> np.ndarray:
    """Convert {(s1_eid, cand_eid) → score} to a numpy array aligned with pair_ids."""
    return np.array([scores_dict.get(pid, default) for pid in pair_ids], dtype=np.float32)
