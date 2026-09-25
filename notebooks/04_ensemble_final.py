"""
notebook_04_ensemble_final.py
================================
Combine CatBoost + DeBERTa + BGE Reranker scores into the final submission.

Prerequisites:
  - outputs/catboost_model.cbm           (from notebook_02)
  - outputs/ce_scores_test.pkl           (from notebook_03)
  - outputs/bge_scores_test.pkl          (optional, from BGE reranker)

This notebook:
  1. Loads all model scores
  2. Optimizes ensemble weights on validation F0.5
  3. Applies singleton rule
  4. Exports final matching_results.tsv
"""

# ════════════════════════════════════════════════════════════
# CELL 1 — Setup
# ════════════════════════════════════════════════════════════
"""
from google.colab import drive
drive.mount('/content/drive')

import os, sys
REPO_PATH = "/content/drive/MyDrive/Amazon_ML_challenge"
os.chdir(REPO_PATH)
sys.path.insert(0, REPO_PATH)

!pip install -q scikit-learn pandas numpy catboost lightgbm rapidfuzz jellyfish tqdm scipy
"""

# ════════════════════════════════════════════════════════════
# CELL 2 — Load test candidates + all model scores
# ════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import pickle, os, sys
sys.path.insert(0, '.')

from catboost import CatBoostClassifier
from src.matching.features_v2 import build_pair_feature_matrix_v2, FEATURE_NAMES_V2
from src.matching.ensemble import EnsembleScorer, tune_threshold, scores_dict_to_array
from src.matching.ensemble import optimize_weights_grid, compute_f05_macro
from src.blocking.blocking_deterministic import run_deterministic_blocking
from src.blocking.blocking_tfidf import run_tfidf_blocking
from src.blocking.candidate_union import union_candidates, filter_to_valid_ids, ensure_all_s1_covered
from src.evaluation.evaluate_blocking import parse_ground_truth

# ── Load test data
TEST_DIR = "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/test"
s1_test = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\\t")
s2_test = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\\t")
s3_test = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\\t")
s_test_all = pd.concat([s2_test, s3_test], ignore_index=True)

# ── Rebuild candidates (or load from cache)
CANDS_CACHE = "cache/test_candidates.pkl"
if os.path.exists(CANDS_CACHE):
    with open(CANDS_CACHE, "rb") as f:
        cands_test = pickle.load(f)
    print(f"Loaded cached candidates: {len(cands_test):,} S1 entities")
else:
    det_s2 = run_deterministic_blocking(s1_test, s2_test)
    tfidf_s2 = run_tfidf_blocking(s1_test, s2_test, top_k_name=30, top_k_nameaddr=40)
    cands_s2 = union_candidates(det_s2, tfidf_s2, max_candidates=100)

    det_s3 = run_deterministic_blocking(s1_test, s3_test)
    tfidf_s3 = run_tfidf_blocking(s1_test, s3_test, top_k_name=30, top_k_nameaddr=40)
    cands_s3 = union_candidates(det_s3, tfidf_s3, max_candidates=100)

    cands_test = {}
    for eid in s1_test['entity_id']:
        cands_test[eid] = (cands_s2.get(eid, set()) | cands_s3.get(eid, set()))
    cands_test = filter_to_valid_ids(cands_test, set(s2_test['entity_id']), set(s3_test['entity_id']))

    with open(CANDS_CACHE, "wb") as f:
        pickle.dump(cands_test, f)

# ── CatBoost features + scores
print("Building CatBoost features...")
X_test, _, test_pair_ids = build_pair_feature_matrix_v2(
    s1_test, s_test_all,
    {k: list(v) for k, v in cands_test.items()},
)

cb_model = CatBoostClassifier()
cb_model.load_model("outputs/catboost_model.cbm")
cb_probs = cb_model.predict_proba(X_test)[:, 1]
print(f"CatBoost scores: {len(cb_probs):,} pairs")

# ── Cross-encoder scores
ce_scores = None
if os.path.exists("outputs/ce_scores_test.pkl"):
    with open("outputs/ce_scores_test.pkl", "rb") as f:
        ce_scores_dict = pickle.load(f)
    ce_probs = scores_dict_to_array(ce_scores_dict, test_pair_ids, default=0.5)
    print(f"Cross-encoder scores: loaded")
else:
    print("Cross-encoder scores not found — using CatBoost only")
    ce_probs = None

# ── BGE Reranker scores (optional)
bge_probs = None
if os.path.exists("outputs/bge_scores_test.pkl"):
    with open("outputs/bge_scores_test.pkl", "rb") as f:
        bge_scores_dict = pickle.load(f)
    bge_probs = scores_dict_to_array(bge_scores_dict, test_pair_ids, default=0.5)
    print("BGE reranker scores: loaded")
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — Optimize weights on validation set
# ════════════════════════════════════════════════════════════
"""
# Load val scores (run analogous code on validation set to get these)
# For now we use default weights or optimize via grid search

import json

# Load saved threshold from CatBoost run
with open("outputs/threshold.json") as f:
    saved = json.load(f)
catboost_threshold = saved["threshold"]
print(f"CatBoost baseline threshold: {catboost_threshold:.3f}")
print(f"CatBoost baseline val F0.5:  {saved['val_f05']:.4f}")

# If we have multiple models, optimize weights
if ce_probs is not None and bge_probs is not None:
    # Build val scores analogously (omitted here for brevity — use your val split)
    print("Using default weights: CB=0.4, DeBERTa=0.35, BGE=0.25")
    w_cb, w_ce, w_bge = 0.40, 0.35, 0.25
elif ce_probs is not None:
    print("Using CB + DeBERTa: CB=0.45, DeBERTa=0.55")
    w_cb, w_ce, w_bge = 0.45, 0.55, 0.0
else:
    print("CatBoost only")
    w_cb, w_ce, w_bge = 1.0, 0.0, 0.0
"""

# ════════════════════════════════════════════════════════════
# CELL 4 — Generate final predictions
# ════════════════════════════════════════════════════════════
"""
import json
import pandas as pd

with open("outputs/threshold.json") as f:
    saved = json.load(f)

# Ensemble
scorer = EnsembleScorer(
    w_catboost=w_cb,
    w_deberta=w_ce,
    w_bge=w_bge,
    singleton_threshold=0.50,
)

final_scores = scorer.combine(
    catboost_scores=cb_probs,
    deberta_scores=ce_probs,
    bge_scores=bge_probs,
)

# Use catboost threshold as starting point; tune further if you have val scores
threshold = catboost_threshold
test_preds = scorer.predict_with_singleton_rule(final_scores, test_pair_ids, threshold=threshold)
ensure_all_s1_covered(test_preds, s1_test['entity_id'].tolist())

# Export
rows = []
for eid in s1_test['entity_id']:
    matches = test_preds.get(eid, set())
    rows.append({
        "source1_entity_id": eid,
        "matched_entity_ids": ",".join(sorted(matches)),
    })
df_out = pd.DataFrame(rows)
df_out.to_csv("outputs/matching_results_ensemble.tsv", sep="\\t", index=False)

n_match = sum(1 for r in rows if r["matched_entity_ids"])
n_singleton = len(rows) - n_match
print(f"Ensemble predictions: {n_match:,} matched, {n_singleton:,} singletons")
print("Saved: outputs/matching_results_ensemble.tsv")
"""

# ════════════════════════════════════════════════════════════
# CELL 5 — Final threshold sweep (if you have a held-out val set)
# ════════════════════════════════════════════════════════════
"""
# Use this to find the best threshold on your val split

# Assuming you have:
# val_final_scores: np.ndarray
# val_pair_ids: list of (s1_eid, cand_eid)
# val_gt: {s1_eid → set of true matches}

# best_t, best_f05 = tune_threshold(
#     val_final_scores, val_pair_ids, val_gt,
#     n_steps=50, use_singleton_rule=True
# )
# print(f"Best ensemble threshold: {best_t:.3f} → F0.5 = {best_f05:.4f}")
"""

# ════════════════════════════════════════════════════════════
# CELL 6 — Download final submission
# ════════════════════════════════════════════════════════════
"""
from google.colab import files

files.download("outputs/matching_results_ensemble.tsv")
files.download("outputs/candidate_pairs.tsv")

print("""
Submission checklist:
 [x] candidate_pairs.tsv
 [x] matching_results_ensemble.tsv  ← rename to matching_results.tsv before upload
 [ ] Zip both files together
 [ ] Upload to leaderboard
""")
"""
