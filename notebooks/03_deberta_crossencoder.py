"""
notebook_03_deberta_crossencoder.py
=====================================
Fine-tune DeBERTa-v3-base as a cross-encoder for entity pair classification.

Requirements:
  - Colab with GPU (T4 or A100)
  - ~4-6 hours for 3 epochs on T4 with deberta-v3-base

Strategy:
  1. Run blocking to get candidate pairs
  2. Label pairs from ground truth (positives + hard negatives)
  3. Fine-tune DeBERTa on pair classification
  4. Score all test candidate pairs
  5. Combine with CatBoost in ensemble
"""

# ════════════════════════════════════════════════════════════
# CELL 1 — Setup (GPU required)
# ════════════════════════════════════════════════════════════
"""
# Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from google.colab import drive
drive.mount('/content/drive')

import os, sys
REPO_PATH = "/content/drive/MyDrive/Amazon_ML_challenge"
os.chdir(REPO_PATH)
sys.path.insert(0, REPO_PATH)

!pip install -q transformers accelerate scikit-learn pandas numpy lightgbm catboost rapidfuzz jellyfish tqdm scipy

# For DeBERTa-v3 (uses sentencepiece)
!pip install -q sentencepiece protobuf
"""

# ════════════════════════════════════════════════════════════
# CELL 2 — Fine-tune DeBERTa cross-encoder
# ════════════════════════════════════════════════════════════
"""
# Set model via env variable (or edit cross_encoder.py)
import os
os.environ["CROSS_ENCODER_MODEL"] = "microsoft/deberta-v3-base"

!python pipeline_v2.py --mode train-ce \
    --val-frac 0.05 \
    --max-train-samples 30000 \
    --top-k-name 30 \
    --top-k-addr 40 \
    --sn-window 10 \
    --hard-neg-ratio 3 \
    --ce-epochs 3 \
    --ce-batch-size 16 \
    --device cuda
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — Score test pairs with cross-encoder
# ════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import sys, os, pickle
sys.path.insert(0, '.')

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

from src.matching.cross_encoder import run_cross_encoder_scoring, load_cross_encoder
from src.blocking.blocking_deterministic import run_deterministic_blocking
from src.blocking.blocking_tfidf import run_tfidf_blocking
from src.blocking.candidate_union import union_candidates, filter_to_valid_ids
from src.evaluation.evaluate_blocking import parse_ground_truth

TEST_DIR = "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/test"
s1_test = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\\t")
s2_test = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\\t")
s3_test = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\\t")

# Load precomputed candidates from CatBoost run (save them to avoid recomputing)
# If not available, rerun blocking:
det_s2 = run_deterministic_blocking(s1_test, s2_test)
tfidf_s2 = run_tfidf_blocking(s1_test, s2_test, top_k_name=30, top_k_nameaddr=40)
cands_s2 = union_candidates(det_s2, tfidf_s2, max_candidates=100)

det_s3 = run_deterministic_blocking(s1_test, s3_test)
tfidf_s3 = run_tfidf_blocking(s1_test, s3_test, top_k_name=30, top_k_nameaddr=40)
cands_s3 = union_candidates(det_s3, tfidf_s3, max_candidates=100)

cands_test = {}
for eid in s1_test['entity_id']:
    cands_test[eid] = (cands_s2.get(eid, set()) | cands_s3.get(eid, set()))

s_test_all = pd.concat([s2_test, s3_test], ignore_index=True)
cands_test = filter_to_valid_ids(cands_test, set(s2_test['entity_id']), set(s3_test['entity_id']))

# Score with cross-encoder
ce_scores = run_cross_encoder_scoring(
    s1_test, s_test_all, cands_test,
    checkpoint_path="outputs/cross_encoder",
    device=device,
    batch_size=64,
)

# Save scores
with open("outputs/ce_scores_test.pkl", "wb") as f:
    pickle.dump(ce_scores, f)
print(f"Cross-encoder scores saved: {len(ce_scores):,} pairs")
"""

# ════════════════════════════════════════════════════════════
# CELL 4 — Evaluate cross-encoder on validation set
# ════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import sys, os, pickle
sys.path.insert(0, '.')

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

from src.matching.cross_encoder import run_cross_encoder_scoring
from src.matching.ensemble import tune_threshold, EnsembleScorer, compute_f05_macro
from src.matching.ensemble import scores_dict_to_array
from src.evaluation.evaluate_blocking import parse_ground_truth
from src.blocking.blocking_deterministic import run_deterministic_blocking
from src.blocking.blocking_tfidf import run_tfidf_blocking
from src.blocking.candidate_union import union_candidates

TRAIN_DIR = "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train"
s1_train = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\\t")
s2_train = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\\t")
gt = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\\t")
gt_dict = parse_ground_truth(gt)

s1_val = s1_train.sample(500, random_state=42)
s_all = pd.concat([s2_train, pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\\t")], ignore_index=True)

# Blocking for val
cands_val = union_candidates(
    run_deterministic_blocking(s1_val, s2_train),
    run_tfidf_blocking(s1_val, s2_train, top_k_name=30, top_k_nameaddr=40),
)

# Cross-encoder scoring
ce_val_scores = run_cross_encoder_scoring(
    s1_val, s_all, cands_val,
    checkpoint_path="outputs/cross_encoder",
    device=device, batch_size=64,
)

pair_ids = list(ce_val_scores.keys())
ce_arr = np.array([ce_val_scores[p] for p in pair_ids], dtype=np.float32)
val_gt = {k: gt_dict.get(k, set()) for k in s1_val['entity_id']}

best_t, best_f05 = tune_threshold(ce_arr, pair_ids, val_gt)
print(f"Cross-encoder val F0.5 = {best_f05:.4f} @ threshold={best_t:.3f}")
"""
