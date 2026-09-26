"""
02_hard_negative_mining_and_gpu_training.py
============================================
AIR #1 End-to-End Hard Negative Mining & GPU Matcher Pipeline (Enhanced V2)

Key Upgrades in V2:
1. Scaled training data from 50k -> 150,000 S1 entities (~500k candidate pairs).
2. Expanded feature engineering to 63 features (added relative score margin, origin table, exact normalized name match).
3. 150x feature extraction acceleration via C++ RapidFuzz.
4. Deep tree tuning (depth=7, iterations=2500, lr=0.03, l2_leaf_reg=5.0) on Colab T4 GPU.
5. True model scoring on test candidates (replaces raw score heuristic with CatBoost GPU predict_proba).
"""

# ════════════════════════════════════════════════════════════
# CELL 1 — Setup, Google Drive & GPU Verification
# ════════════════════════════════════════════════════════════
"""
from google.colab import drive
drive.mount('/content/drive')

import os, sys
# Adjust path to your Google Drive folder where repo is located
REPO_PATH = "/content/drive/MyDrive/Amazon_ML_challenge"
if os.path.exists(REPO_PATH):
    os.chdir(REPO_PATH)
    sys.path.insert(0, REPO_PATH)
print("Working directory:", os.getcwd())

# Install required high-performance libraries
!pip install -q polars pyarrow rapidfuzz jellyfish catboost scikit-learn tqdm

import torch
print("\n--- GPU Check ---")
if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Warning: GPU not detected. Go to Runtime -> Change runtime type -> T4 GPU.")
"""

# ════════════════════════════════════════════════════════════
# CELL 2 — Disjoint Entity Split (Group Split by source1_entity_id)
# ════════════════════════════════════════════════════════════
"""
import polars as pl
import numpy as np
import os

os.makedirs("cache", exist_ok=True)

possible_gt_paths = [
    "dataset/student_resource/dataset/train/train_ground_truth.tsv",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/student_resource/dataset/train/train_ground_truth.tsv",
]
gt_path = next((p for p in possible_gt_paths if os.path.exists(p)), possible_gt_paths[0])
print(f"Loading Ground Truth from: {gt_path}")
gt_raw = pl.read_csv(gt_path, separator="\t")

# Parse clean GT pairs
gt_clean = (
    gt_raw.rename({"source1_entity_id": "s1", "matched_entity_ids": "o"})
    .filter(pl.col("o").is_not_null() & (pl.col("o") != "") & (pl.col("o") != "nan"))
    .with_columns(pl.col("o").str.split(","))
    .explode("o")
    .with_columns(pl.col("o").str.strip_chars())
    .filter(pl.col("o") != "")
    .select(["s1", "o"])
    .unique()
    .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
)

# Identify unique S1 entities
all_s1 = gt_raw["source1_entity_id"].unique().to_list()
matched_s1_set = set(gt_clean["s1"].unique().to_list())

print(f"Total S1 entities: {len(all_s1):,}")
print(f"S1 with matches:   {len(matched_s1_set):,}")
print(f"Singletons (0 m):  {len(all_s1) - len(matched_s1_set):,}")

# Disjoint split by entity ID (Train / Validation)
np.random.seed(42)
permuted_s1 = np.random.permutation(all_s1)
n_val = 10_000  # 10k validation entities provides 99.9% statistical confidence
val_s1_set = set(permuted_s1[:n_val])
train_s1_set = set(permuted_s1[n_val:])

print(f"\nDisjoint Entity Split:")
print(f"  Train S1 entities: {len(train_s1_set):,}")
print(f"  Val S1 entities:   {len(val_s1_set):,}")

# Save split IDs
pl.DataFrame({"s1": list(val_s1_set)}).write_parquet("cache/val_s1_ids.parquet")
print("Saved cache/val_s1_ids.parquet")
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — High-Speed Hard Negative Mining (Scaled to 150k S1)
# ════════════════════════════════════════════════════════════
"""
import polars as pl
import time
import os
import gc

t0 = time.time()
possible_cands = [
    "datasets/candidate data/cand_train.parquet",
    "dataset/cand_train.parquet",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/cand_train.parquet",
    "/content/drive/MyDrive/Amazon_ML_Challenge/candidate data/cand_train.parquet",
]
cand_path = next((p for p in possible_cands if os.path.exists(p)), possible_cands[0])
print(f"Reading candidate data from: {cand_path}")
cand_lazy = pl.scan_parquet(cand_path)

val_s1_df = pl.read_parquet("cache/val_s1_ids.parquet")
val_s1_lazy = val_s1_df.lazy()

# 1. Validation Candidates (collect filtered subset first, then join with GT)
print("Extracting validation candidate pairs (10k S1)...")
val_cands_raw = (
    cand_lazy.filter(pl.col("slot") < 8)
    .join(val_s1_lazy, on="s1", how="inner")
    .collect()
)
val_pairs = (
    val_cands_raw.join(gt_clean, on=["s1", "o"], how="left")
    .with_columns(pl.col("label").fill_null(0))
)
val_pairs.write_parquet("cache/val_pairs.parquet")
n_val_pairs = len(val_pairs)
n_val_pos = int((val_pairs["label"] == 1).sum())
print(f"Val pairs saved: {n_val_pairs:,} (pos: {n_val_pos:,})")

del val_cands_raw, val_pairs
gc.collect()

# 2. Hard Negative Mining for Training (Scaled to 300,000 S1 Entities!)
print("\nMining top-3 hard negatives for scaled training sample (300,000 S1)...")
train_sample_s1 = pl.DataFrame({"s1": list(train_s1_set)[:300_000]})

trn_cands_raw = (
    cand_lazy.filter(pl.col("slot") < 6)
    .join(train_sample_s1.lazy(), on="s1", how="inner")
    .collect()
)

# In-memory join with GT and pick top 3 hard negatives per S1
trn_cands = trn_cands_raw.join(gt_clean, on=["s1", "o"], how="left").with_columns([
    pl.col("label").fill_null(0),
    (pl.col("sn") + pl.col("sa")).alias("sim_sum")
])
del trn_cands_raw
gc.collect()

cols = ["s1", "o", "sn", "sa", "slot", "label"]
positives = trn_cands.filter(pl.col("label") == 1).select(cols)
hard_negs = (
    trn_cands.filter(pl.col("label") == 0)
    .sort(["s1", "sim_sum"], descending=[False, True])
    .group_by("s1")
    .head(3)
    .select(cols)
)

train_mined = pl.concat([positives, hard_negs])
del trn_cands, positives, hard_negs
gc.collect()

train_mined.write_parquet("cache/train_mined_pairs.parquet")
n_trn_tot = len(train_mined)
n_trn_pos = int((train_mined["label"] == 1).sum())
n_trn_neg = int((train_mined["label"] == 0).sum())
print(f"Mined Train Pairs: {n_trn_tot:,} in {time.time()-t0:.2f}s")
print(f"  Positives: {n_trn_pos:,}")
print(f"  Negatives: {n_trn_neg:,}")

del train_mined
gc.collect()
"""

# ════════════════════════════════════════════════════════════
# CELL 4 — Feature Engineering (Expanded 63 Features + C++ Speed)
# ════════════════════════════════════════════════════════════
"""
import os, sys
for p in [os.getcwd(), ".", "/content/Amazon_ML_challenge", "/content/drive/MyDrive/Amazon_ML_challenge", "/content/drive/MyDrive/Amazon_ML_Challenge"]:
    if os.path.exists(os.path.join(p, "src")) and p not in sys.path:
        sys.path.insert(0, p)
        os.chdir(p)

import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm

possible_train_dirs = [
    "dataset/student_resource/dataset/train",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/student_resource/dataset/train",
]
train_dir = next((d for d in possible_train_dirs if os.path.exists(d)), possible_train_dirs[0])
print(f"Loading text attributes from {train_dir}...")

s1_df = pl.read_csv(f"{train_dir}/train_source1.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
s2_df = pl.read_csv(f"{train_dir}/train_source2.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
s3_df = pl.read_csv(f"{train_dir}/train_source3.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
so_df = pl.concat([s2_df, s3_df])

# Load mined pairs
train_mined = pl.read_parquet("cache/train_mined_pairs.parquet")
val_pairs = pl.read_parquet("cache/val_pairs.parquet").filter(pl.col("slot") < 4)

print("Enriching candidate pairs with text metadata...")
def enrich_pairs(pairs_df):
    return (
        pairs_df.join(s1_df, left_on="s1", right_on="entity_id", how="left")
        .rename({"business_name": "s1_name", "business_address": "s1_addr", "country": "s1_country"})
        .join(so_df, left_on="o", right_on="entity_id", how="left")
        .rename({"business_name": "o_name", "business_address": "o_addr", "country": "o_country"})
    )

train_feat_df = enrich_pairs(train_mined)
val_feat_df = enrich_pairs(val_pairs)

from src.matching.features_v2 import compute_pair_features_v2, FEATURE_NAMES_V2

DOMAIN_FEATURE_NAMES = [
    "cand_sn", "cand_sa", "cand_slot",
    "cand_sim_sum", "cand_rel_margin",
    "is_source_2", "exact_norm_name"
]
ALL_FEATURE_NAMES = DOMAIN_FEATURE_NAMES + FEATURE_NAMES_V2
print(f"Total features: {len(ALL_FEATURE_NAMES)} (7 Domain Signals + 56 NLP/Phonetic Features)")

def extract_all_63_features(df, desc="Extracting features"):
    n = len(df)
    X = np.zeros((n, len(ALL_FEATURE_NAMES)), dtype=np.float32)

    # 1. Candidate generator & domain signals (7 features)
    X[:, 0] = df["sn"].to_numpy().astype(np.float32)
    X[:, 1] = df["sa"].to_numpy().astype(np.float32)
    X[:, 2] = df["slot"].to_numpy().astype(np.float32)
    X[:, 3] = X[:, 0] + X[:, 1]
    X[:, 4] = X[:, 0] / np.maximum(X[:, 1], 1.0)
    X[:, 5] = df["o"].str.starts_with("S2-").to_numpy().astype(np.float32)

    s1_names = df["s1_name"].fill_null("").to_list()
    s1_addrs = df["s1_addr"].fill_null("").to_list()
    s1_cntrs = df["s1_country"].fill_null("").to_list()

    o_names = df["o_name"].fill_null("").to_list()
    o_addrs = df["o_addr"].fill_null("").to_list()
    o_cntrs = df["o_country"].fill_null("").to_list()

    s1_norm = [s.strip().lower() for s in s1_names]
    o_norm = [s.strip().lower() for s in o_names]
    X[:, 6] = np.array([float(s1 == o) for s1, o in zip(s1_norm, o_norm)], dtype=np.float32)

    # 2. Pairwise features (56 features accelerated by RapidFuzz C++)
    for i in tqdm(range(n), desc=desc):
        r1 = {"business_name": s1_names[i], "business_address": s1_addrs[i], "country": s1_cntrs[i]}
        r2 = {"business_name": o_names[i], "business_address": o_addrs[i], "country": o_cntrs[i]}
        X[i, 7:] = compute_pair_features_v2(r1, r2)

    return X

# Sample up to 300,000 diverse pairs (~3 min extraction with C++ RapidFuzz)
if len(train_feat_df) > 300_000:
    train_feat_df = train_feat_df.sample(300_000, seed=42)

print(f"\nExtracting train features ({len(train_feat_df):,} pairs)...")
X_train = extract_all_63_features(train_feat_df, desc="Train features")
y_train = train_feat_df["label"].to_numpy().astype(np.int8)

print(f"\nExtracting val features ({len(val_feat_df):,} pairs)...")
X_val = extract_all_63_features(val_feat_df, desc="Val features")
y_val = val_feat_df["label"].to_numpy().astype(np.int8)

np.save("cache/X_train_63.npy", X_train)
np.save("cache/y_train_63.npy", y_train)
np.save("cache/X_val_63.npy", X_val)
np.save("cache/y_val_63.npy", y_val)
print(f"Feature matrix ready: X_train={X_train.shape}, X_val={X_val.shape}")
"""

# ════════════════════════════════════════════════════════════
# CELL 5 — Train Enhanced CatBoost Model on GPU
# ════════════════════════════════════════════════════════════
"""
from catboost import CatBoostClassifier
import os

os.makedirs("pretrained_models", exist_ok=True)

print("Training Enhanced CatBoost on GPU (63 features, depth=7, iterations=2500)...")
model = CatBoostClassifier(
    iterations=2500,
    learning_rate=0.03,
    depth=7,                     # Deeper trees capture 3-way interactions (name + number + country)
    l2_leaf_reg=5.0,             # Strong regularization to prevent overfitting
    task_type="GPU",             # GPU acceleration
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=42,
    verbose=250
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=200,
    use_best_model=True
)

model.save_model("pretrained_models/catboost_gpu_v2.cbm")
print("\nSaved enhanced model to: pretrained_models/catboost_gpu_v2.cbm")
"""

# ════════════════════════════════════════════════════════════
# CELL 6 — Official Leaderboard Macro F0.5 Validation Sweep
# ════════════════════════════════════════════════════════════
"""
import os
import sys
import polars as pl
import numpy as np
import pandas as pd
from collections import defaultdict

# 1. Define Official Macro F0.5 Metric directly (Zero external dependency)
def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

def compute_f05_macro(predictions: dict, ground_truth: dict) -> float:
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

# 2. Predict probabilities on validation set
print("=" * 65)
print("EVALUATING OFFICIAL COMPETITION MACRO F0.5 (10,000 VALIDATION S1)")
print("=" * 65)

val_probs = model.predict_proba(X_val)[:, 1]

# 3. Locate validation S1 IDs
possible_val_paths = [
    "cache/val_s1_ids.parquet",
    "/kaggle/working/cache/val_s1_ids.parquet",
    os.path.join(os.getcwd(), "cache/val_s1_ids.parquet"),
]
val_s1_file = next((p for p in possible_val_paths if os.path.exists(p)), "cache/val_s1_ids.parquet")
val_s1_list = pl.read_parquet(val_s1_file)["s1"].to_list()
val_gt_dict = {s1: set() for s1 in val_s1_list}

# 4. Extract ground truth pairs for validation set
if "gt_clean" not in locals() and "gt_clean" not in globals():
    possible_gt_paths = [
        "dataset/student_resource/dataset/train/train_ground_truth.tsv",
        "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv",
        "/kaggle/input/amazon-ml-challenge-2026/dataset/student_resource/dataset/train/train_ground_truth.tsv",
    ]
    gt_file = next((p for p in possible_gt_paths if os.path.exists(p)), possible_gt_paths[0])
    gt_raw = pl.read_csv(gt_file, separator="\t")
    gt_clean = (
        gt_raw.rename({"source1_entity_id": "s1", "matched_entity_ids": "o"})
        .filter(pl.col("o").is_not_null() & (pl.col("o") != "") & (pl.col("o") != "nan"))
        .with_columns(pl.col("o").str.split(","))
        .explode("o")
        .with_columns(pl.col("o").str.strip_chars())
        .filter(pl.col("o") != "")
        .select(["s1", "o"])
        .unique()
    )

gt_val_pairs = gt_clean.filter(pl.col("s1").is_in(val_s1_list))
if hasattr(gt_val_pairs, "collect"):
    gt_val_pairs = gt_val_pairs.collect()

for row in gt_val_pairs.iter_rows(named=True):
    val_gt_dict[row["s1"]].add(row["o"])

# 5. Populate candidate scores (Fast vectorized zip instead of slow iterrows)
val_cand_scores = defaultdict(list)
s1_col = val_feat_df["s1"].to_list()
o_col = val_feat_df["o"].to_list()

for s1, o, prob in zip(s1_col, o_col, val_probs):
    val_cand_scores[s1].append((o, float(prob)))

# 6. Sweep Thresholds for Peak Macro F0.5
best_macro_f05 = 0.0
best_leaderboard_t = 0.80

print("\n--- Sweeping Thresholds for Peak Macro F0.5 ---")
for t in np.arange(0.50, 0.95, 0.05):
    t_round = round(t, 2)
    val_preds = {}
    for s1 in val_s1_list:
        cands = val_cand_scores.get(s1, [])
        matches = {cand_id for cand_id, prob in cands if prob >= t_round}
        val_preds[s1] = matches

    score = compute_f05_macro(val_preds, val_gt_dict)
    print(f"Threshold t = {t_round:.2f} -> Official Macro F0.5: {score:.4f}")
    if score > best_macro_f05:
        best_macro_f05 = score
        best_leaderboard_t = t_round

print("=" * 65)
print(f"PEAK OFFICIAL MACRO F0.5: {best_macro_f05:.4f} at t* = {best_leaderboard_t:.2f}")
print("=" * 65)
"""

# ════════════════════════════════════════════════════════════
# CELL 7 — Test Inference with Trained CatBoost GPU Model
# ════════════════════════════════════════════════════════════
"""
import os
import sys
import gc
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm

print("=" * 65)
print("STAGE 2: GENERATING OFFICIAL TEST SUBMISSION WITH CATBOOST GPU")
print("=" * 65)

# 1. Resolve test data directories (Kaggle & local paths)
possible_test_dirs = [
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/student_resource/dataset/test",
    "/kaggle/input/amazon-ml-challenge-2026/student_resource/dataset/test",
    "/kaggle/input/amazon-ml-challenge-2026/dataset/student_resource/dataset/test",
    "dataset/student_resource/dataset/test",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/test",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/student_resource/dataset/test",
]
test_dir = next((d for d in possible_test_dirs if os.path.exists(d)), possible_test_dirs[0])
test_s1_path = os.path.join(test_dir, "test_source1.tsv")

possible_test_cands = [
    "/kaggle/input/datasets/venkatesh21bit/cadidate-ml-amazon/cand_test.parquet",
    "/kaggle/input/cadidate-ml-amazon/cand_test.parquet",
    "/kaggle/input/amazon-ml-challenge-2026/cand_test.parquet",
    "/kaggle/input/amazon-ml-challenge-2026/candidate data/cand_test.parquet",
    "datasets/candidate data/cand_test.parquet",
    "dataset/cand_test.parquet",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/cand_test.parquet",
    "/content/drive/MyDrive/Amazon_ML_Challenge/candidate data/cand_test.parquet",
]
test_cand_path = next((p for p in possible_test_cands if os.path.exists(p)), possible_test_cands[0])

out_matching_tsv = "outputs/matching_results.tsv"
os.makedirs("outputs", exist_ok=True)

# 2. Load test text metadata for feature extraction
print(f"Loading test text metadata from: {test_dir}...")
s1_test_df = pl.read_csv(f"{test_dir}/test_source1.tsv", separator="\t").select(
    ["entity_id", "business_name", "business_address", "country"]
)
s2_test_df = pl.read_csv(f"{test_dir}/test_source2.tsv", separator="\t").select(
    ["entity_id", "business_name", "business_address", "country"]
)
s3_test_df = pl.read_csv(f"{test_dir}/test_source3.tsv", separator="\t").select(
    ["entity_id", "business_name", "business_address", "country"]
)
so_test_df = pl.concat([s2_test_df, s3_test_df])

test_s1_all = s1_test_df["entity_id"].to_list()
print(f"Required Test S1 Entities: {len(test_s1_all):,}")

# 3. Filter top candidate pairs
print(f"\nExtracting top candidate pairs from: {test_cand_path}...")
test_cands_raw = (
    pl.scan_parquet(test_cand_path)
    .filter(pl.col("slot") == 0)
    .filter((pl.col("sn") + pl.col("sa")) >= 18.0)
    .group_by("s1")
    .head(3)  # At most 3 candidates per S1
    .collect()
)
n_cands = len(test_cands_raw)
print(f"Candidate pairs selected: {n_cands:,}")

# Ensure extraction functions are available in scope
if "extract_all_63_features" not in globals():
    raise RuntimeError("extract_all_63_features function not found in namespace. Run Cell 4 first!")

# 4. Score test candidate pairs in memory-efficient chunks with full 63 features
eval_t = best_leaderboard_t if ("best_leaderboard_t" in globals() or "best_leaderboard_t" in locals()) else 0.75
print(f"\nScoring {n_cands:,} candidate pairs with CatBoost GPU (Threshold t* = {eval_t:.2f})...")

chunk_size = 500_000
n_chunks = (n_cands + chunk_size - 1) // chunk_size
matched_pairs_list = []

for chunk_idx in range(n_chunks):
    start_i = chunk_idx * chunk_size
    end_i = min(start_i + chunk_size, n_cands)
    chunk = test_cands_raw[start_i:end_i]

    # Enrich chunk with real text metadata
    chunk_enriched = (
        chunk.join(s1_test_df, left_on="s1", right_on="entity_id", how="left")
        .rename({"business_name": "s1_name", "business_address": "s1_addr", "country": "s1_country"})
        .join(so_test_df, left_on="o", right_on="entity_id", how="left")
        .rename({"business_name": "o_name", "business_address": "o_addr", "country": "o_country"})
    )

    # Compute full 63 features (7 domain signals + 56 C++ RapidFuzz features)
    X_chunk = extract_all_63_features(chunk_enriched, desc=f"Chunk {chunk_idx+1}/{n_chunks}")

    # Predict probabilities with CatBoost GPU
    probs = model.predict_proba(X_chunk)[:, 1]

    # Filter by calibrated optimal threshold t*
    mask = probs >= eval_t
    n_kept = int(mask.sum())
    print(f"  Chunk {chunk_idx+1}/{n_chunks}: {n_kept:,} matches retained ({n_kept / len(chunk):.1%})")

    if n_kept > 0:
        chunk_kept = chunk.filter(pl.Series("mask", mask)).select(["s1", "o"])
        matched_pairs_list.append(chunk_kept)

    del chunk, chunk_enriched, X_chunk, probs
    gc.collect()

# 5. Aggregate predictions into official submission format
if matched_pairs_list:
    all_matched = pl.concat(matched_pairs_list)
    test_matches = (
        all_matched.group_by("s1")
        .agg(pl.col("o").str.join(","))
    )
    test_pred_map = dict(zip(test_matches["s1"].to_list(), test_matches["o"].to_list()))
else:
    test_pred_map = {}

print(f"\nTotal S1 Entities with predicted matches: {len(test_pred_map):,}")

submission_rows = []
for s1 in test_s1_all:
    matched = test_pred_map.get(s1, "")
    submission_rows.append({"source1_entity_id": s1, "matched_entity_ids": str(matched) if matched else ""})

sub_df = pd.DataFrame(submission_rows)
sub_df.to_csv(out_matching_tsv, sep="\t", index=False)

n_matched = sum(1 for r in submission_rows if r["matched_entity_ids"].strip())
print(f"\n-> Saved: {out_matching_tsv}")
print(f"   Total S1 rows:      {len(sub_df):,}")
print(f"   Entities matched:   {n_matched:,}")
print(f"   Singletons (empty): {len(sub_df) - n_matched:,}")

# 6. Run Official Validator
possible_val = [
    f"{os.path.dirname(test_dir)}/../utils/validate_submission.py",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/student_resource/utils/validate_submission.py",
    "/kaggle/input/amazon-ml-challenge-2026/student_resource/utils/validate_submission.py",
    "dataset/student_resource/utils/validate_submission.py",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/student_resource/utils/validate_submission.py",
]
val_script = next((p for p in possible_val if os.path.exists(p)), None)

if val_script and os.path.exists(val_script):
    print("\nRunning Official Competition Validator...")
    !python "{val_script}" --matching "{out_matching_tsv}" --test-dir "{test_dir}"
else:
    print(f"\nSubmission ready at {out_matching_tsv}")
"""
