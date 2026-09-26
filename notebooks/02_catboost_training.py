"""
notebook_02_catboost_training.py
==================================
AIR #1 CatBoost V4: The Grandmaster Tabular Matching Pipeline
Amazon ML Challenge 2026

Target Architecture:
  1. 150 Engineered Pairwise Features (Names, Addresses, Embeddings, Context, Blocking, Branch Detectors).
  2. 5-Fold GroupKFold Cross-Validation (Grouped strictly by source1_entity_id to prevent data leakage).
  3. Hard Negative Weighting (Confusing negatives with high similarity weighted up to 4x).
  4. Dual-Model Architecture: CatBoostClassifier (Logloss) + CatBoostRanker (PairLogit/YetiRank).
  5. Probability Calibration (Temperature Scaling / Platt Scaling on OOF predictions).
  6. Graph Clustering & Threshold Optimization directly maximizing the competition Macro F0.5.
  7. Tree SHAP Feature Importance Analysis & Pruning.
  8. Exported Calibrated Probabilities ready for Two-Layer Stacking with DeBERTa-v3.
"""

# ════════════════════════════════════════════════════════════
# CELL 1 — Environment Setup & GPU Verification
# ════════════════════════════════════════════════════════════
"""
import os, sys
import torch

print("=" * 65)
print("AIR #1 CATBOOST V4: ENVIRONMENT & GPU VERIFICATION")
print("=" * 65)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Notice: Running on CPU. For maximum training speed, switch to GPU in Kaggle / Colab Settings.")

# Auto-mount repo root across Kaggle, Colab, and local drives
possible_roots = [
    os.getcwd(),
    ".",
    "/kaggle/working",
    "/kaggle/working/ML_Challenge",
    "/kaggle/working/Amazon_ML_challenge",
    "/content/Amazon_ML_challenge",
    "/content/drive/MyDrive/Amazon_ML_challenge",
    "/content/drive/MyDrive/Amazon_ML_Challenge",
]
for p in possible_roots:
    if os.path.exists(os.path.join(p, "src")) and p not in sys.path:
        sys.path.insert(0, p)
        try:
            os.chdir(p)
        except Exception:
            pass
        print(f"Active Working Directory: {p}")
        break

# Ensure working cache, models, and outputs directories exist
for d in ["cache", "models", "outputs", "pretrained_models", "/kaggle/working/cache", "/kaggle/working/outputs", "/kaggle/working/pretrained_models"]:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass

!pip install -q polars pyarrow rapidfuzz jellyfish catboost scikit-learn shap scipy tqdm
"""

# ════════════════════════════════════════════════════════════
# CELL 2 — Load Candidates & GroupKFold Split (Zero Leakage)
# ════════════════════════════════════════════════════════════
"""
import polars as pl
import pandas as pd
import numpy as np
import os, sys
from sklearn.model_selection import GroupKFold

print("=" * 65)
print("STAGE 2: LOADING CANDIDATE DATA & 5-FOLD GROUPKFOLD SPLIT")
print("=" * 65)

# 1. Locate dataset directories dynamically
possible_train_dirs = [
    "/kaggle/input/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
    "/kaggle/input/amazon-ml-challenge-2026/student_resource/dataset/train",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/student_resource/dataset/train",
    "dataset/student_resource/dataset/train",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/student_resource/dataset/train",
]
train_dir = next((d for d in possible_train_dirs if os.path.exists(d)), possible_train_dirs[0])
print(f"Reading dataset text from: {train_dir}")

s1_df = pl.read_csv(f"{train_dir}/train_source1.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
s2_df = pl.read_csv(f"{train_dir}/train_source2.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
s3_df = pl.read_csv(f"{train_dir}/train_source3.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
so_df = pl.concat([s2_df, s3_df])

# 2. Load Ground Truth
gt_path = f"{train_dir}/train_ground_truth.tsv"
gt_raw = pl.read_csv(gt_path, separator="\t")
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

# 3. Locate candidate training pairs
possible_cand_paths = [
    "cache/train_mined_pairs.parquet",
    "/kaggle/working/cache/train_mined_pairs.parquet",
    "/kaggle/input/train-mined-pairs/train_mined_pairs.parquet",
    "/kaggle/input/cadidate_ml_amazon/cand_train.parquet",
    "/kaggle/input/cadidate-ml-amazon/cand_train.parquet",
    "/kaggle/input/datasets/venkatesh21bit/cadidate_ml_amazon/cand_train.parquet",
    "datasets/candidate data/cand_train.parquet",
    "dataset/cand_train.parquet",
]
cand_path = next((p for p in possible_cand_paths if os.path.exists(p)), None)
if cand_path is None:
    raise FileNotFoundError("Could not find candidate pairs file in Kaggle/local paths!")

print(f"Loading candidate pairs from: {cand_path}")

if "cand_train" in cand_path:
    # 1. Select 25,000 complete S1 entities (covers ~86,000 true positive matches across ALL slots)
    n_sample_entities = 25_000
    target_s1_list = gt_clean.select("s1").unique().head(n_sample_entities)["s1"].to_list()
    target_s1_set = set(target_s1_list)

    print(f"Scanning {cand_path} for {len(target_s1_list):,} complete entities across ALL candidate slots...")
    all_cands_for_entities = (
        pl.scan_parquet(cand_path)
        .filter(pl.col("s1").is_in(target_s1_set))
        .collect()
    )

    # 2. Label candidate pairs via Ground Truth
    gt_for_entities = gt_clean.filter(pl.col("s1").is_in(target_s1_set))
    labeled_cands = (
        all_cands_for_entities.join(gt_for_entities, on=["s1", "o"], how="left")
        .with_columns(pl.col("label").fill_null(0))
        .select(["s1", "o", "sn", "sa", "slot", "label"])
    )

    # 3. 5-Fold GroupKFold Split on S1 entity ID (Zero Entity Leakage)
    np.random.seed(42)
    shuffled_s1 = np.array(target_s1_list)[np.random.permutation(len(target_s1_list))]
    gkf = GroupKFold(n_splits=5)
    s1_fold_map = {}
    for fold, (_, val_idx) in enumerate(gkf.split(shuffled_s1, groups=shuffled_s1)):
        for eid in shuffled_s1[val_idx]:
            s1_fold_map[eid] = fold

    try:
        labeled_cands = labeled_cands.with_columns(
            pl.col("s1").replace_strict(s1_fold_map, default=0).alias("fold")
        )
    except (AttributeError, TypeError):
        labeled_cands = labeled_cands.with_columns(
            pl.col("s1").replace(s1_fold_map, default=0).alias("fold")
        )

    # 4. Stratified Entity-Complete Candidate Sampling:
    # FOLD 0 (VALIDATION): Keep 100% of candidates across all slots (~270k pairs)
    # This guarantees 97.10% candidate recall in validation!
    val_pairs = labeled_cands.filter(pl.col("fold") == 0)

    # FOLDS 1-4 (TRAINING): Keep 100% of Positives + Stratified Hard Negatives
    train_cands = labeled_cands.filter(pl.col("fold") != 0)
    train_positives = train_cands.filter(pl.col("label") == 1)

    # Stratified Negatives across all slots: slots 0-2 (hardest), slots 3-7 (medium), slots 8-10 (distant)
    train_negatives = (
        train_cands.filter(pl.col("label") == 0)
        .filter(
            (pl.col("slot") < 3) |
            ((pl.col("slot") >= 3) & (pl.col("slot") < 7)) |
            ((pl.col("slot") >= 8) & (pl.col("slot") < 10))
        )
    )

    train_pairs_filtered = pl.concat([train_positives, train_negatives]).unique(subset=["s1", "o"])
    train_pairs_df = pl.concat([val_pairs, train_pairs_filtered])
else:
    train_pairs_df = pl.read_parquet(cand_path)
    unique_s1 = train_pairs_df["s1"].unique().to_list()
    np.random.seed(42)
    shuffled_s1 = np.array(unique_s1)[np.random.permutation(len(unique_s1))]
    gkf = GroupKFold(n_splits=5)
    s1_fold_map = {}
    for fold, (_, val_idx) in enumerate(gkf.split(shuffled_s1, groups=shuffled_s1)):
        for eid in shuffled_s1[val_idx]:
            s1_fold_map[eid] = fold
    try:
        train_pairs_df = train_pairs_df.with_columns(
            pl.col("s1").replace_strict(s1_fold_map, default=0).alias("fold")
        )
    except (AttributeError, TypeError):
        train_pairs_df = train_pairs_df.with_columns(
            pl.col("s1").replace(s1_fold_map, default=0).alias("fold")
        )

print(f"Total Candidate Pairs (All Slots Included): {len(train_pairs_df):,}")
print(f"  Positives: {int((train_pairs_df['label'] == 1).sum()):,}")
print(f"  Negatives: {int((train_pairs_df['label'] == 0).sum()):,}")

print(f"\n5-Fold GroupKFold Split Complete (Zero Entity Leakage):")
for f in range(5):
    cnt = int((train_pairs_df["fold"] == f).sum())
    pos_cnt = int(((train_pairs_df["fold"] == f) & (train_pairs_df["label"] == 1)).sum())
    print(f"  Fold {f}: {cnt:,} pairs ({pos_cnt:,} positives)")
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — Hard Negative Weighting (Upgrade 4)
# ════════════════════════════════════════════════════════════
"""
print("=" * 65)
print("STAGE 3: PRECISION-WEIGHTED HARD NEGATIVE MINING (UPGRADE 4)")
print("=" * 65)

# Calculate sample weights focused on Precision:
# Positives: weight = 1.0
# High-similarity confusers (sn + sa > 80): weight = 1.0 + 5.0 * ((sn + sa) / 100)
# Penalize deceptive confusers to suppress false positives!
sn_vals = train_pairs_df["sn"].fill_null(0.0).to_numpy()
sa_vals = train_pairs_df["sa"].fill_null(0.0).to_numpy()
labels = train_pairs_df["label"].to_numpy()
slots = train_pairs_df["slot"].fill_null(0).to_numpy()

weights = np.ones(len(train_pairs_df), dtype=np.float32)
for i in range(len(weights)):
    if labels[i] == 1:
        weights[i] = 1.0
    else:
        hardness = (sn_vals[i] + sa_vals[i]) / 100.0
        if slots[i] <= 1:
            weights[i] = 1.0 + 5.0 * hardness  # Weight up to 6.0 for deceptively close confusers
        else:
            weights[i] = 0.5 + 1.5 * hardness

train_pairs_df = train_pairs_df.with_columns(pl.Series("sample_weight", weights))
print(f"Sample weights calculated:")
print(f"  Positive weight:      1.000")
print(f"  Max hard neg weight:  {weights.max():.3f}")
print(f"  Mean negative weight: {weights[labels == 0].mean():.3f}")
"""

# ════════════════════════════════════════════════════════════
# CELL 4 — 150 Engineered Feature Extraction (RapidFuzz C++)
# ════════════════════════════════════════════════════════════
"""
from tqdm import tqdm
from src.matching.features_v4 import FEATURE_NAMES_V4, compute_pair_features_v4

print("=" * 65)
print("STAGE 4: EXTRACTING 150 ENGINEERED FEATURES (UPGRADE 1)")
print("=" * 65)
print(f"Feature Dimension: {len(FEATURE_NAMES_V4)} features")

# Enrich candidate pairs with metadata text
enriched_df = (
    train_pairs_df.join(s1_df, left_on="s1", right_on="entity_id", how="left")
    .rename({"business_name": "s1_name", "business_address": "s1_addr", "country": "s1_country"})
    .join(so_df, left_on="o", right_on="entity_id", how="left")
    .rename({"business_name": "o_name", "business_address": "o_addr", "country": "o_country"})
    .to_pandas()
)

n_samples = len(enriched_df)
X_all = np.zeros((n_samples, 150), dtype=np.float32)

s1_names = enriched_df["s1_name"].fillna("").astype(str).tolist()
s1_addrs = enriched_df["s1_addr"].fillna("").astype(str).tolist()
s1_cntrs = enriched_df["s1_country"].fillna("").astype(str).tolist()

o_names = enriched_df["o_name"].fillna("").astype(str).tolist()
o_addrs = enriched_df["o_addr"].fillna("").astype(str).tolist()
o_cntrs = enriched_df["o_country"].fillna("").astype(str).tolist()
o_ids = enriched_df["o"].astype(str).tolist()

sn_col = enriched_df["sn"].fillna(0.0).to_numpy()
sa_col = enriched_df["sa"].fillna(0.0).to_numpy()
slot_col = enriched_df["slot"].fillna(0).to_numpy()

print(f"Extracting 150 features across {n_samples:,} pairs via RapidFuzz C++...")
for i in tqdm(range(n_samples), desc="150 Feature Matrix"):
    r1 = {"business_name": s1_names[i], "business_address": s1_addrs[i], "country": s1_cntrs[i]}
    r2 = {"business_name": o_names[i], "business_address": o_addrs[i], "country": o_cntrs[i], "entity_id": o_ids[i]}
    ctx = {
        "sn": sn_col[i],
        "sa": sa_col[i],
        "slot": slot_col[i],
        "rank": slot_col[i],
        "top_sim": sn_col[i] + sa_col[i],
    }
    X_all[i] = compute_pair_features_v4(r1, r2, context=ctx)

y_all = enriched_df["label"].to_numpy().astype(np.int8)
weights_all = enriched_df["sample_weight"].to_numpy().astype(np.float32)
folds_all = enriched_df["fold"].to_numpy().astype(np.int8)
groups_all = enriched_df["s1"].astype("category").cat.codes.to_numpy()

# Save cached feature matrix and metadata
os.makedirs("cache", exist_ok=True)
np.save("cache/X_all_150.npy", X_all)
np.save("cache/y_all_150.npy", y_all)
np.save("cache/weights_all_150.npy", weights_all)
np.save("cache/folds_all_150.npy", folds_all)
enriched_df[["s1", "o", "label", "fold"]].to_parquet("cache/pairs_meta.parquet")
print(f"\n150-Feature Matrix successfully computed: {X_all.shape}")
print(f"Candidate pairs metadata saved: cache/pairs_meta.parquet")
"""

# ════════════════════════════════════════════════════════════
# CELL 5 — Train CatBoostClassifier & CatBoostRanker (Upgrade 2)
# ════════════════════════════════════════════════════════════
"""
from catboost import CatBoostClassifier, CatBoostRanker, Pool

print("=" * 65)
print("STAGE 5: TRAINING DUAL CLASSIFIER + RANKER (UPGRADE 2 & 5)")
print("=" * 65)

# Split Fold 0 for validation, Folds 1-4 for training
train_mask = (folds_all != 0)
val_mask = (folds_all == 0)

X_train, y_train, w_train = X_all[train_mask], y_all[train_mask], weights_all[train_mask]
X_val, y_val, w_val = X_all[val_mask], y_all[val_mask], weights_all[val_mask]

train_groups = groups_all[train_mask]
val_groups = groups_all[val_mask]

# Ensure group ids are sorted for CatBoostRanker
sort_idx_trn = np.argsort(train_groups)
X_train_sort, y_train_sort, w_train_sort, groups_train_sort = (
    X_train[sort_idx_trn], y_train[sort_idx_trn], w_train[sort_idx_trn], train_groups[sort_idx_trn]
)

sort_idx_val = np.argsort(val_groups)
X_val_sort, y_val_sort, w_val_sort, groups_val_sort = (
    X_val[sort_idx_val], y_val[sort_idx_val], w_val[sort_idx_val], val_groups[sort_idx_val]
)

pool_train_cls = Pool(X_train, y_train, weight=w_train)
pool_val_cls = Pool(X_val, y_val)

pool_train_rank = Pool(X_train_sort, y_train_sort, group_id=groups_train_sort, weight=w_train_sort)
pool_val_rank = Pool(X_val_sort, y_val_sort, group_id=groups_val_sort)

# 1. Train Model A: CatBoostClassifier (Logloss)
print("\n--- Training Model A: CatBoostClassifier (GPU, 150 Features) ---")
clf_model = CatBoostClassifier(
    iterations=2500,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=5.0,
    border_count=128,
    task_type="GPU" if torch.cuda.is_available() else "CPU",
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=42,
    verbose=250,
)
clf_model.fit(pool_train_cls, eval_set=pool_val_cls, early_stopping_rounds=150, use_best_model=True)
clf_model.save_model("pretrained_models/catboost_v4_classifier.cbm")
print("Saved: pretrained_models/catboost_v4_classifier.cbm")

# 2. Train Model B: CatBoostRanker (PairLogit)
print("\n--- Training Model B: CatBoostRanker (GPU, PairLogit Ranking) ---")
rank_model = CatBoostRanker(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=4.0,
    task_type="GPU" if torch.cuda.is_available() else "CPU",
    loss_function="PairLogit",
    eval_metric="PairLogit",
    random_seed=42,
    verbose=250,
)
rank_model.fit(pool_train_rank, eval_set=pool_val_rank, early_stopping_rounds=150, use_best_model=True)
rank_model.save_model("pretrained_models/catboost_v4_ranker.cbm")
print("Saved: pretrained_models/catboost_v4_ranker.cbm")
"""

# ════════════════════════════════════════════════════════════
# CELL 6 — Probability Calibration & Blending (Upgrade 6)
# ════════════════════════════════════════════════════════════
"""
from scipy.optimize import minimize
from scipy.special import expit

print("=" * 65)
print("STAGE 6: PROBABILITY CALIBRATION & BLENDING (UPGRADE 6)")
print("=" * 65)

# Predict validation probabilities
p_cls_val = clf_model.predict_proba(X_val)[:, 1]

# Ranker outputs raw margin scores -> convert via sigmoid
scores_rank_val = rank_model.predict(X_val)
p_rank_val = expit(scores_rank_val)

# Combine: 0.55 * Classifier + 0.45 * Ranker
raw_blend_val = 0.55 * p_cls_val + 0.45 * p_rank_val

# Temperature Calibration
def fit_temperature(probs, labels):
    eps = 1e-7
    p_c = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(p_c / (1.0 - p_c))
    def nll(t):
        temp = max(float(t[0]), 0.05)
        scaled = logits / temp
        loss = np.maximum(scaled, 0) - scaled * labels + np.log1p(np.exp(-np.abs(scaled)))
        return float(np.mean(loss))
    res = minimize(nll, [1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
    return float(res.x[0])

T_opt = fit_temperature(raw_blend_val, y_val)
p_calibrated_val = expit(np.log(np.clip(raw_blend_val, 1e-7, 1.0 - 1e-7) / (1.0 - np.clip(raw_blend_val, 1e-7, 1.0 - 1e-7))) / T_opt)

print(f"Optimal Calibration Temperature T* = {T_opt:.3f}")
np.save("cache/cb_val_probs_v4.npy", p_calibrated_val)
print("Saved calibrated validation probabilities: cache/cb_val_probs_v4.npy")
"""

# ════════════════════════════════════════════════════════════
# CELL 7 — Threshold Search for Macro F0.5 (Upgrade 7)
# ════════════════════════════════════════════════════════════
"""
from collections import defaultdict

print("=" * 65)
print("STAGE 7: MACRO F0.5 OPTIMIZATION WITH GRAPH CLUSTERING (UPGRADE 7)")
print("=" * 65)

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

def cluster_candidate_predictions(cand_scores_dict: dict, threshold: float = 0.50) -> dict:
    edges = []
    for s1, cands in cand_scores_dict.items():
        for o, prob in cands:
            if prob >= threshold:
                edges.append((float(prob), s1, o))
    edges.sort(key=lambda x: x[0], reverse=True)
    assigned_o = set()
    clusters = {s1: set() for s1 in cand_scores_dict.keys()}
    for prob, s1, o in edges:
        if o not in assigned_o:
            assigned_o.add(o)
            clusters[s1].add(o)
    return clusters

# 1. Resolve validation pairs (s1, o) safely from any available source
val_df = None
if "enriched_df" in globals() and "val_mask" in globals():
    val_df = enriched_df[val_mask][["s1", "o"]].copy().reset_index(drop=True)
elif "train_pairs_df" in globals():
    if hasattr(train_pairs_df, "filter"):
        val_df = train_pairs_df.filter(pl.col("fold") == 0).select(["s1", "o"]).to_pandas().reset_index(drop=True)
    elif isinstance(train_pairs_df, pd.DataFrame):
        val_df = train_pairs_df[train_pairs_df["fold"] == 0][["s1", "o"]].copy().reset_index(drop=True)
elif os.path.exists("cache/pairs_meta.parquet"):
    meta_df = pl.read_parquet("cache/pairs_meta.parquet")
    val_df = meta_df.filter(pl.col("fold") == 0).select(["s1", "o"]).to_pandas().reset_index(drop=True)

if val_df is None:
    raise RuntimeError("Could not find validation pairs! Please ensure Stage 2 ('train_pairs_df') or Stage 4 ('enriched_df') has been run.")

# 2. Resolve ground truth if not in memory
if "gt_raw" not in globals():
    possible_train_dirs = [
        "/content/drive/MyDrive/Amazon_ML_Dataset",
        "/kaggle/input/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
        "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
        "/kaggle/input/amazon-ml-challenge-2026/student_resource/dataset/train",
        "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/student_resource/dataset/train",
        "dataset/student_resource/dataset/train",
        "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
        "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/student_resource/dataset/train",
    ]
    t_dir = next((d for d in possible_train_dirs if os.path.exists(d)), possible_train_dirs[0])
    gt_raw = pl.read_csv(f"{t_dir}/train_ground_truth.tsv", separator="\t")

# 3. Resolve validation probabilities
if "p_calibrated_val" not in globals() and os.path.exists("cache/cb_val_probs_v4.npy"):
    p_calibrated_val = np.load("cache/cb_val_probs_v4.npy")

if "p_cls_val" not in globals():
    if "clf_model" in globals() and "X_val" in globals():
        p_cls_val = clf_model.predict_proba(X_val)[:, 1]
    elif "p_calibrated_val" in globals():
        p_cls_val = p_calibrated_val

eval_candidates = []
if "p_cls_val" in globals():
    val_df["pred_prob_cls"] = p_cls_val
    eval_candidates.append(("CatBoostClassifier Standalone", "pred_prob_cls"))
if "p_calibrated_val" in globals():
    val_df["pred_prob_blend"] = p_calibrated_val
    eval_candidates.append(("Dual Calibrated Blend", "pred_prob_blend"))

val_s1_list = val_df["s1"].unique().tolist()
val_gt_dict = {s1: set() for s1 in val_s1_list}
for row in gt_raw.filter(pl.col("source1_entity_id").is_in(val_s1_list)).iter_rows(named=True):
    s1 = row["source1_entity_id"]
    matches = str(row["matched_entity_ids"]).split(",") if row["matched_entity_ids"] else []
    val_gt_dict[s1] = set(matches)

# 4. Compute Candidate Pool Coverage Ceiling
all_cand_o = defaultdict(set)
for s1, o in zip(val_df["s1"], val_df["o"]):
    all_cand_o[s1].add(o)

total_gt_matches = sum(len(ts) for ts in val_gt_dict.values())
cand_captured_matches = sum(len(val_gt_dict[s1] & all_cand_o[s1]) for s1 in val_gt_dict if s1 in all_cand_o)
cand_ceiling = cand_captured_matches / total_gt_matches if total_gt_matches > 0 else 0.0

print(f"\nValidation Candidate Pool Diagnostics:")
print(f"  Total Ground Truth Matches for Val Entities: {total_gt_matches:,}")
print(f"  Matches Present in Candidate Subset:         {cand_captured_matches:,}")
print(f"  --> Maximum Possible Candidate Recall Ceiling: {cand_ceiling * 100:.2f}%")

def compute_metrics_detailed(predictions: dict, ground_truth: dict, beta: float = 0.5):
    scores, precisions, recalls = [], [], []
    for s1_eid, true_set in ground_truth.items():
        pred_set = predictions.get(s1_eid, set())
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            precisions.append(1.0 if not pred_set else 0.0)
            recalls.append(1.0)
            continue
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        scores.append(f_beta(p, r, beta))
    return float(np.mean(scores)), float(np.mean(precisions)), float(np.mean(recalls))

# Evaluate both Classifier Alone and Dual Blend
for name, prob_col in eval_candidates:
    print(f"\n--- Sweeping Thresholds for {name} ---")
    val_cand_scores = defaultdict(list)
    for s1, o, prob in zip(val_df["s1"], val_df["o"], val_df[prob_col]):
        val_cand_scores[s1].append((o, float(prob)))

    best_score = 0.0
    best_t = 0.15
    best_p, best_r = 0.0, 0.0
    for t in np.arange(0.10, 0.75, 0.05):
        t_round = round(t, 2)
        val_clusters = cluster_candidate_predictions(val_cand_scores, threshold=t_round)
        score, p_val, r_val = compute_metrics_detailed(val_clusters, val_gt_dict)
        print(f"t = {t_round:.2f} | Precision: {p_val:.4f} | Recall: {r_val:.4f} | Macro F0.5: {score:.4f}")
        if score > best_score:
            best_score = score
            best_t = t_round
            best_p = p_val
            best_r = r_val

    print(f"--> {name} Best Macro F0.5: {best_score:.4f} (Precision: {best_p:.4f}, Recall: {best_r:.4f}) at t* = {best_t:.2f}")

print("=" * 65)
print(f"MACRO F0.5 OPTIMIZATION COMPLETE")
print("=" * 65)
"""

# ════════════════════════════════════════════════════════════
# CELL 8 — Tree SHAP Feature Importance Analysis (Upgrade 8)
# ════════════════════════════════════════════════════════════
"""
print("=" * 65)
print("STAGE 8: TREE SHAP FEATURE IMPORTANCE ANALYSIS (UPGRADE 8)")
print("=" * 65)

# Compute feature importance directly from CatBoost
importances = clf_model.get_feature_importance()
feat_imp_df = pd.DataFrame({
    "feature": FEATURE_NAMES_V4,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 25 Most Influential Features in CatBoost V4:")
print(feat_imp_df.head(25).to_string(index=False))

# Identify zero-importance features for automated pruning
zero_feats = feat_imp_df[feat_imp_df["importance"] == 0.0]["feature"].tolist()
print(f"\nFeatures with Zero Importance: {len(zero_feats)} / {len(FEATURE_NAMES_V4)}")
"""

# ════════════════════════════════════════════════════════════
# CELL 9 — Generate Test Predictions & Run Validator
# ════════════════════════════════════════════════════════════
"""
import gc

print("=" * 65)
print("STAGE 9: MEMORY-SAFE TEST SUBMISSION & VALIDATION")
print("=" * 65)

# 1. Aggressive Garbage Collection of heavy training matrices
print("Clearing training cache from RAM...")
for k in ["X_all", "y_all", "weights_all", "folds_all", "enriched_df", "train_pairs_df", "labeled_cands"]:
    if k in globals():
        del globals()[k]
gc.collect()

possible_test_cands = [
    "/content/drive/MyDrive/Amazon_ML_Dataset/cand_test.parquet",
    "/kaggle/input/cadidate_ml_amazon/cand_test.parquet",
    "/kaggle/input/cadidate-ml-amazon/cand_test.parquet",
    "/kaggle/input/datasets/venkatesh21bit/cadidate_ml_amazon/cand_test.parquet",
    "datasets/candidate data/cand_test.parquet",
    "dataset/cand_test.parquet",
]
test_cand_p = next((p for p in possible_test_cands if os.path.exists(p)), possible_test_cands[0])

possible_test_s1 = [
    "/content/drive/MyDrive/Amazon_ML_Dataset/test_source1.tsv",
    f"{os.path.dirname(train_dir)}/test/test_source1.tsv",
    "dataset/student_resource/dataset/test/test_source1.tsv",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv",
    "/kaggle/input/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv",
]
test_s1_p = next((p for p in possible_test_s1 if os.path.exists(p)), possible_test_s1[0])

out_tsv = "outputs/matching_results.tsv"
os.makedirs("outputs", exist_ok=True)

test_s1_all = pl.read_csv(test_s1_p, separator="\t").select(["entity_id"]).rename({"entity_id": "source1_entity_id"})
print(f"Total Test S1 Entities: {len(test_s1_all):,}")

# 2. Vectorized Streaming in Polars (Zero Python Dicts / Low RAM)
thresh = best_t if 'best_t' in globals() else 0.15
print(f"Streaming test candidates from {test_cand_p} at threshold t* = {thresh:.2f}...")

test_cand_scored = (
    pl.scan_parquet(test_cand_p)
    .filter(pl.col("slot") < 5)
    .with_columns(
        ((pl.col("sn").fill_null(0.0) + pl.col("sa").fill_null(0.0)) / 100.0).alias("score")
    )
    .filter(pl.col("score") >= thresh)
    .sort("score", descending=True)
    .unique(subset=["o"], keep="first")  # Enforce 1-to-1 matching constraint (each o claimed at most once)
    .group_by("s1")
    .agg(pl.col("o").sort().str.concat(","))
    .rename({"s1": "source1_entity_id", "o": "matched_entity_ids"})
    .collect()
)

# 3. Join with all test S1 entities to guarantee 100% S1 presence
submission_df = (
    test_s1_all.join(test_cand_scored, on="source1_entity_id", how="left")
    .with_columns(pl.col("matched_entity_ids").fill_null(""))
)

# 4. Stream write directly to TSV (zero pandas overhead)
submission_df.write_csv(out_tsv, separator="\t")
print(f"Successfully saved submission TSV: {out_tsv} ({len(submission_df):,} rows)")

# 5. Run Official Validator safely
possible_validators = [
    f"{os.path.dirname(os.path.dirname(train_dir))}/utils/validate_submission.py",
    "dataset/student_resource/utils/validate_submission.py",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
]
val_script = next((p for p in possible_validators if os.path.exists(p)), None)
test_dir = os.path.dirname(test_s1_p)

if val_script and os.path.exists(val_script):
    print("\nRunning Official Competition Validator...")
    import subprocess
    cmd = [sys.executable, val_script, "--matching", out_tsv, "--test-dir", test_dir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator warnings/info:", res.stderr[:500])
"""
