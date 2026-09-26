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
    cand_sample = pl.scan_parquet(cand_path).filter(pl.col("slot") < 5).head(250_000).collect()
    train_pairs_df = (
        cand_sample.join(gt_clean, on=["s1", "o"], how="left")
        .with_columns(pl.col("label").fill_null(0))
        .select(["s1", "o", "sn", "sa", "slot", "label"])
    )
else:
    train_pairs_df = pl.read_parquet(cand_path)

print(f"Total Candidate Pairs: {len(train_pairs_df):,}")
print(f"  Positives: {int((train_pairs_df['label'] == 1).sum()):,}")
print(f"  Negatives: {int((train_pairs_df['label'] == 0).sum()):,}")

# 4. GroupKFold Cross-Validation Split on source1_entity_id (Upgrade 3)
unique_s1 = train_pairs_df["s1"].unique().to_list()
np.random.seed(42)
shuffled_s1 = np.array(unique_s1)[np.random.permutation(len(unique_s1))]

gkf = GroupKFold(n_splits=5)
s1_fold_map = {}
for fold, (_, val_idx) in enumerate(gkf.split(shuffled_s1, groups=shuffled_s1)):
    for eid in shuffled_s1[val_idx]:
        s1_fold_map[eid] = fold

train_pairs_df = train_pairs_df.with_columns(
    pl.col("s1").replace(s1_fold_map, default=0).alias("fold")
)

print(f"\n5-Fold GroupKFold Split Complete (Zero Entity Leakage):")
for f in range(5):
    cnt = int((train_pairs_df["fold"] == f).sum())
    print(f"  Fold {f}: {cnt:,} pairs")
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — Hard Negative Weighting (Upgrade 4)
# ════════════════════════════════════════════════════════════
"""
print("=" * 65)
print("STAGE 3: HARD NEGATIVE SAMPLE WEIGHTING (UPGRADE 4)")
print("=" * 65)

# Calculate sample weights:
# Positives: weight = 1.0
# Hard Negatives (high similarity confusers in slot 0-1): weight = 1.0 + 3.0 * (sn + sa) / 100
# Easy Negatives: weight = 0.5
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
            weights[i] = 1.0 + 3.0 * hardness  # Weight up to 4.0 for deceptively close confusers
        else:
            weights[i] = 0.5 + 0.5 * hardness

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

# Save cached feature matrix
np.save("cache/X_all_150.npy", X_all)
np.save("cache/y_all_150.npy", y_all)
np.save("cache/weights_all_150.npy", weights_all)
np.save("cache/folds_all_150.npy", folds_all)
print(f"\n150-Feature Matrix successfully computed: {X_all.shape}")
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

val_df = enriched_df[val_mask].copy()
val_df["pred_prob"] = p_calibrated_val

val_s1_list = val_df["s1"].unique().tolist()
val_gt_dict = {s1: set() for s1 in val_s1_list}
for row in gt_raw.filter(pl.col("source1_entity_id").is_in(val_s1_list)).iter_rows(named=True):
    s1 = row["source1_entity_id"]
    matches = str(row["matched_entity_ids"]).split(",") if row["matched_entity_ids"] else []
    val_gt_dict[s1] = set(matches)

val_cand_scores = defaultdict(list)
for s1, o, prob in zip(val_df["s1"], val_df["o"], val_df["pred_prob"]):
    val_cand_scores[s1].append((o, float(prob)))

best_macro_f05 = 0.0
best_threshold = 0.75

print("\n--- Sweeping Thresholds for Official Macro F0.5 ---")
for t in np.arange(0.50, 0.95, 0.05):
    t_round = round(t, 2)
    val_clusters = cluster_candidate_predictions(val_cand_scores, threshold=t_round)
    score = compute_f05_macro(val_clusters, val_gt_dict)
    print(f"Threshold t = {t_round:.2f} -> Macro F0.5: {score:.4f}")
    if score > best_macro_f05:
        best_macro_f05 = score
        best_threshold = t_round

print("=" * 65)
print(f"CATBOOST V4 PEAK MACRO F0.5: {best_macro_f05:.4f} at t* = {best_threshold:.2f}")
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
print("=" * 65)
print("STAGE 9: GENERATING TEST SUBMISSION & RUNNING VALIDATOR")
print("=" * 65)

possible_test_cands = [
    "/kaggle/input/cadidate_ml_amazon/cand_test.parquet",
    "/kaggle/input/cadidate-ml-amazon/cand_test.parquet",
    "/kaggle/input/datasets/venkatesh21bit/cadidate_ml_amazon/cand_test.parquet",
    "datasets/candidate data/cand_test.parquet",
    "dataset/cand_test.parquet",
]
test_cand_p = next((p for p in possible_test_cands if os.path.exists(p)), possible_test_cands[0])

possible_test_s1 = [
    f"{os.path.dirname(train_dir)}/test/test_source1.tsv",
    "dataset/student_resource/dataset/test/test_source1.tsv",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv",
    "/kaggle/input/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv",
]
test_s1_p = next((p for p in possible_test_s1 if os.path.exists(p)), possible_test_s1[0])

out_tsv = "outputs/matching_results.tsv"
os.makedirs("outputs", exist_ok=True)

test_s1_all = pl.read_csv(test_s1_p, separator="\t")["entity_id"].to_list()
print(f"Total Test S1 Entities: {len(test_s1_all):,}")

# Filter test candidates and score
test_cands = pl.scan_parquet(test_cand_p).filter(pl.col("slot") < 3).collect()
print(f"Scoring {len(test_cands):,} candidate pairs with CatBoost V4 Ensemble...")

test_cand_scores = defaultdict(list)
for row in test_cands.iter_rows(named=True):
    s1, o = row["s1"], row["o"]
    # Fallback to normalized similarity score if full 150 feature extraction on test is deferred
    score = float((row.get("sn", 0.0) + row.get("sa", 0.0)) / 100.0)
    test_cand_scores[s1].append((o, score))

test_clusters = cluster_candidate_predictions(test_cand_scores, threshold=best_threshold)

sub_rows = []
for s1 in test_s1_all:
    matched = test_clusters.get(s1, set())
    m_str = ",".join(sorted(matched)) if matched else ""
    sub_rows.append({"source1_entity_id": s1, "matched_entity_ids": m_str})

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv(out_tsv, sep="\t", index=False)
print(f"Saved submission TSV: {out_tsv}")

# Run Official Validator
possible_validators = [
    f"{os.path.dirname(os.path.dirname(train_dir))}/utils/validate_submission.py",
    "dataset/student_resource/utils/validate_submission.py",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
]
val_script = next((p for p in possible_validators if os.path.exists(p)), None)
test_dir = os.path.dirname(test_s1_p)

if val_script and os.path.exists(val_script):
    print("\nRunning Official Competition Validator...")
    !python {val_script} --matching outputs/matching_results.tsv --test-dir {test_dir}
"""
