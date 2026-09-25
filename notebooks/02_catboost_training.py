"""
notebook_02_catboost_training.py
==================================
Train CatBoost / LightGBM on 56 pairwise features.
This is your FAST baseline — run this first before DeBERTa.

Expected runtime on Colab CPU: ~30-60 min
Expected val F0.5: 0.70-0.82 (before cross-encoder)
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
print("CWD:", os.getcwd())

!pip install -q scikit-learn pandas numpy faiss-cpu lightgbm catboost rapidfuzz jellyfish tqdm scipy
"""

# ════════════════════════════════════════════════════════════
# CELL 2 — Run CatBoost pipeline (no GPU needed)
# ════════════════════════════════════════════════════════════
"""
!python pipeline_v2.py --mode train-catboost \
    --val-frac 0.05 \
    --max-train-samples 50000 \
    --top-k-name 30 \
    --top-k-addr 40 \
    --sn-window 10 \
    --max-candidates 150 \
    --hard-neg-ratio 3 \
    --n-estimators 2000 \
    --lr 0.03
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — (Optional) With BGE-M3 embedding features
# ════════════════════════════════════════════════════════════
"""
!pip install -q FlagEmbedding sentence-transformers

!python pipeline_v2.py --mode train-catboost \
    --val-frac 0.05 \
    --max-train-samples 50000 \
    --top-k-name 30 \
    --top-k-addr 40 \
    --sn-window 10 \
    --use-dense \
    --top-k-dense 30 \
    --use-embeddings \
    --hard-neg-ratio 3 \
    --n-estimators 2000 \
    --lr 0.03 \
    --device cuda
"""

# ════════════════════════════════════════════════════════════
# CELL 4 — Interactive: check feature importance
# ════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from catboost import CatBoostClassifier
from src.matching.features_v2 import FEATURE_NAMES_V2

model = CatBoostClassifier()
model.load_model("outputs/catboost_model.cbm")

importances = model.get_feature_importance()
feat_df = pd.DataFrame({
    'feature': FEATURE_NAMES_V2,
    'importance': importances
}).sort_values('importance', ascending=False)

print("Top 20 features:")
print(feat_df.head(20).to_string(index=False))
"""

# ════════════════════════════════════════════════════════════
# CELL 5 — Analyze errors on validation set
# ════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, '.')

from catboost import CatBoostClassifier
from src.matching.features_v2 import FEATURE_NAMES_V2, build_pair_feature_matrix_v2
from src.evaluation.evaluate_blocking import parse_ground_truth
from src.blocking.blocking_deterministic import run_deterministic_blocking
from src.blocking.blocking_tfidf import run_tfidf_blocking
from src.blocking.candidate_union import union_candidates

TRAIN_DIR = "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train"
s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\\t")
s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\\t")
s3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\\t")
gt = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\\t")
gt_dict = parse_ground_truth(gt)

# Sample val
s1_val = s1.sample(500, random_state=42)
s_all = pd.concat([s2, s3], ignore_index=True)

# Quick blocking
det = union_candidates(
    run_deterministic_blocking(s1_val, s2),
    run_tfidf_blocking(s1_val, s2, top_k_name=30, top_k_nameaddr=40),
)

X, y, pair_ids = build_pair_feature_matrix_v2(
    s1_val, s_all,
    {k: list(v) for k, v in det.items()},
    labels={k: gt_dict.get(k, set()) for k in det},
)

model = CatBoostClassifier()
model.load_model("outputs/catboost_model.cbm")
probs = model.predict_proba(X)[:, 1]

# Find false negatives (missed matches)
df_analysis = pd.DataFrame({
    's1_eid': [p[0] for p in pair_ids],
    'cand_eid': [p[1] for p in pair_ids],
    'prob': probs,
    'label': y,
})

fn = df_analysis[(df_analysis['label'] == 1) & (df_analysis['prob'] < 0.5)]
fp = df_analysis[(df_analysis['label'] == 0) & (df_analysis['prob'] >= 0.5)]

print(f"False Negatives (missed matches): {len(fn)}")
print(f"False Positives (wrong matches):  {len(fp)}")
print("\\nTop false negatives (high-confidence misses):")

s1_map = {r['entity_id']: r.to_dict() for _, r in s1_val.iterrows()}
s_map = {r['entity_id']: r.to_dict() for _, r in s_all.iterrows()}

for _, row in fn.nlargest(5, 'prob').iterrows():
    s1r = s1_map.get(row['s1_eid'], {})
    s2r = s_map.get(row['cand_eid'], {})
    print(f"  prob={row['prob']:.3f}")
    print(f"  S1: {s1r.get('business_name','')} | {s1r.get('business_address','')}")
    print(f"  S2: {s2r.get('business_name','')} | {s2r.get('business_address','')}")
    print()
"""

# ════════════════════════════════════════════════════════════
# CELL 6 — Download submission files
# ════════════════════════════════════════════════════════════
"""
from google.colab import files

# Download both submission files
files.download("outputs/matching_results.tsv")
files.download("outputs/candidate_pairs.tsv")
"""
