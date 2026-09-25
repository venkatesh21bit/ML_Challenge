"""
notebook_01_blocking_benchmark.py
==================================
Run this on Google Colab to benchmark all blocking methods.

Copy-paste each section into a separate Colab cell.
Target: blocking recall >= 99% before proceeding to matching.
"""

# ════════════════════════════════════════════════════════════
# CELL 1 — Mount Drive + Install Dependencies
# ════════════════════════════════════════════════════════════
"""
from google.colab import drive
drive.mount('/content/drive')

# Clone/copy your repo to Colab
import os
REPO_PATH = "/content/drive/MyDrive/Amazon_ML_challenge"  # adjust to your path
os.chdir(REPO_PATH)
print("Working directory:", os.getcwd())
"""

# ════════════════════════════════════════════════════════════
# CELL 2 — Install packages
# ════════════════════════════════════════════════════════════
"""
!pip install -q scikit-learn pandas numpy faiss-cpu sentence-transformers \
    lightgbm catboost rapidfuzz jellyfish tqdm scipy
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — Run blocking benchmark (no GPU needed)
# ════════════════════════════════════════════════════════════
"""
!python pipeline_v2.py --mode block-only \
    --val-frac 0.1 \
    --top-k-name 30 \
    --top-k-addr 40 \
    --sn-window 10
"""

# ════════════════════════════════════════════════════════════
# CELL 4 — With dense FAISS blocking (needs sentence-transformers)
# ════════════════════════════════════════════════════════════
"""
!python pipeline_v2.py --mode block-only \
    --val-frac 0.1 \
    --top-k-name 30 \
    --top-k-addr 40 \
    --sn-window 10 \
    --use-dense \
    --top-k-dense 30
"""

# ════════════════════════════════════════════════════════════
# CELL 5 — View blocking recall report
# ════════════════════════════════════════════════════════════
"""
import pandas as pd
report = pd.read_csv("reports/blocking_benchmark.csv")
print(report.to_string(index=False))
"""

# ════════════════════════════════════════════════════════════
# CELL 6 — Quick diagnostic: how many true matches are being missed?
# ════════════════════════════════════════════════════════════
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from src.blocking.blocking_deterministic import run_deterministic_blocking
from src.blocking.blocking_tfidf import run_tfidf_blocking
from src.blocking.candidate_union import union_candidates
from src.evaluation.evaluate_blocking import parse_ground_truth

TRAIN_DIR = "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train"

s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\\t")
s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\\t")
gt = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\\t")
gt_dict = parse_ground_truth(gt)

# Sample to 1000 S1 records for quick testing
s1_sample = s1.sample(1000, random_state=42)

det = run_deterministic_blocking(s1_sample, s2)
tfidf = run_tfidf_blocking(s1_sample, s2, top_k_name=30, top_k_nameaddr=40)
union = union_candidates(det, tfidf)

# Recall computation
s1_eids_with_s2_matches = [
    eid for eid in s1_sample['entity_id']
    if any(e.startswith('S2') for e in gt_dict.get(eid, set()))
]

found = 0
total = 0
missed_examples = []
for eid in s1_eids_with_s2_matches:
    true_s2 = {e for e in gt_dict.get(eid, set()) if e.startswith('S2')}
    cands = union.get(eid, set())
    total += len(true_s2)
    found += len(true_s2 & cands)
    missed = true_s2 - cands
    if missed:
        missed_examples.append((eid, missed))

print(f"Blocking Recall (S2): {found}/{total} = {found/max(total,1)*100:.2f}%")
print(f"Missed: {total-found} true matches out of {total}")
print("\\nMissed examples (first 5):")
for eid, missed in missed_examples[:5]:
    print(f"  S1={eid} missed={missed}")
"""
