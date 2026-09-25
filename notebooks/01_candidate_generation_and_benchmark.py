"""
01_candidate_generation_and_benchmark.py
=========================================
Standalone runner & Colab template for:
  1. Benchmarking blocking recall on validation split
  2. Generating official test candidate pairs (outputs/candidate_pairs.tsv)
  3. Pre-caching train candidates for rapid ML model training
"""

import os
import sys

# ════════════════════════════════════════════════════════════
# CELL 1 — Setup & Paths
# ════════════════════════════════════════════════════════════
"""
from google.colab import drive
drive.mount('/content/drive')

import os, sys
REPO_PATH = "/content/drive/MyDrive/Amazon_ML_challenge"
os.chdir(REPO_PATH)
sys.path.insert(0, REPO_PATH)
print("Working directory:", os.getcwd())

!pip install -q scikit-learn pandas numpy faiss-cpu lightgbm catboost rapidfuzz jellyfish tqdm scipy
"""

# ════════════════════════════════════════════════════════════
# CELL 2 — Run Blocking Benchmark on Validation Split
# ════════════════════════════════════════════════════════════
"""
!python pipeline_v2.py --mode block-only \
    --val-frac 0.05 \
    --max-val-samples 10000 \
    --top-k-name 30 \
    --top-k-addr 40 \
    --sn-window 10
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — Inspect Benchmark Report
# ════════════════════════════════════════════════════════════
"""
import pandas as pd
df_report = pd.read_csv("reports/blocking_benchmark.csv")
display(df_report)
"""

# ════════════════════════════════════════════════════════════
# CELL 4 — Generate Full Candidate Pairs for Test Set
# ════════════════════════════════════════════════════════════
"""
!python pipeline_v2.py --mode generate-candidates \
    --target-split test \
    --top-k-name 30 \
    --top-k-addr 40 \
    --sn-window 10 \
    --max-candidates 150
"""

# ════════════════════════════════════════════════════════════
# CELL 5 — Validate Submission File
# ════════════════════════════════════════════════════════════
"""
!python datasets/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --candidate outputs/candidate_pairs.tsv \
    --test-dir datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/test
"""

# ════════════════════════════════════════════════════════════
# CELL 6 — Generate & Cache Train Candidates (for CatBoost & DeBERTa)
# ════════════════════════════════════════════════════════════
"""
!python pipeline_v2.py --mode generate-candidates \
    --target-split train \
    --max-train-samples 50000 \
    --top-k-name 30 \
    --top-k-addr 40 \
    --sn-window 10
"""
