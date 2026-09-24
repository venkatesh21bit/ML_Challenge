# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## Problem
Match business records from Source 1 against Source 2 and Source 3.
Metric: **F₀.₅** (precision-heavy macro-average).

## Quick Start

```powershell
# Always use Python 3.11 (NOT the system python which is 3.12 without pip)
$py = "C:\Users\91902\AppData\Local\Programs\Python\Python311\python.exe"
$env:PYTHONUTF8 = "1"

# 1. Install dependencies
& $py -m pip install --user -r requirements.txt

# 2. Smoke test (verify normalization)
& $py test_normalize.py

# 3. Benchmark blocking (DO THIS FIRST — ~30-50 min)
& $py pipeline.py --mode block-only --val-frac 0.02

# 4. Full train + predict on test
& $py pipeline.py --mode full --val-frac 0.1

# 5. Validate submission files
& $py datasets/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py `
    --matching outputs/matching_results.tsv `
    --candidate outputs/candidate_pairs.tsv `
    --test-dir datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

## Architecture — Multi-Pass Hybrid Blocking

```
Source 1
│
├── Layer 1: Deterministic Blocking (fast, ~2-4 min)
│   5 blocking keys: country+name_prefix, country+addr_prefix,
│   name_prefix_6, sorted_tokens, name+addr_prefix
│
├── Layer 2: Char n-gram TF-IDF (primary fuzzy, ~10-15 min)
│   - Index A: name only  (top-20)
│   - Index B: name+addr  (top-30)
│   - 2-4 gram char_wb analyzer
│
├── Layer 3: Dense FAISS (semantic, --use-dense flag)
│   - sentence-transformers/all-MiniLM-L6-v2
│   - IVF index for >100K records
│
└── Layer 4: Sorted Neighbourhood (fallback, ~3-5 min)
    - Sort by country + normalized_name
    - Sliding window size=10
│
▼
UNION + DEDUPLICATE (~30-150 candidates/entity)
│
▼
21 pairwise similarity features
(Jaccard, token overlap, edit distance, char n-gram,
 prefix match, length ratio, country match)
│
▼
LightGBM classifier + F₀.₅ threshold sweep
│
▼
matching_results.tsv + candidate_pairs.tsv
```

## Key Pipeline Args

| Arg | Default | Description |
|-----|---------|-------------|
| `--val-frac` | 0.05 | Fraction of S1 held out for validation |
| `--top-k-name` | 20 | TF-IDF top-k for name index |
| `--top-k-addr` | 30 | TF-IDF top-k for name+address index |
| `--sn-window` | 10 | Sorted Neighbourhood window size |
| `--use-dense` | False | Enable FAISS dense blocking (needs GPU/Kaggle) |
| `--n-estimators` | 1000 | LightGBM trees |

## Source Files

| File | Description |
|------|-------------|
| `src/data/normalize.py` | Text normalization, legal suffix stripping, blocking key generators |
| `src/blocking/blocking_deterministic.py` | Layer 1: inverted-index blocking |
| `src/blocking/blocking_tfidf.py` | Layer 2: char n-gram TF-IDF |
| `src/blocking/blocking_dense.py` | Layer 3: FAISS ANN search |
| `src/blocking/blocking_sorted_neighbourhood.py` | Layer 4: sorted window |
| `src/blocking/candidate_union.py` | Union + cap + TSV export |
| `src/matching/features.py` | 21 pairwise similarity features |
| `src/matching/matcher.py` | LightGBM/CatBoost + F₀.₅ tuning |
| `src/evaluation/evaluate_blocking.py` | Blocking recall benchmark |
| `pipeline.py` | End-to-end orchestrator |

## Outputs

```
outputs/
├── matching_results.tsv    ← upload to leaderboard
├── candidate_pairs.tsv     ← include in final zip
└── lgbm_model.txt          ← saved LightGBM model
reports/
└── blocking_benchmark.csv  ← per-method recall table
```

## Important Notes

- **Test set includes France** (not in training). Pipeline handles country as open string — no hardcoding.
- **Always use `--user` for pip installs** on this machine (system Python 3.12 conflicts with scipy).
- **Benchmark first** — `--mode block-only` tells you blocking recall before wasting time training.
- **Target**: FINAL_UNION blocking recall ≥ 98% before touching the matcher.
- **Singletons matter**: F₀.₅ gives 1.0 for correct empty predictions and 0.0 for any wrong match on singletons.

## Team GPU Split

| Platform | Use For |
|----------|---------|
| Local RTX 4060 | EDA, CatBoost, ensemble, CPU blocking |
| Kaggle (4 accounts) | TF-IDF + dense embedding extraction in parallel |
| AWS ($200 credits) | Heavy fine-tuning Day 2-3 only |
