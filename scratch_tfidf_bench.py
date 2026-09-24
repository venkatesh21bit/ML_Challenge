"""
Quick TF-IDF-only blocking benchmark on 1% of S1 (~22K records).
Skips deterministic blocking entirely to get TF-IDF recall fast.
"""
import sys, os, time, pandas as pd, numpy as np
sys.path.insert(0, '.')

TRAIN = 'datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train'

print("[1] Loading data...")
s1 = pd.read_csv(f'{TRAIN}/train_source1.tsv', sep='\t')
s2 = pd.read_csv(f'{TRAIN}/train_source2.tsv', sep='\t')
gt = pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t')
print(f"  S1={len(s1):,} S2={len(s2):,} GT={len(gt):,}")

# Parse GT
print("[2] Parsing ground truth...")
gt_dict = {}
for eid, raw in zip(gt['source1_entity_id'].tolist(),
                    gt['matched_entity_ids'].fillna('').astype(str).tolist()):
    raw = raw.strip()
    gt_dict[eid] = set(x.strip() for x in raw.split(',') if x.strip()) if raw and raw != 'nan' else set()

# 1% sample
s1_val = s1.sample(frac=0.01, random_state=42).reset_index(drop=True)
gt_val = {eid: gt_dict.get(eid, set()) for eid in s1_val['entity_id']}
n_true = sum(len(v) for v in gt_val.values())
print(f"  Val S1={len(s1_val):,}, true pairs={n_true:,}")

# Run TF-IDF blocking S1 -> S2
from src.blocking.blocking_tfidf import run_tfidf_blocking

print("\n[3] TF-IDF blocking S1 → S2...")
t = time.time()
tfidf_s2 = run_tfidf_blocking(s1_val, s2, top_k_name=30, top_k_nameaddr=40, verbose=True)
elapsed = time.time() - t

# Compute recall
recovered, total, missed = 0, 0, 0
for eid, true_set in gt_val.items():
    s2_true = {m for m in true_set if m.startswith('S2-')}
    total += len(s2_true)
    for tm in s2_true:
        if tm in tfidf_s2.get(eid, set()): recovered += 1
        else: missed += 1

recall = recovered / total if total > 0 else 1.0
sizes = [len(v) for v in tfidf_s2.values()]
print(f"\n=== TF-IDF S1->S2 RESULTS ===")
print(f"  recall:     {recall:.4f} ({recall*100:.2f}%)")
print(f"  avg_cands:  {np.mean(sizes):.1f}")
print(f"  max_cands:  {max(sizes)}")
print(f"  total_pairs:{sum(sizes):,}")
print(f"  recovered:  {recovered}/{total} (missed={missed})")
print(f"  time:       {elapsed:.1f}s")
if recall >= 0.95:
    print("  TARGET MET (>=95% recall)")
elif recall >= 0.90:
    print("  CLOSE — increase top_k to reach 95%")
else:
    print("  LOW — check normalization or increase top_k")
