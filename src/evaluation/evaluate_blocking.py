"""
evaluate_blocking.py — Blocking benchmark: recall, reduction ratio, runtime.

This is the MOST IMPORTANT experiment before touching matching models.
Run this first to know which blocking configuration gives ≥99% recall.

Usage:
    python -m src.evaluation.evaluate_blocking \
        --train-dir datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train \
        --val-frac 0.1 \
        --output reports/blocking_benchmark.csv
"""

import argparse
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional
import sys, os

# ── path fix for running as __main__ ─────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.blocking.blocking_deterministic    import run_deterministic_blocking
from src.blocking.blocking_tfidf            import run_tfidf_blocking
from src.blocking.blocking_sorted_neighbourhood import run_sorted_neighbourhood
from src.blocking.candidate_union           import union_candidates, candidate_stats


# ─────────────────────────────────────────────────────────────────────────────
# Recall computation
# ─────────────────────────────────────────────────────────────────────────────

def parse_ground_truth(gt: pd.DataFrame) -> Dict[str, Set[str]]:
    """Parse ground truth TSV into {s1_eid → set of matched eids}. Vectorized."""
    eids = gt["source1_entity_id"].tolist()
    raws = gt["matched_entity_ids"].fillna("").astype(str).tolist()
    result = {}
    for eid, raw in zip(eids, raws):
        raw = raw.strip()
        if not raw or raw == "nan":
            result[eid] = set()
        else:
            result[eid] = set(x.strip() for x in raw.split(",") if x.strip())
    return result


def compute_blocking_recall(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    source_filter: Optional[str] = None,   # "S2" or "S3" or None (all)
) -> Tuple[float, int, int, int]:
    """
    Compute blocking recall = (true pairs recovered) / (total true pairs).
    Also returns: recovered, total_true_pairs, missed.
    """
    recovered = 0
    missed    = 0
    total     = 0

    for s1_eid, true_matches in ground_truth.items():
        if source_filter:
            true_matches = {m for m in true_matches if m.startswith(source_filter)}
        total += len(true_matches)
        for tm in true_matches:
            if tm in candidates.get(s1_eid, set()):
                recovered += 1
            else:
                missed += 1

    recall = recovered / total if total > 0 else 1.0
    return recall, recovered, total, missed


def benchmark_single_pass(
    label: str,
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    elapsed: float,
) -> dict:
    stats = candidate_stats(candidates)
    recall_all, rec, tot, miss = compute_blocking_recall(candidates, ground_truth)
    print(f"  [{label}]  recall={recall_all:.4f}  "
          f"avg_cands={stats['avg_candidates']:.1f}  "
          f"total_pairs={stats['total_pairs']:,}  "
          f"time={elapsed:.1f}s  "
          f"missed={miss}/{tot}")
    return {
        "method": label,
        "recall": recall_all,
        "recovered": rec,
        "total_true": tot,
        "missed": miss,
        "avg_candidates": stats["avg_candidates"],
        "median_candidates": stats["median_candidates"],
        "max_candidates": stats["max_candidates"],
        "n_singletons": stats["n_singletons"],
        "total_pairs": stats["total_pairs"],
        "time_sec": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    train_dir: str,
    val_frac: float = 0.05,
    max_val_samples: int = 10000,
    top_k_tfidf_name: int = 20,
    top_k_tfidf_nameaddr: int = 30,
    top_k_dense: int = 30,
    sn_window: int = 10,
    output_csv: str = "reports/blocking_benchmark.csv",
    skip_dense: bool = True,   # skip dense by default (slow at 5M scale without GPU)
    seed: int = 42,
):
    import gc
    print("=" * 70)
    print("BLOCKING BENCHMARK — Amazon ML Challenge")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    s1 = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t")
    gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t")

    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}  GT: {len(gt):,}")

    # ── Validation split from S1 ─────────────────────────────────────────────
    val_size = min(int(len(s1) * val_frac), max_val_samples) if max_val_samples > 0 else int(len(s1) * val_frac)
    print(f"\n[2] Sampling {val_size:,} of S1 for fast validation (capped at {max_val_samples:,})...")
    s1_val = s1.sample(n=val_size, random_state=seed).reset_index(drop=True)
    gt_dict = parse_ground_truth(gt)
    gt_val  = {eid: gt_dict.get(eid, set()) for eid in s1_val["entity_id"]}

    # Free memory immediately to prevent Colab OOM
    del s1, gt, gt_dict
    gc.collect()

    print(f"  Val S1 size: {len(s1_val):,}")
    n_with_matches = sum(1 for v in gt_val.values() if v)
    print(f"  S1 entities with ≥1 match: {n_with_matches:,}")

    results = []

    # ── Layer 1: Deterministic ───────────────────────────────────────────────
    print("\n[3] Running DETERMINISTIC blocking (S1 → S2)...")
    t = time.time()
    det_s2 = run_deterministic_blocking(s1_val, s2, verbose=True)
    results.append(benchmark_single_pass("deterministic_s2", det_s2, gt_val, time.time() - t))

    print("\n[4] Running DETERMINISTIC blocking (S1 → S3)...")
    t = time.time()
    det_s3 = run_deterministic_blocking(s1_val, s3, verbose=True)
    results.append(benchmark_single_pass("deterministic_s3", det_s3, gt_val, time.time() - t))

    det_union = union_candidates(det_s2, det_s3)
    results.append(benchmark_single_pass("deterministic_union", det_union, gt_val, 0))

    # ── Layer 2: TF-IDF ──────────────────────────────────────────────────────
    print("\n[5] Running TF-IDF blocking (S1 → S2)...")
    t = time.time()
    tfidf_s2 = run_tfidf_blocking(
        s1_val, s2,
        top_k_name=top_k_tfidf_name,
        top_k_nameaddr=top_k_tfidf_nameaddr,
        verbose=True,
    )
    results.append(benchmark_single_pass("tfidf_s2", tfidf_s2, gt_val, time.time() - t))

    print("\n[6] Running TF-IDF blocking (S1 → S3)...")
    t = time.time()
    tfidf_s3 = run_tfidf_blocking(
        s1_val, s3,
        top_k_name=top_k_tfidf_name,
        top_k_nameaddr=top_k_tfidf_nameaddr,
        verbose=True,
    )
    results.append(benchmark_single_pass("tfidf_s3", tfidf_s3, gt_val, time.time() - t))

    tfidf_union = union_candidates(tfidf_s2, tfidf_s3)
    results.append(benchmark_single_pass("tfidf_union", tfidf_union, gt_val, 0))

    # ── Layer 4: Sorted Neighbourhood ────────────────────────────────────────
    print("\n[7] Running SORTED NEIGHBOURHOOD (S1 → S2)...")
    t = time.time()
    sn_s2 = run_sorted_neighbourhood(s1_val, s2, window_size=sn_window, verbose=True)
    results.append(benchmark_single_pass("sorted_nbr_s2", sn_s2, gt_val, time.time() - t))

    print("\n[8] Running SORTED NEIGHBOURHOOD (S1 → S3)...")
    t = time.time()
    sn_s3 = run_sorted_neighbourhood(s1_val, s3, window_size=sn_window, verbose=True)
    results.append(benchmark_single_pass("sorted_nbr_s3", sn_s3, gt_val, time.time() - t))

    # ── Dense (optional) ─────────────────────────────────────────────────────
    if not skip_dense:
        print("\n[9] Running DENSE (FAISS) blocking (S1 → S2)...")
        from src.blocking.blocking_dense import run_dense_blocking
        t = time.time()
        dense_s2 = run_dense_blocking(s1_val, s2, top_k=top_k_dense, verbose=True)
        results.append(benchmark_single_pass("dense_s2", dense_s2, gt_val, time.time() - t))

        print("\n[10] Running DENSE (FAISS) blocking (S1 → S3)...")
        t = time.time()
        dense_s3 = run_dense_blocking(s1_val, s3, top_k=top_k_dense, verbose=True)
        results.append(benchmark_single_pass("dense_s3", dense_s3, gt_val, time.time() - t))

    # ── FINAL UNION ──────────────────────────────────────────────────────────
    print("\n[UNION] Computing UNION of all methods...")
    all_dicts = [det_s2, det_s3, tfidf_s2, tfidf_s3, sn_s2, sn_s3]
    if not skip_dense:
        all_dicts += [dense_s2, dense_s3]

    final_union = union_candidates(*all_dicts, max_candidates=200)
    results.append(benchmark_single_pass("FINAL_UNION", final_union, gt_val, 0))

    # ── Error analysis on missed pairs ───────────────────────────────────────
    print("\n[ERROR ANALYSIS] Missed true pairs in FINAL_UNION:")
    missed_pairs = []
    for s1_eid, true_matches in gt_val.items():
        for tm in true_matches:
            if tm not in final_union.get(s1_eid, set()):
                missed_pairs.append((s1_eid, tm))
                if len(missed_pairs) >= 20:
                    break
        if len(missed_pairs) >= 20:
            break

    if missed_pairs:
        needed_s1 = {p[0] for p in missed_pairs}
        needed_s2 = {p[1] for p in missed_pairs if p[1].startswith("S2")}
        needed_s3 = {p[1] for p in missed_pairs if p[1].startswith("S3")}

        s1_id_map = s1_val[s1_val["entity_id"].isin(needed_s1)].set_index("entity_id").to_dict("index")
        s2_id_map = s2[s2["entity_id"].isin(needed_s2)].set_index("entity_id").to_dict("index") if needed_s2 else {}
        s3_id_map = s3[s3["entity_id"].isin(needed_s3)].set_index("entity_id").to_dict("index") if needed_s3 else {}

        missed_examples = []
        for s1_eid, tm in missed_pairs:
            s1_r = s1_id_map.get(s1_eid, {})
            other_map = s2_id_map if tm.startswith("S2") else s3_id_map
            s_r = other_map.get(tm, {})
            missed_examples.append({
                "s1_eid": s1_eid,
                "s1_name": s1_r.get("business_name", ""),
                "s1_addr": s1_r.get("business_address", ""),
                "match_eid": tm,
                "match_name": s_r.get("business_name", ""),
                "match_addr": s_r.get("business_address", ""),
            })

    if missed_examples:
        print("\n  Top missed pairs (analyze these to add new blocking rules):")
        for ex in missed_examples[:10]:
            print(f"    S1: '{ex['s1_name']}' | '{ex['s1_addr']}'")
            print(f"    →?: '{ex['match_name']}' | '{ex['match_addr']}'")
            print()

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\n[SAVED] Benchmark results → {output_csv}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(df_results[["method", "recall", "avg_candidates", "total_pairs", "time_sec"]].to_string(index=False))

    # Key target
    final = df_results[df_results["method"] == "FINAL_UNION"].iloc[0]
    print(f"\n🎯 FINAL UNION recall = {final['recall']:.4f} ({final['recall']*100:.2f}%)")
    if final["recall"] >= 0.98:
        print("   ✅ Target ≥98% reached!")
    else:
        print("   ⚠️  Below 98% — increase top_k or add more blocking rules")

    return df_results, final_union


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blocking benchmark")
    parser.add_argument("--train-dir",  default="datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train")
    parser.add_argument("--val-frac",   type=float, default=0.05)
    parser.add_argument("--top-k-name", type=int, default=20)
    parser.add_argument("--top-k-addr", type=int, default=30)
    parser.add_argument("--top-k-dense",type=int, default=30)
    parser.add_argument("--sn-window",  type=int, default=10)
    parser.add_argument("--output",     default="reports/blocking_benchmark.csv")
    parser.add_argument("--with-dense", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        train_dir=args.train_dir,
        val_frac=args.val_frac,
        top_k_tfidf_name=args.top_k_name,
        top_k_tfidf_nameaddr=args.top_k_addr,
        top_k_dense=args.top_k_dense,
        sn_window=args.sn_window,
        output_csv=args.output,
        skip_dense=not args.with_dense,
    )
