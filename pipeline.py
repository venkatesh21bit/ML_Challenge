"""
pipeline.py — End-to-end entity resolution pipeline.

Orchestrates:
  1. Load data
  2. Multi-pass blocking (deterministic + TF-IDF + dense + sorted neighbourhood)
  3. Union candidates
  4. Pairwise feature engineering
  5. LightGBM matching model
  6. Threshold tuning (F₀.₅)
  7. Export matching_results.tsv + candidate_pairs.tsv

Usage:
  # Train + predict on test:
  python pipeline.py --mode full

  # Just benchmark blocking (do this first!):
  python pipeline.py --mode block-only

  # Just predict (load saved model):
  python pipeline.py --mode predict --model-path outputs/lgbm_model.txt
"""

import argparse
import os
import sys
import time
import pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.blocking.blocking_deterministic         import run_deterministic_blocking
from src.blocking.blocking_tfidf                 import run_tfidf_blocking
from src.blocking.blocking_sorted_neighbourhood  import run_sorted_neighbourhood
from src.blocking.candidate_union                import (
    union_candidates, filter_to_valid_ids,
    ensure_all_s1_covered, export_candidate_pairs, candidate_stats,
)
from src.matching.features  import build_pair_feature_matrix, FEATURE_NAMES
from src.matching.matcher   import (
    train_lightgbm, predict_proba,
    tune_threshold, build_predictions_from_probs, compute_f05_macro,
)
from src.evaluation.evaluate_blocking import parse_ground_truth


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DATASET = "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset"
TRAIN_DIR    = os.path.join(BASE_DATASET, "train")
TEST_DIR     = os.path.join(BASE_DATASET, "test")
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("cache", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_train():
    print("[LOAD] Training data...")
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t")
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t")
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  GT={len(gt):,}")
    return s1, s2, s3, gt


def load_test():
    print("[LOAD] Test data...")
    s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t")
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
    return s1, s2, s3


# ─────────────────────────────────────────────────────────────────────────────
# Blocking pass (one source at a time)
# ─────────────────────────────────────────────────────────────────────────────

def block_one_source(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    source_label: str,
    top_k_tfidf_name: int = 20,
    top_k_tfidf_nameaddr: int = 30,
    sn_window: int = 10,
    max_candidates: int = 150,
    use_dense: bool = False,
    top_k_dense: int = 30,
) -> dict:
    print(f"\n{'='*50}")
    print(f"BLOCKING: S1 → {source_label}")
    print(f"{'='*50}")

    # Layer 1: Deterministic
    t = time.time()
    det = run_deterministic_blocking(s1, s_other, verbose=True)
    print(f"  Deterministic done in {time.time()-t:.1f}s")

    # Layer 2: TF-IDF
    t = time.time()
    tfidf = run_tfidf_blocking(
        s1, s_other,
        top_k_name=top_k_tfidf_name,
        top_k_nameaddr=top_k_tfidf_nameaddr,
        verbose=True,
    )
    print(f"  TF-IDF done in {time.time()-t:.1f}s")

    # Layer 4: Sorted Neighbourhood
    t = time.time()
    sn = run_sorted_neighbourhood(s1, s_other, window_size=sn_window, verbose=True)
    print(f"  Sorted Neighbourhood done in {time.time()-t:.1f}s")

    # Optional Layer 3: Dense FAISS
    layers = [det, tfidf, sn]
    if use_dense:
        from src.blocking.blocking_dense import run_dense_blocking
        t = time.time()
        dense = run_dense_blocking(s1, s_other, top_k=top_k_dense, verbose=True)
        print(f"  Dense done in {time.time()-t:.1f}s")
        layers.append(dense)

    # Union
    union = union_candidates(*layers, max_candidates=max_candidates)
    stats = candidate_stats(union)
    print(f"\n  UNION: avg={stats['avg_candidates']:.1f} "
          f"median={stats['median_candidates']:.0f} "
          f"max={stats['max_candidates']} "
          f"total_pairs={stats['total_pairs']:,}")
    return union


# ─────────────────────────────────────────────────────────────────────────────
# Export submission
# ─────────────────────────────────────────────────────────────────────────────

def export_matching_results(
    predictions: dict,
    s1_entity_ids: list,
    output_path: str,
):
    rows = []
    for eid in s1_entity_ids:
        matches = predictions.get(eid, set())
        rows.append({
            "source1_entity_id": eid,
            "matched_entity_ids": ",".join(sorted(matches)),
        })
    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Matching results saved: {output_path} ({len(df):,} rows)")
    n_match = sum(1 for r in rows if r["matched_entity_ids"])
    print(f"  {n_match:,} entities have at least 1 match, "
          f"{len(df)-n_match:,} singletons")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(args):
    # ── 1. Load train ────────────────────────────────────────────────────────
    s1_train, s2_train, s3_train, gt_train = load_train()
    gt_dict = parse_ground_truth(gt_train)

    # ── 2. Val split ─────────────────────────────────────────────────────────
    val_frac = args.val_frac
    s1_val   = s1_train.sample(frac=val_frac, random_state=42)
    s1_trn   = s1_train.drop(s1_val.index)
    print(f"\n[SPLIT] train={len(s1_trn):,}  val={len(s1_val):,}")

    # ── 3. Blocking on train ──────────────────────────────────────────────────
    print("\n[BLOCKING] Train set...")
    cands_trn_s2 = block_one_source(s1_trn, s2_train, "S2",
                                     top_k_tfidf_name=args.top_k_name,
                                     top_k_tfidf_nameaddr=args.top_k_addr,
                                     sn_window=args.sn_window,
                                     use_dense=args.use_dense)
    cands_trn_s3 = block_one_source(s1_trn, s3_train, "S3",
                                     top_k_tfidf_name=args.top_k_name,
                                     top_k_tfidf_nameaddr=args.top_k_addr,
                                     sn_window=args.sn_window,
                                     use_dense=args.use_dense)
    # Merge S2+S3 candidates per S1
    cands_trn = {}
    for eid in s1_trn["entity_id"]:
        cands_trn[eid] = (cands_trn_s2.get(eid, set()) |
                          cands_trn_s3.get(eid, set()))

    # ── 4. Blocking on val ───────────────────────────────────────────────────
    print("\n[BLOCKING] Val set...")
    cands_val_s2 = block_one_source(s1_val, s2_train, "S2",
                                     top_k_tfidf_name=args.top_k_name,
                                     top_k_tfidf_nameaddr=args.top_k_addr,
                                     sn_window=args.sn_window,
                                     use_dense=args.use_dense)
    cands_val_s3 = block_one_source(s1_val, s3_train, "S3",
                                     top_k_tfidf_name=args.top_k_name,
                                     top_k_tfidf_nameaddr=args.top_k_addr,
                                     sn_window=args.sn_window,
                                     use_dense=args.use_dense)
    cands_val = {}
    for eid in s1_val["entity_id"]:
        cands_val[eid] = (cands_val_s2.get(eid, set()) |
                          cands_val_s3.get(eid, set()))

    # ── 5. Feature engineering ───────────────────────────────────────────────
    print("\n[FEATURES] Building train features...")
    s_all = pd.concat([s2_train, s3_train], ignore_index=True)

    X_trn, y_trn, _ = build_pair_feature_matrix(
        s1_trn, s_all,
        {k: list(v) for k, v in cands_trn.items()},
        labels={k: gt_dict.get(k, set()) for k in cands_trn},
    )
    print(f"  Train pairs: {len(X_trn):,}  pos={y_trn.sum():,}  neg={(y_trn==0).sum():,}")

    print("[FEATURES] Building val features...")
    X_val_f, y_val_f, val_pair_ids = build_pair_feature_matrix(
        s1_val, s_all,
        {k: list(v) for k, v in cands_val.items()},
        labels={k: gt_dict.get(k, set()) for k in cands_val},
    )
    print(f"  Val pairs: {len(X_val_f):,}  pos={y_val_f.sum():,}  neg={(y_val_f==0).sum():,}")

    # ── 6. Train matcher ─────────────────────────────────────────────────────
    print("\n[TRAIN] LightGBM matcher...")
    model_path = os.path.join(OUTPUT_DIR, "lgbm_model.txt")
    model = train_lightgbm(
        X_trn, y_trn,
        X_val=X_val_f, y_val=y_val_f,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        model_path=model_path,
    )

    # ── 7. Threshold tuning on val ───────────────────────────────────────────
    print("\n[THRESHOLD] Tuning on validation set...")
    val_gt = {k: gt_dict.get(k, set()) for k in s1_val["entity_id"]}
    val_probs = predict_proba(model, X_val_f)
    best_t, best_f05 = tune_threshold(val_probs, val_pair_ids, val_gt)
    print(f"  Val F₀.₅ = {best_f05:.4f} @ threshold={best_t:.3f}")

    # ── 8. Predict on test ───────────────────────────────────────────────────
    print("\n[TEST] Loading test data and running blocking...")
    s1_test, s2_test, s3_test = load_test()

    cands_test_s2 = block_one_source(s1_test, s2_test, "S2-test",
                                      top_k_tfidf_name=args.top_k_name,
                                      top_k_tfidf_nameaddr=args.top_k_addr,
                                      sn_window=args.sn_window,
                                      use_dense=args.use_dense)
    cands_test_s3 = block_one_source(s1_test, s3_test, "S3-test",
                                      top_k_tfidf_name=args.top_k_name,
                                      top_k_tfidf_nameaddr=args.top_k_addr,
                                      sn_window=args.sn_window,
                                      use_dense=args.use_dense)
    cands_test = {}
    for eid in s1_test["entity_id"]:
        cands_test[eid] = (cands_test_s2.get(eid, set()) |
                           cands_test_s3.get(eid, set()))

    ensure_all_s1_covered(cands_test, s1_test["entity_id"].tolist())

    # Filter to valid IDs
    valid_s2 = set(s2_test["entity_id"])
    valid_s3 = set(s3_test["entity_id"])
    cands_test = filter_to_valid_ids(cands_test, valid_s2, valid_s3)

    # Export candidate pairs
    export_candidate_pairs(
        cands_test,
        os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
    )

    # Build test features
    print("[FEATURES] Building test features...")
    s_test_all = pd.concat([s2_test, s3_test], ignore_index=True)
    X_test, _, test_pair_ids = build_pair_feature_matrix(
        s1_test, s_test_all,
        {k: list(v) for k, v in cands_test.items()},
    )
    print(f"  Test pairs: {len(X_test):,}")

    # Predict
    test_probs   = predict_proba(model, X_test)
    test_preds   = build_predictions_from_probs(test_probs, test_pair_ids, threshold=best_t)
    ensure_all_s1_covered(test_preds, s1_test["entity_id"].tolist())

    # Export matching results
    export_matching_results(
        test_preds,
        s1_test["entity_id"].tolist(),
        os.path.join(OUTPUT_DIR, "matching_results.tsv"),
    )

    print(f"\n{'='*60}")
    print(f"DONE. Outputs saved to {OUTPUT_DIR}/")
    print(f"  matching_results.tsv  ← upload to leaderboard")
    print(f"  candidate_pairs.tsv   ← include in submission zip")
    print(f"  Val F₀.₅ = {best_f05:.4f}")
    print(f"{'='*60}")


def run_block_only(args):
    """Just run the blocking benchmark (fast mode)."""
    from src.evaluation.evaluate_blocking import run_benchmark
    run_benchmark(
        train_dir=TRAIN_DIR,
        val_frac=args.val_frac,
        top_k_tfidf_name=args.top_k_name,
        top_k_tfidf_nameaddr=args.top_k_addr,
        sn_window=args.sn_window,
        output_csv="reports/blocking_benchmark.csv",
        skip_dense=not args.use_dense,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument("--mode",       choices=["full", "block-only"], default="block-only",
                        help="full=train+predict | block-only=benchmark blocking")
    parser.add_argument("--val-frac",   type=float, default=0.05,
                        help="Fraction of S1 held out for validation (default 5%%)")
    parser.add_argument("--top-k-name", type=int, default=20,
                        help="TF-IDF top-k for name index")
    parser.add_argument("--top-k-addr", type=int, default=30,
                        help="TF-IDF top-k for name+address index")
    parser.add_argument("--sn-window",  type=int, default=10,
                        help="Sorted Neighbourhood window size")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--lr",         type=float, default=0.05)
    parser.add_argument("--use-dense",  action="store_true",
                        help="Enable FAISS dense blocking (slow without GPU)")
    parser.add_argument("--model-path", default=None,
                        help="Load saved model instead of training")
    args = parser.parse_args()

    if args.mode == "block-only":
        run_block_only(args)
    elif args.mode == "full":
        run_full_pipeline(args)
