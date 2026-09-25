"""
pipeline_v2.py — Full AIR #1 entity resolution pipeline (upgraded).

Stages:
  A. Data loading + normalization
  B. Multi-pass blocking (deterministic + TF-IDF + dense FAISS + sorted neighbourhood)
  C. Hard negative mining
  D. BGE-M3 embedding computation (optional, cached)
  E. Feature engineering v2 (56 features including embeddings)
  F. CatBoost / LightGBM training with GroupKFold CV
  G. DeBERTa cross-encoder fine-tuning (optional)
  H. BGE Reranker scoring (optional)
  I. Ensemble + weight optimization
  J. Threshold tuning for F0.5
  K. Export candidate_pairs.tsv + matching_results.tsv

Usage on Colab / local:
  python pipeline_v2.py --mode block-only       # Benchmark blocking recall first
  python pipeline_v2.py --mode train-catboost   # Train GBM only (fast baseline)
  python pipeline_v2.py --mode full             # Full ensemble pipeline
  python pipeline_v2.py --mode train-ce         # Fine-tune cross-encoder
"""

import argparse
import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.blocking.blocking_deterministic         import run_deterministic_blocking
from src.blocking.blocking_tfidf                 import run_tfidf_blocking
from src.blocking.blocking_sorted_neighbourhood  import run_sorted_neighbourhood
from src.blocking.candidate_union                import (
    union_candidates, filter_to_valid_ids,
    ensure_all_s1_covered, export_candidate_pairs, candidate_stats,
)
from src.matching.features_v2   import build_pair_feature_matrix_v2, FEATURE_NAMES_V2
from src.matching.hard_negatives import build_balanced_training_pairs
from src.matching.matcher       import (
    train_lightgbm, train_catboost, predict_proba,
)
from src.matching.ensemble      import (
    EnsembleScorer, tune_threshold, optimize_weights_grid,
    scores_dict_to_array, compute_f05_macro,
)
from src.evaluation.evaluate_blocking import parse_ground_truth


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DATASET = "dataset/student_resource/dataset"
TRAIN_DIR    = os.path.join(BASE_DATASET, "train")
TEST_DIR     = os.path.join(BASE_DATASET, "test")
OUTPUT_DIR   = "outputs"
CACHE_DIR    = "cache"
REPORTS_DIR  = "reports"

for d in [OUTPUT_DIR, CACHE_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


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
    s1, s_other, source_label,
    top_k_name=30, top_k_addr=40, sn_window=10,
    max_candidates=150, use_dense=False, top_k_dense=30,
    faiss_cache_path=None,
):
    print(f"\n{'='*50}\nBLOCKING: S1 → {source_label}\n{'='*50}")
    layers = []

    t = time.time()
    det = run_deterministic_blocking(s1, s_other, verbose=True)
    print(f"  Deterministic done in {time.time()-t:.1f}s")
    layers.append(det)

    t = time.time()
    tfidf = run_tfidf_blocking(s1, s_other, top_k_name=top_k_name,
                                top_k_nameaddr=top_k_addr, verbose=True)
    print(f"  TF-IDF done in {time.time()-t:.1f}s")
    layers.append(tfidf)

    t = time.time()
    sn = run_sorted_neighbourhood(s1, s_other, window_size=sn_window, verbose=True)
    print(f"  Sorted Neighbourhood done in {time.time()-t:.1f}s")
    layers.append(sn)

    if use_dense:
        from src.blocking.blocking_dense import run_dense_blocking
        t = time.time()
        dense = run_dense_blocking(
            s1, s_other, top_k=top_k_dense, verbose=True,
            index_cache_path=faiss_cache_path,
        )
        print(f"  Dense FAISS done in {time.time()-t:.1f}s")
        layers.append(dense)

    union = union_candidates(*layers, max_candidates=max_candidates)
    stats = candidate_stats(union)
    print(f"\n  UNION: avg={stats['avg_candidates']:.1f} "
          f"median={stats['median_candidates']:.0f} "
          f"max={stats['max_candidates']} "
          f"total_pairs={stats['total_pairs']:,}")
    return union


def merge_s2_s3_candidates(s1_eids, cands_s2, cands_s3):
    merged = {}
    for eid in s1_eids:
        merged[eid] = (cands_s2.get(eid, set()) | cands_s3.get(eid, set()))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Optional: BGE-M3 embedding computation
# ─────────────────────────────────────────────────────────────────────────────

def get_or_compute_embeddings(df, label, cache_path, args):
    if not args.use_embeddings:
        return None
    from src.matching.bge_reranker import build_entity_embeddings, save_embeddings, load_embeddings
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings: {cache_path}")
        return load_embeddings(cache_path)
    print(f"  Computing BGE-M3 embeddings for {label} ({len(df):,} records)...")
    embs = build_entity_embeddings(df, device=args.device)
    save_embeddings(embs, cache_path)
    return embs


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_matching_results(predictions, s1_entity_ids, output_path):
    rows = []
    for eid in s1_entity_ids:
        matches = predictions.get(eid, set())
        rows.append({
            "source1_entity_id": eid,
            "matched_entity_ids": ",".join(sorted(matches)),
        })
    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(output_path, sep="\t", index=False)
    n_match = sum(1 for r in rows if r["matched_entity_ids"])
    print(f"  Saved {output_path}: {n_match:,} matched, {len(df)-n_match:,} singletons")


# ─────────────────────────────────────────────────────────────────────────────
# MODE: CatBoost training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_catboost_pipeline(args):
    # ── Load data
    s1_train, s2_train, s3_train, gt_train = load_train()
    gt_dict = parse_ground_truth(gt_train)

    # ── Val split
    s1_val = s1_train.sample(frac=args.val_frac, random_state=42)
    s1_trn = s1_train.drop(s1_val.index)
    if args.max_train_samples > 0:
        s1_trn = s1_trn.sample(min(args.max_train_samples, len(s1_trn)), random_state=42)
    s1_trn = s1_trn.reset_index(drop=True)
    print(f"\n[SPLIT] train={len(s1_trn):,}  val={len(s1_val):,}")

    # ── Blocking
    print("\n[BLOCKING] Train set...")
    cands_trn = merge_s2_s3_candidates(
        s1_trn["entity_id"],
        block_one_source(s1_trn, s2_train, "S2-train", **_block_kwargs(args)),
        block_one_source(s1_trn, s3_train, "S3-train", **_block_kwargs(args)),
    )
    print("\n[BLOCKING] Val set...")
    cands_val = merge_s2_s3_candidates(
        s1_val["entity_id"],
        block_one_source(s1_val, s2_train, "S2-val", **_block_kwargs(args)),
        block_one_source(s1_val, s3_train, "S3-val", **_block_kwargs(args)),
    )

    # ── Embeddings (optional)
    s_all = pd.concat([s2_train, s3_train], ignore_index=True)
    emb_s1_trn = get_or_compute_embeddings(
        s1_trn, "s1_train", f"{CACHE_DIR}/emb_s1_train.pkl", args)
    emb_s1_val = get_or_compute_embeddings(
        s1_val, "s1_val", f"{CACHE_DIR}/emb_s1_val.pkl", args)
    emb_sother = get_or_compute_embeddings(
        s_all, "s_other_train", f"{CACHE_DIR}/emb_s_other_train.pkl", args)

    # ── Hard negative mining
    print("\n[HARD NEGATIVES] Mining training negatives...")
    trn_pairs, trn_labels = build_balanced_training_pairs(
        cands_trn, gt_dict,
        hard_neg_ratio=args.hard_neg_ratio,
    )

    # Build feature matrices from selected pairs (not all candidates)
    trn_cands_selected = {}
    for (s1_eid, cand_eid), label in zip(trn_pairs, trn_labels):
        trn_cands_selected.setdefault(s1_eid, []).append(cand_eid)

    print("\n[FEATURES] Building train features...")
    X_trn, y_trn, _ = build_pair_feature_matrix_v2(
        s1_trn, s_all,
        {k: list(v) for k, v in trn_cands_selected.items()},
        labels={k: gt_dict.get(k, set()) for k in trn_cands_selected},
        embeddings_s1=emb_s1_trn,
        embeddings_other=emb_sother,
    )
    print(f"  Train: {len(X_trn):,} pairs | pos={y_trn.sum():,} | neg={(y_trn==0).sum():,}")

    print("[FEATURES] Building val features...")
    X_val, y_val, val_pair_ids = build_pair_feature_matrix_v2(
        s1_val, s_all,
        {k: list(v) for k, v in cands_val.items()},
        labels={k: gt_dict.get(k, set()) for k in cands_val},
        embeddings_s1=emb_s1_val,
        embeddings_other=emb_sother,
    )
    print(f"  Val: {len(X_val):,} pairs | pos={y_val.sum():,} | neg={(y_val==0).sum():,}")

    # ── Train CatBoost
    print("\n[TRAIN] CatBoost matcher...")
    cb_path = os.path.join(OUTPUT_DIR, "catboost_model.cbm")
    model = train_catboost(
        X_trn, y_trn, X_val, y_val,
        iterations=args.n_estimators,
        learning_rate=args.lr,
        depth=8,
        model_path=cb_path,
    )

    # ── Threshold tuning
    print("\n[THRESHOLD] Tuning on validation...")
    val_gt = {k: gt_dict.get(k, set()) for k in s1_val["entity_id"]}
    val_probs = predict_proba(model, X_val)
    best_t, best_f05 = tune_threshold(val_probs, val_pair_ids, val_gt)

    # Save threshold
    with open(os.path.join(OUTPUT_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": best_t, "val_f05": best_f05}, f)
    print(f"  Val F0.5 = {best_f05:.4f} @ threshold={best_t:.3f}")

    # ── Test prediction
    print("\n[TEST] Running test pipeline...")
    s1_test, s2_test, s3_test = load_test()
    cache_test = os.path.join(CACHE_DIR, "cands_test.pkl")
    if os.path.exists(cache_test) and not getattr(args, "force_reblock", False):
        print(f"  [CACHE] Loading precomputed test candidates from {cache_test}...")
        with open(cache_test, "rb") as f:
            cands_test = pickle.load(f)
    else:
        cands_test = merge_s2_s3_candidates(
            s1_test["entity_id"],
            block_one_source(s1_test, s2_test, "S2-test", **_block_kwargs(args)),
            block_one_source(s1_test, s3_test, "S3-test", **_block_kwargs(args)),
        )
        ensure_all_s1_covered(cands_test, s1_test["entity_id"].tolist())
        valid_s2 = set(s2_test["entity_id"])
        valid_s3 = set(s3_test["entity_id"])
        cands_test = filter_to_valid_ids(cands_test, valid_s2, valid_s3)
        with open(cache_test, "wb") as f:
            pickle.dump(cands_test, f)
        print(f"  [CACHE] Saved test candidates to {cache_test}")
    export_candidate_pairs(cands_test, os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"))

    s_test_all = pd.concat([s2_test, s3_test], ignore_index=True)
    emb_s1_test = get_or_compute_embeddings(
        s1_test, "s1_test", f"{CACHE_DIR}/emb_s1_test.pkl", args)
    emb_stest = get_or_compute_embeddings(
        s_test_all, "s_other_test", f"{CACHE_DIR}/emb_s_other_test.pkl", args)

    print("[FEATURES] Building test features...")
    X_test, _, test_pair_ids = build_pair_feature_matrix_v2(
        s1_test, s_test_all,
        {k: list(v) for k, v in cands_test.items()},
        embeddings_s1=emb_s1_test,
        embeddings_other=emb_stest,
    )

    test_probs = predict_proba(model, X_test)

    scorer = EnsembleScorer()
    test_preds = scorer.predict_with_singleton_rule(test_probs, test_pair_ids, threshold=best_t)
    ensure_all_s1_covered(test_preds, s1_test["entity_id"].tolist())
    export_matching_results(test_preds, s1_test["entity_id"].tolist(),
                            os.path.join(OUTPUT_DIR, "matching_results.tsv"))

    print(f"\n{'='*60}")
    print(f"CATBOOST PIPELINE DONE")
    print(f"  Val F0.5 = {best_f05:.4f}")
    print(f"  matching_results.tsv → upload to leaderboard")
    print(f"  candidate_pairs.tsv  → include in submission zip")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# MODE: Cross-encoder fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_encoder_training(args):
    """Fine-tune DeBERTa cross-encoder on training pairs."""
    from src.matching.cross_encoder import train_cross_encoder

    s1_train, s2_train, s3_train, gt_train = load_train()
    gt_dict = parse_ground_truth(gt_train)

    s1_val = s1_train.sample(frac=args.val_frac, random_state=42)
    s1_trn = s1_train.drop(s1_val.index)
    if args.max_train_samples > 0:
        s1_trn = s1_trn.sample(min(args.max_train_samples, len(s1_trn)), random_state=42)
    s1_trn = s1_trn.reset_index(drop=True)

    print("\n[BLOCKING] For cross-encoder training data...")
    cands_trn = merge_s2_s3_candidates(
        s1_trn["entity_id"],
        block_one_source(s1_trn, s2_train, "S2-train", **_block_kwargs(args)),
        block_one_source(s1_trn, s3_train, "S3-train", **_block_kwargs(args)),
    )
    cands_val = merge_s2_s3_candidates(
        s1_val["entity_id"],
        block_one_source(s1_val, s2_train, "S2-val", **_block_kwargs(args)),
        block_one_source(s1_val, s3_train, "S3-val", **_block_kwargs(args)),
    )

    s_all = pd.concat([s2_train, s3_train], ignore_index=True)
    s1_map = {r["entity_id"]: r.to_dict() for _, r in s1_trn.iterrows()}
    so_map = {r["entity_id"]: r.to_dict() for _, r in s_all.iterrows()}

    # Build training text pairs
    print("\n[CE] Building text pairs for training...")
    trn_pairs_raw, trn_labels_raw = build_balanced_training_pairs(
        cands_trn, gt_dict, hard_neg_ratio=args.hard_neg_ratio)

    train_pairs = []
    train_labels = []
    for (s1_eid, cand_eid), label in zip(trn_pairs_raw, trn_labels_raw):
        s1r = s1_map.get(s1_eid, {})
        s2r = so_map.get(cand_eid, {})
        if not s2r:
            continue
        train_pairs.append((
            str(s1r.get("business_name", "")),
            str(s1r.get("business_address", "")),
            str(s2r.get("business_name", "")),
            str(s2r.get("business_address", "")),
        ))
        train_labels.append(label)

    # Build val pairs
    s1_val_map = {r["entity_id"]: r.to_dict() for _, r in s1_val.iterrows()}
    val_pairs_text = []
    val_labels_text = []
    for s1_eid, cand_list in list(cands_val.items())[:5000]:  # cap for speed
        s1r = s1_val_map.get(s1_eid, {})
        true_m = gt_dict.get(s1_eid, set())
        for cand_eid in cand_list[:20]:
            s2r = so_map.get(cand_eid, {})
            if not s2r:
                continue
            val_pairs_text.append((
                str(s1r.get("business_name", "")),
                str(s1r.get("business_address", "")),
                str(s2r.get("business_name", "")),
                str(s2r.get("business_address", "")),
            ))
            val_labels_text.append(int(cand_eid in true_m))

    print(f"  CE train pairs: {len(train_pairs):,} | val pairs: {len(val_pairs_text):,}")

    train_cross_encoder(
        train_pairs, train_labels,
        val_pairs=val_pairs_text, val_labels=val_labels_text,
        output_dir=os.path.join(OUTPUT_DIR, "cross_encoder"),
        num_epochs=args.ce_epochs,
        batch_size=args.ce_batch_size,
        fp16=(args.device == "cuda"),
    )
    print("\nCross-encoder training complete. Run --mode full to use it in ensemble.")


# ─────────────────────────────────────────────────────────────────────────────
# MODE: Block-only benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_block_only(args):
    from src.evaluation.evaluate_blocking import run_benchmark
    run_benchmark(
        train_dir=TRAIN_DIR,
        val_frac=args.val_frac,
        max_val_samples=getattr(args, "max_val_samples", 10000),
        top_k_tfidf_name=args.top_k_name,
        top_k_tfidf_nameaddr=args.top_k_addr,
        sn_window=args.sn_window,
        output_csv="reports/blocking_benchmark.csv",
        skip_dense=not args.use_dense,
    )


def run_generate_candidates(args):
    """
    Generate candidates for the full test set (for candidate_pairs.tsv submission)
    and/or train set (for fast cached training and recall evaluation).
    """
    print("\n" + "=" * 60)
    print("STAGE: CANDIDATE GENERATION FOR FULL DATASET")
    print("=" * 60)

    # 1. Test set candidates (produces candidate_pairs.tsv)
    if args.target_split in ("test", "both"):
        print("\n[TEST CANDIDATES] Loading test data and running multi-pass blocking...")
        s1_test, s2_test, s3_test = load_test()
        cache_test = os.path.join(CACHE_DIR, "cands_test.pkl")

        if os.path.exists(cache_test) and not args.force_reblock:
            print(f"  [CACHE] Found existing test candidates in {cache_test}")
            with open(cache_test, "rb") as f:
                cands_test = pickle.load(f)
        else:
            cands_test = merge_s2_s3_candidates(
                s1_test["entity_id"],
                block_one_source(s1_test, s2_test, "S2-test", **_block_kwargs(args)),
                block_one_source(s1_test, s3_test, "S3-test", **_block_kwargs(args)),
            )
            ensure_all_s1_covered(cands_test, s1_test["entity_id"].tolist())
            valid_s2 = set(s2_test["entity_id"])
            valid_s3 = set(s3_test["entity_id"])
            cands_test = filter_to_valid_ids(cands_test, valid_s2, valid_s3)

            with open(cache_test, "wb") as f:
                pickle.dump(cands_test, f)
            print(f"  [CACHE] Saved test candidates to {cache_test}")

        out_cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
        export_candidate_pairs(cands_test, out_cand_path)
        stats = candidate_stats(cands_test)
        print(f"\n[TEST CANDIDATE STATS]")
        print(f"  Total S1 entities:     {stats['n_s1_entities']:,}")
        print(f"  Total Candidate Pairs: {stats['total_pairs']:,}")
        print(f"  Avg candidates/S1:     {stats['avg_candidates']:.2f}")
        print(f"  Median candidates:     {stats['median_candidates']:.0f}")
        print(f"  Max candidates:        {stats['max_candidates']}")
        print(f"  Singletons (no cands): {stats['n_singletons']:,}")
        print(f"  -> File ready for submission: {out_cand_path}")

    # 2. Train set candidates (for training & evaluation)
    if args.target_split in ("train", "both"):
        print("\n[TRAIN CANDIDATES] Loading train data...")
        s1_train, s2_train, s3_train, gt_train = load_train()
        gt_dict = parse_ground_truth(gt_train)

        if args.max_train_samples > 0 and args.max_train_samples < len(s1_train):
            print(f"  Sampling train S1 to {args.max_train_samples:,} records...")
            s1_train = s1_train.sample(args.max_train_samples, random_state=42).reset_index(drop=True)

        cache_train = os.path.join(CACHE_DIR, f"cands_train_{len(s1_train)}.pkl")
        if os.path.exists(cache_train) and not args.force_reblock:
            print(f"  [CACHE] Found existing train candidates in {cache_train}")
            with open(cache_train, "rb") as f:
                cands_train = pickle.load(f)
        else:
            cands_train = merge_s2_s3_candidates(
                s1_train["entity_id"],
                block_one_source(s1_train, s2_train, "S2-train", **_block_kwargs(args)),
                block_one_source(s1_train, s3_train, "S3-train", **_block_kwargs(args)),
            )
            ensure_all_s1_covered(cands_train, s1_train["entity_id"].tolist())
            with open(cache_train, "wb") as f:
                pickle.dump(cands_train, f)
            print(f"  [CACHE] Saved train candidates to {cache_train}")

        # Compute recall against ground truth
        from src.evaluation.evaluate_blocking import compute_blocking_recall
        recall, rec, tot, miss = compute_blocking_recall(cands_train, gt_dict)
        stats = candidate_stats(cands_train)
        print(f"\n[TRAIN BLOCKING RECALL & STATS]")
        print(f"  Blocking Recall:       {recall * 100:.2f}% ({rec:,}/{tot:,} true matches recovered)")
        print(f"  Missed true matches:   {miss:,}")
        print(f"  Avg candidates/S1:     {stats['avg_candidates']:.2f}")
        print(f"  Median candidates:     {stats['median_candidates']:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _block_kwargs(args):
    return dict(
        top_k_name=args.top_k_name,
        top_k_addr=args.top_k_addr,
        sn_window=args.sn_window,
        max_candidates=args.max_candidates,
        use_dense=args.use_dense,
        top_k_dense=args.top_k_dense,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIR #1 Entity Resolution Pipeline v2")

    parser.add_argument("--mode", choices=[
        "block-only",            # Just benchmark blocking recall
        "generate-candidates",   # Generate & export candidates for full dataset
        "train-catboost",        # CatBoost pipeline (fast, good baseline)
        "train-ce",              # Fine-tune DeBERTa cross-encoder
        "full",                  # All models + ensemble
    ], default="block-only")

    # Data
    parser.add_argument("--val-frac",          type=float, default=0.05)
    parser.add_argument("--max-val-samples",   type=int,   default=10000,
                        help="Cap validation samples for benchmark (prevents OOM on Colab)")
    parser.add_argument("--max-train-samples", type=int,   default=50000)
    parser.add_argument("--target-split",      choices=["test", "train", "both"], default="both",
                        help="Target split for candidate generation")
    parser.add_argument("--force-reblock",     action="store_true",
                        help="Force re-running blocking even if cached")

    # Blocking
    parser.add_argument("--top-k-name",    type=int,   default=30)
    parser.add_argument("--top-k-addr",    type=int,   default=40)
    parser.add_argument("--sn-window",     type=int,   default=10)
    parser.add_argument("--max-candidates",type=int,   default=150)
    parser.add_argument("--use-dense",     action="store_true")
    parser.add_argument("--top-k-dense",   type=int,   default=30)

    # Training
    parser.add_argument("--n-estimators",    type=int,   default=2000)
    parser.add_argument("--lr",              type=float, default=0.03)
    parser.add_argument("--hard-neg-ratio",  type=int,   default=3)
    parser.add_argument("--use-embeddings",  action="store_true",
                        help="Compute/use BGE-M3 embeddings in features")

    # Cross-encoder
    parser.add_argument("--ce-epochs",      type=int,   default=3)
    parser.add_argument("--ce-batch-size",  type=int,   default=16)

    # Hardware
    parser.add_argument("--device", default="cuda",
                        help="cuda or cpu")

    args = parser.parse_args()

    if args.mode == "block-only":
        run_block_only(args)
    elif args.mode == "generate-candidates":
        run_generate_candidates(args)
    elif args.mode == "train-catboost":
        run_catboost_pipeline(args)
    elif args.mode == "train-ce":
        run_cross_encoder_training(args)
    elif args.mode == "full":
        print("Full ensemble mode: run train-catboost + train-ce first, then combine.")
        print("See notebooks/04_ensemble_final.ipynb for the full ensemble workflow.")
        run_catboost_pipeline(args)
