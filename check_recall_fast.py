import sys
import time
import polars as pl

gt_path = 'datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv'
cand_path = 'datasets/candidate data/cand_train.parquet'

print(f"Loading Ground Truth from {gt_path}...", flush=True)
t0 = time.time()
gt_raw = pl.read_csv(gt_path, separator="\t")
print(f"GT raw rows: {len(gt_raw):,} in {time.time()-t0:.2f}s", flush=True)

# Parse GT into clean (s1, o) pairs
gt_clean = (
    gt_raw.rename({"source1_entity_id": "s1", "matched_entity_ids": "o"})
    .filter(pl.col("o").is_not_null() & (pl.col("o") != "") & (pl.col("o") != "nan"))
    .with_columns(pl.col("o").str.split(","))
    .explode("o")
    .with_columns(pl.col("o").str.strip_chars())
    .filter(pl.col("o") != "")
    .select(["s1", "o"])
    .unique()
)

total_gt_pairs = len(gt_clean)
n_gt_s1 = gt_clean.select("s1").n_unique()
print(f"Total True Match Pairs in GT: {total_gt_pairs:,} across {n_gt_s1:,} S1 entities (in {time.time()-t0:.2f}s)", flush=True)

# Scan candidate parquet
print(f"\nAnalyzing Candidate Parquet {cand_path}...", flush=True)
t1 = time.time()
cand_lazy = pl.scan_parquet(cand_path).select(["s1", "o"])

# Count candidate rows
cand_stats = cand_lazy.select([
    pl.len().alias("total_cands"),
    pl.col("s1").n_unique().alias("unique_s1")
]).collect()

total_cands = cand_stats["total_cands"][0]
unique_cands_s1 = cand_stats["unique_s1"][0]
avg_cands = total_cands / max(1, unique_cands_s1)
print(f"Total Candidate Pairs: {total_cands:,}", flush=True)
print(f"Unique S1 Entities in Candidates: {unique_cands_s1:,}", flush=True)
print(f"Average Candidates per S1: {avg_cands:.2f}", flush=True)

# Compute Recall via inner join
print("\nComputing Candidate Recall...", flush=True)
t2 = time.time()
matches_found = (
    cand_lazy.join(gt_clean.lazy(), on=["s1", "o"], how="inner")
    .select(pl.len())
    .collect()
    .item()
)

recall = matches_found / total_gt_pairs
print("=" * 60, flush=True)
print(f"CANDIDATE BLOCKING RECALL: {recall * 100:.2f}%", flush=True)
print(f"True Matches Recovered:   {matches_found:,} / {total_gt_pairs:,}", flush=True)
print(f"True Matches Missed:      {total_gt_pairs - matches_found:,}", flush=True)
print(f"Time Taken:               {time.time()-t2:.2f}s", flush=True)
print("=" * 60, flush=True)
