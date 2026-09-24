"""
candidate_union.py — Union, deduplicate, and cap candidate sets.

Takes output dicts from all 4 blocking layers and merges them.
Also provides helpers to:
  - Filter candidates to valid entity IDs (prevents submission errors)
  - Cap max candidates per S1 entity
  - Export candidate_pairs.tsv
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, Optional, List


def union_candidates(
    *candidate_dicts: Dict[str, Set[str]],
    max_candidates: int = 200,
) -> Dict[str, Set[str]]:
    """
    Union all candidate dicts. All dicts must have the same set of S1 keys.

    max_candidates: hard cap per S1 entity (prevents runaway candidate counts).
    If over the cap, we keep deterministic > tfidf > dense > sn priority.
    """
    # Collect all S1 entity IDs
    all_s1_ids: Set[str] = set()
    for d in candidate_dicts:
        all_s1_ids.update(d.keys())

    result: Dict[str, Set[str]] = {}
    for s1_eid in all_s1_ids:
        merged: Set[str] = set()
        for d in candidate_dicts:
            merged |= d.get(s1_eid, set())
        # Remove self (shouldn't happen, but safety)
        merged.discard(s1_eid)
        # Cap
        if len(merged) > max_candidates:
            merged = set(list(merged)[:max_candidates])
        result[s1_eid] = merged

    return result


def filter_to_valid_ids(
    candidates: Dict[str, Set[str]],
    valid_s2_ids: Set[str],
    valid_s3_ids: Set[str],
) -> Dict[str, Set[str]]:
    """Remove candidate IDs that don't exist in the actual source files."""
    valid = valid_s2_ids | valid_s3_ids
    return {
        s1_eid: cands & valid
        for s1_eid, cands in candidates.items()
    }


def ensure_all_s1_covered(
    candidates: Dict[str, Set[str]],
    s1_entity_ids: List[str],
) -> Dict[str, Set[str]]:
    """Ensure every S1 entity has an entry (empty set = singleton)."""
    for eid in s1_entity_ids:
        if eid not in candidates:
            candidates[eid] = set()
    return candidates


def export_candidate_pairs(
    candidates: Dict[str, Set[str]],
    output_path: str,
) -> None:
    """Write candidate_pairs.tsv in the required format."""
    rows = []
    for s1_eid, cands in candidates.items():
        rows.append({
            "source1_entity_id": s1_eid,
            "candidate_entity_ids": ",".join(sorted(cands)),
        })
    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Candidate pairs saved: {output_path} ({len(df):,} rows)")


def candidate_stats(candidates: Dict[str, Set[str]]) -> dict:
    sizes = [len(v) for v in candidates.values()]
    return {
        "n_s1_entities": len(sizes),
        "avg_candidates": float(np.mean(sizes)),
        "median_candidates": float(np.median(sizes)),
        "max_candidates": int(max(sizes)),
        "min_candidates": int(min(sizes)),
        "n_singletons": int(sum(1 for s in sizes if s == 0)),
        "total_pairs": int(sum(sizes)),
    }
