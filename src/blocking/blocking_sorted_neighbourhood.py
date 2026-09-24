"""
blocking_sorted_neighbourhood.py — Layer 4: Sorted Neighbourhood fallback.

Handles the case where spelling errors push records apart in exact/prefix
keys AND in TF-IDF/dense space. Sorting by normalized name + sliding window
naturally groups lexicographically similar records.

Example: "Samzung Electronics" stays near "Samsung Electronics" after sorting.

At 5M records, this is done efficiently using a pre-sorted numpy array.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, List

from src.data.normalize import normalize_name, normalize_country


def _build_sort_key(row) -> str:
    """Composite sort key: country + normalized_name (no spaces)."""
    country = normalize_country(str(row.get("country", "")))
    name    = normalize_name(str(row.get("business_name", "")))
    return country + "|" + name.replace(" ", "")


def run_sorted_neighbourhood(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    window_size: int = 10,
    verbose: bool = True,
) -> Dict[str, Set[str]]:
    """
    Sorted Neighbourhood blocking.

    Algorithm:
      1. Merge S1 + S_other into one pool with sort keys.
      2. Sort by sort key.
      3. For every S1 record at position i, add all S2/S3 records within
         [i - window_size, i + window_size] as candidates.

    Returns: {s1_entity_id → set of candidate entity_ids}
    """
    if verbose:
        print(f"  Building sort keys for {len(s1)+len(s_other):,} records (window={window_size})...")

    # Build sort keys — vectorized (avoid apply() at 5M scale)
    s1_rows    = s1[["entity_id", "business_name", "country"]].copy()
    other_rows = s_other[["entity_id", "business_name", "country"]].copy()

    def _make_sort_keys_vec(df: pd.DataFrame) -> pd.Series:
        countries = df["country"].fillna("").astype(str).str.lower().str.strip()
        names = (
            df["business_name"].fillna("").astype(str)
            .str.lower()
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", "", regex=True)
        )
        return countries + "|" + names

    s1_rows["_sort_key"]    = _make_sort_keys_vec(s1_rows)
    other_rows["_sort_key"] = _make_sort_keys_vec(other_rows)
    s1_rows["_is_s1"]       = True
    other_rows["_is_s1"]    = False

    combined = pd.concat([s1_rows, other_rows], ignore_index=True)
    combined.sort_values("_sort_key", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    is_s1  = combined["_is_s1"].values
    eids   = combined["entity_id"].values

    if verbose:
        print(f"  Sliding window over {len(combined):,} sorted records...")

    result: Dict[str, Set[str]] = {}

    for i in range(len(combined)):
        if not is_s1[i]:
            continue

        s1_eid = eids[i]
        candidates: Set[str] = set()
        lo = max(0, i - window_size)
        hi = min(len(combined), i + window_size + 1)

        for j in range(lo, hi):
            if not is_s1[j]:
                candidates.add(eids[j])

        result[s1_eid] = candidates

    if verbose:
        sizes = [len(v) for v in result.values()]
        print(f"  Sorted Neighbourhood candidates: avg={np.mean(sizes):.1f}, "
              f"median={np.median(sizes):.0f}, max={max(sizes)}")
    return result
