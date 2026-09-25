"""
blocking_deterministic.py — Layer 1: Deterministic blocking rules.

Uses UNIFIED vectorized normalization for BOTH index and query so keys
always match. Longer prefixes (8-10 chars) keep buckets small and specific.

Key design decisions from data analysis (15 sample true pairs):
  - 13/15 pairs share ≥7 chars of compact name prefix
  - Word-order swaps ("Sarasva India" vs "limited sarasva india") handled by
    sorted-token key after legal suffix stripping
  - Domain names ("cardiologymetrocare.com") handled by 10-char prefix match
  - No max_per_key cap (avg bucket size ~4, cap was cutting true matches)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared vectorized normalization (used for BOTH index build and query)
# ─────────────────────────────────────────────────────────────────────────────

_LEGAL_PAT = (
    r"\b(pvt\.?\s*ltd\.?|private\s+limited|private\s+ltd\.?|p\.?\s*ltd\.?|"
    r"llp|llc|inc\.?|corp\.?|corporation|limited|ltd\.?|co\.?|company|"
    r"enterprises?|industries|industry|group|holdings?|trading|traders?|"
    r"distributors?|solutions?|technologies|technology|tech|services?|"
    r"international|intl\.?|s\.a\.s|s\.a\.|sarl|sas|eurl|srl|"
    r"proprietorship|proprietor|prop\.?|& sons|and sons|brothers|bros\.?|"
    r"pllc|plc|associates?|association|foundation|trust|school|college|"
    r"hospital|clinic|center|centre)\b"
)


def _vec_norm_name(s: pd.Series) -> pd.Series:
    """Vectorized name normalization: lower → strip legal → strip punct → collapse."""
    return (
        s.fillna("").astype(str)
        .str.lower()
        .str.replace(_LEGAL_PAT, " ", regex=True)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _vec_norm_addr(s: pd.Series) -> pd.Series:
    """Vectorized address normalization: lower → strip punct → collapse."""
    return (
        s.fillna("").astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _vec_norm_country(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.strip()


def _compute_keys(df: pd.DataFrame) -> Tuple[List, List, List, List, List, List]:
    """
    Compute all 6 blocking keys for a DataFrame, fully vectorized.
    Returns 6 lists: key_a, key_b, key_c, key_d, key_e, key_f
    """
    names_v = _vec_norm_name(df["business_name"])
    addrs_v = _vec_norm_addr(df["business_address"])
    ctry_v  = _vec_norm_country(df["country"])

    nc = names_v.str.replace(" ", "", regex=False)   # compact name
    ac = addrs_v.str.replace(" ", "", regex=False)   # compact addr

    # Key A: country|name_compact[:8]  (long enough to avoid huge buckets)
    key_a = (ctry_v + "|" + nc.str[:8]).tolist()

    # Key B: country|addr_compact[:8]
    key_b = (ctry_v + "|" + ac.str[:8]).tolist()

    # Key C: name_compact[:10]  (handles cross-country same-name)
    key_c = nc.str[:10].tolist()

    # Key D: sorted name tokens (handles word-order transpositions)
    #   After legal strip: "Sarasva India" and "limited Sarasva India" both → "india sarasva"
    key_d = names_v.str.split().apply(
        lambda t: " ".join(sorted(t)[:5]) if isinstance(t, list) and t else ""
    ).tolist()

    # Key E: name_compact[:6]|addr_compact[:6] (combined discriminator)
    key_e = (nc.str[:6] + "|" + ac.str[:6]).tolist()

    # Key F: name_compact[:5]|country (short name + country, catches abbreviations)
    key_f = (nc.str[:5] + "|" + ctry_v).tolist()

    return key_a, key_b, key_c, key_d, key_e, key_f


# ─────────────────────────────────────────────────────────────────────────────
# Index build
# ─────────────────────────────────────────────────────────────────────────────

def build_inverted_index_fast(df: pd.DataFrame, max_bucket_size: int = 150) -> Dict[str, List[str]]:
    """
    Build inverted index from S2/S3 DataFrame using vectorized key computation.
    Skips mega-buckets (> max_bucket_size) to avoid noisy stop-word collisions.
    Returns: {blocking_key → [entity_id, ...]}
    """
    eids = df["entity_id"].tolist()
    key_a, key_b, key_c, key_d, key_e, key_f = _compute_keys(df)

    idx: Dict[str, List[str]] = defaultdict(list)
    for i, eid in enumerate(eids):
        ka = key_a[i]; (idx[ka].append(eid) if len(ka) > 5 else None)
        kb = key_b[i]; (idx[kb].append(eid) if len(kb) > 8 else None)  # min 8 chars for address
        kc = key_c[i]; (idx[kc].append(eid) if len(kc) >= 5 else None)
        kd = key_d[i]; (idx[kd].append(eid) if len(kd) >= 4 else None)
        ke = key_e[i]; (idx[ke].append(eid) if len(ke) > 6 else None)
        kf = key_f[i]; (idx[kf].append(eid) if len(kf) > 6 else None)  # min 6 chars for short name+country

    # Filter out mega-buckets (stop-words / generic terms like 'hotel' or 'near bus stand')
    filtered_idx = {k: v for k, v in idx.items() if len(v) <= max_bucket_size}
    del idx
    return filtered_idx


# ─────────────────────────────────────────────────────────────────────────────
# Batch query (vectorized — processes all S1 records at once)
# ─────────────────────────────────────────────────────────────────────────────

def run_deterministic_blocking(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    max_per_key: int = 40,        # max candidates from any single key
    max_cands_per_s1: int = 80,   # hard cap per entity for deterministic pass
    verbose: bool = True,
) -> Dict[str, Set[str]]:
    """
    Full deterministic blocking pass with unified vectorized normalization.
    Returns: {s1_entity_id → set of candidate entity_ids}
    """
    import gc
    if verbose:
        print(f"  Building inverted index for {len(s_other):,} records...")
    idx = build_inverted_index_fast(s_other)
    if verbose:
        print(f"  Index has {len(idx):,} unique keys, "
              f"avg bucket={sum(len(v) for v in idx.values())/len(idx):.1f}")

    # Compute S1 keys with SAME vectorized normalization
    s1_eids = s1["entity_id"].tolist()
    ka1, kb1, kc1, kd1, ke1, kf1 = _compute_keys(s1)

    result: Dict[str, Set[str]] = {}
    for i, eid in enumerate(s1_eids):
        cands: Set[str] = set()
        for k in [ka1[i], kb1[i], kc1[i], kd1[i], ke1[i], kf1[i]]:
            hits = idx.get(k, [])
            if hits:
                cands.update(hits[:max_per_key])
                if len(cands) >= max_cands_per_s1:
                    break
        result[eid] = cands

    del idx
    gc.collect()

    if verbose:
        sizes = [len(v) for v in result.values()]
        print(f"  Deterministic candidates: avg={np.mean(sizes):.1f}, "
              f"median={np.median(sizes):.0f}, max={max(sizes)}")
    return result
