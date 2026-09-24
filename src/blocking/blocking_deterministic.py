"""
blocking_deterministic.py — Layer 1: Deterministic blocking rules.

Generates candidate pairs using multiple blocking keys (country+name prefix,
address prefix, sorted tokens, etc.) and returns their UNION.

For a 2M × 5M dataset this runs in seconds using dict-based inverted indexes.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, DefaultDict

from src.data.normalize import build_blocking_keys, normalize_name, normalize_address, normalize_country


def build_inverted_index(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build inverted index (key -> list of entity_ids)."""
    """
    Build an inverted index: blocking_key → [entity_id, ...]
    for a source-2 or source-3 dataframe.
    """
    idx: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        keys = build_blocking_keys(row)
        eid  = row["entity_id"]
        for k in keys:
            idx[k].append(eid)
    return dict(idx)


def build_inverted_index_fast(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Vectorized inverted index builder. Computes all 5 blocking keys with
    pandas string operations instead of per-row Python function calls.
    ~5x faster than the loop version at 5M rows.
    """
    _legal = (
        r"\b(pvt\.?\s*ltd\.?|private\s+limited|private\s+ltd\.?|p\.?\s*ltd\.?|"
        r"llp|llc|inc\.?|corp\.?|corporation|limited|ltd\.?|co\.?|company|"
        r"enterprises?|industries|industry|group|holdings?|trading|traders?|"
        r"distributors?|solutions?|technologies|technology|tech|services?|"
        r"international|intl\.?|s\.a\.s|s\.a\.|sarl|sas|eurl|srl|"
        r"proprietorship|proprietor|prop\.?|& sons|and sons|brothers|bros\.?)\b"
    )

    # Vectorized name normalization
    names_v = (
        df["business_name"].fillna("").astype(str)
        .str.lower()
        .str.replace(_legal, " ", regex=True)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    # Vectorized address normalization (light)
    addrs_v = (
        df["business_address"].fillna("").astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    ctry_v = df["country"].fillna("").astype(str).str.lower().str.strip()

    name_compact = names_v.str.replace(" ", "", regex=False)
    addr_compact = addrs_v.str.replace(" ", "", regex=False)

    eids  = df["entity_id"].tolist()
    key_a = (ctry_v + "|" + name_compact.str[:4]).tolist()
    key_b = (ctry_v + "|" + addr_compact.str[:5]).tolist()
    key_c = name_compact.str[:6].tolist()
    key_d = names_v.str.split().apply(
        lambda t: " ".join(sorted(t)[:4]) if isinstance(t, list) and t else ""
    ).tolist()
    key_e = (name_compact.str[:3] + "|" + addr_compact.str[:3]).tolist()

    idx: Dict[str, List[str]] = defaultdict(list)
    for i, eid in enumerate(eids):
        ka = key_a[i]
        if len(ka) > 3:  idx[ka].append(eid)
        kb = key_b[i]
        if len(kb) > 3:  idx[kb].append(eid)
        kc = key_c[i]
        if len(kc) >= 3: idx[kc].append(eid)
        kd = key_d[i]
        if kd:           idx[kd].append(eid)
        ke = key_e[i]
        if len(ke) > 4:  idx[ke].append(eid)

    return dict(idx)


def query_deterministic(
    s1_row: dict,
    idx: Dict[str, List[str]],
    max_per_key: int = 50,
) -> Set[str]:
    """
    For a single Source-1 record, retrieve all candidates from the
    inverted index by taking the UNION across all blocking keys.

    max_per_key: cap per individual key to avoid one dominant key flooding.
    """
    from src.data.normalize import (
        normalize_name, normalize_address, normalize_country,
        key_country_nameprefix, key_country_addr_prefix,
        key_name_prefix, key_name_tokens_sorted, key_name_addr_prefix,
    )

    name    = normalize_name(str(s1_row.get("business_name", "")))
    addr    = normalize_address(str(s1_row.get("business_address", "")))
    country = normalize_country(str(s1_row.get("country", "")))

    candidates: Set[str] = set()

    for key_fn, args in [
        (key_country_nameprefix, (country, name, 4)),
        (key_country_addr_prefix, (country, addr, 5)),
        (key_name_prefix,         (name, 6)),
        (key_name_tokens_sorted,  (name,)),
        (key_name_addr_prefix,    (name, addr, 3, 3)),
    ]:
        k = key_fn(*args)
        hits = idx.get(k, [])
        candidates.update(hits[:max_per_key])

    return candidates


def run_deterministic_blocking(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    max_per_key: int = 50,
    verbose: bool = True,
) -> Dict[str, Set[str]]:
    """
    Full deterministic blocking pass.
    Returns: {s1_entity_id → set of candidate entity_ids}
    """
    if verbose:
        print(f"  Building inverted index for {len(s_other):,} records...")
    idx = build_inverted_index_fast(s_other)
    if verbose:
        print(f"  Index has {len(idx):,} unique keys")

    result: Dict[str, Set[str]] = {}
    names    = s1["business_name"].fillna("").astype(str).tolist()
    addrs    = s1["business_address"].fillna("").astype(str).tolist()
    countries= s1["country"].fillna("").astype(str).tolist()
    eids     = s1["entity_id"].tolist()

    for i, eid in enumerate(eids):
        row = {
            "business_name": names[i],
            "business_address": addrs[i],
            "country": countries[i],
        }
        result[eid] = query_deterministic(row, idx, max_per_key)

    if verbose:
        sizes = [len(v) for v in result.values()]
        print(f"  Deterministic candidates: avg={np.mean(sizes):.1f}, "
              f"median={np.median(sizes):.0f}, max={max(sizes)}")
    return result
