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

# ── Honorific / generic business prefix patterns ────────────────────────────
_HONORIFIC_PAT = (
    r"^(shri|sri|shree|m/s|ms|dr|doctor|prof|the)\b\s*"
)

# ── Legal suffix patterns ────────────────────────────────────────────────────
_LEGAL_PAT = (
    r"\b(pvt\.?\s*ltd\.?|private\s+limited|private\s+ltd\.?|p\.?\s*ltd\.?|"
    r"llp|llc|inc\.?|corp\.?|corporation|limited|ltd\.?|co\.?|company|"
    r"enterprises?|industries|industry|group|holdings?|trading|traders?|"
    r"distributors?|solutions?|technologies|technology|tech|services?|"
    r"international|intl\.?|plc|gmbh|s\.a\.s|s\.a\.|sarl|sas|eurl|srl|"
    r"proprietorship|proprietor|prop\.?|& sons|and sons|brothers|bros\.?|"
    r"pllc|associates?|association|foundation|trust)\b"
)

# Common street abbreviations
_STREET_ABBREV = {
    r"\bct\b": "court",
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bdr\b": "drive",
    r"\bpl\b": "place",
    r"\bflr\b|\bfl\b": "floor",
    r"\bave\b|\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bhno\b": "house",
    r"\bapt\b": "apartment",
}


def _vec_norm_name(s: pd.Series) -> pd.Series:
    """Vectorized name normalization: strip honorifics → strip legal → collapse."""
    res = s.fillna("").astype(str).str.lower()
    res = res.str.replace(_HONORIFIC_PAT, "", regex=True)
    res = res.str.replace(_LEGAL_PAT, " ", regex=True)
    res = res.str.replace(r"[^\w\s]", " ", regex=True)
    res = res.str.replace(r"\s+", " ", regex=True)
    return res.str.strip()


def _vec_norm_addr(s: pd.Series) -> pd.Series:
    """Vectorized address normalization: expand abbreviations → strip punct."""
    res = s.fillna("").astype(str).str.lower()
    for pat, rep in _STREET_ABBREV.items():
        res = res.str.replace(pat, rep, regex=True)
    res = res.str.replace(r"[^\w\s]", " ", regex=True)
    res = res.str.replace(r"\s+", " ", regex=True)
    return res.str.strip()


def _vec_norm_country(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.strip()


def _compute_keys(df: pd.DataFrame) -> Tuple[List, List, List, List, List, List]:
    """
    Compute 6 high-precision deterministic blocking keys:
      1. key_a: country|compact_name[:8]
      2. key_b: country|first_significant_name_token
      3. key_c: country|sorted_tokens[:3] (handles word transpositions)
      4. key_d: country|house_number|street_token (anchors exact location)
      5. key_e: country|pin_code (5-6 digits)
      6. key_f: country|compact_name[:5] (handles short names / abbreviations)
    """
    names_v = _vec_norm_name(df["business_name"])
    addrs_v = _vec_norm_addr(df["business_address"])
    ctry_v  = _vec_norm_country(df["country"])

    nc = names_v.str.replace(" ", "", regex=False)   # compact name
    tokens_v = names_v.str.split()

    # Key A: country|name_compact[:8]
    key_a = (ctry_v + "|" + nc.str[:8]).tolist()

    # Key B: country|first_significant_token
    first_tokens = tokens_v.apply(lambda t: t[0] if isinstance(t, list) and len(t) > 0 and len(t[0]) >= 3 else "")
    key_b = (ctry_v + "|first_" + first_tokens).tolist()

    # Key C: country|sorted_tokens[:3]
    sorted_tokens = tokens_v.apply(lambda t: " ".join(sorted(t[:3])) if isinstance(t, list) and len(t) >= 2 else "")
    key_c = (ctry_v + "|sorted_" + sorted_tokens).tolist()

    # Key D: country|house_number|street_word
    house_nums = addrs_v.str.extract(r"\b(\d{1,6})\b")[0].fillna("")
    street_words = addrs_v.str.extract(r"\b([a-z]{4,})\b")[0].fillna("")
    has_street_anchor = (house_nums != "") & (street_words != "")
    key_d = np.where(has_street_anchor, ctry_v + "|" + house_nums + "|" + street_words, "").tolist()

    # Key E: country|postal_code (5 or 6 digits)
    pins = addrs_v.str.extract(r"\b(\d{5,6})\b")[0].fillna("")
    key_e = np.where(pins != "", ctry_v + "|pin_" + pins, "").tolist()

    # Key F: country|name_compact[:5] (fallback for short acronyms)
    key_f = (ctry_v + "|" + nc.str[:5]).tolist()

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
