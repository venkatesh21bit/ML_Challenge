"""
features.py — Pairwise feature engineering for the matching model.

For each (S1, candidate) pair, computes:
  - Name similarity features (TF-IDF cosine, Jaccard, token overlap, Levenshtein)
  - Address similarity features
  - Country match
  - Combined text similarity
  - Length ratios

Features are intentionally fast to compute (no neural models here).
The matching model (LightGBM/CatBoost) learns which features matter.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from src.data.normalize import normalize_name, normalize_address, normalize_country


# ─────────────────────────────────────────────────────────────────────────────
# String similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_tokens(a: str, b: str) -> float:
    """Jaccard similarity of token sets."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def token_overlap(a: str, b: str) -> float:
    """Symmetric token overlap (shorter/longer)."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def common_prefix_len(a: str, b: str) -> int:
    """Length of common prefix (character level)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def levenshtein(a: str, b: str) -> int:
    """Standard edit distance (pure Python, capped for speed)."""
    a, b = a[:60], b[:60]   # cap to avoid O(n²) on long strings
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein / max_len → [0, 1] where 0=identical."""
    if not a and not b:
        return 0.0
    d = levenshtein(a, b)
    return d / max(len(a), len(b))


def char_ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """Character n-gram overlap (Jaccard on n-gram sets)."""
    def ngrams(s, n):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    sa, sb = ngrams(a, n), ngrams(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Name features
    "name_jaccard",
    "name_token_overlap",
    "name_edit_distance_norm",
    "name_char3_overlap",
    "name_char4_overlap",
    "name_prefix_6",
    "name_prefix_4",
    "name_len_ratio",
    "name_common_tokens",
    # Address features
    "addr_jaccard",
    "addr_token_overlap",
    "addr_edit_distance_norm",
    "addr_char3_overlap",
    "addr_len_ratio",
    # Country features
    "country_exact_match",
    # Cross features
    "name_addr_jaccard",
    "name_in_addr",
    # Length features
    "s1_name_len",
    "s2_name_len",
    "s1_addr_len",
    "s2_addr_len",
]


def compute_pair_features(
    s1_row: dict,
    s2_row: dict,
) -> np.ndarray:
    """
    Compute feature vector for a single (S1, candidate) pair.
    All inputs are raw (un-normalized) dicts with business_name, business_address, country.
    """
    n1  = normalize_name(str(s1_row.get("business_name", "")))
    n2  = normalize_name(str(s2_row.get("business_name", "")))
    a1  = normalize_address(str(s1_row.get("business_address", "")))
    a2  = normalize_address(str(s2_row.get("business_address", "")))
    c1  = normalize_country(str(s1_row.get("country", "")))
    c2  = normalize_country(str(s2_row.get("country", "")))

    feats = [
        # Name
        jaccard_tokens(n1, n2),
        token_overlap(n1, n2),
        normalized_edit_distance(n1, n2),
        char_ngram_overlap(n1, n2, 3),
        char_ngram_overlap(n1, n2, 4),
        float(n1[:6] == n2[:6]),
        float(n1[:4] == n2[:4]),
        len(n1) / (len(n2) + 1e-6),
        float(len(set(n1.split()) & set(n2.split()))),
        # Address
        jaccard_tokens(a1, a2),
        token_overlap(a1, a2),
        normalized_edit_distance(a1, a2),
        char_ngram_overlap(a1, a2, 3),
        len(a1) / (len(a2) + 1e-6),
        # Country
        float(c1 == c2),
        # Cross
        jaccard_tokens(n1 + " " + a1, n2 + " " + a2),
        float(n1[:6] in a2 or n2[:6] in a1),
        # Lengths
        float(len(n1)),
        float(len(n2)),
        float(len(a1)),
        float(len(a2)),
    ]
    return np.array(feats, dtype=np.float32)


def build_pair_feature_matrix(
    s1_df: pd.DataFrame,
    s_other_df: pd.DataFrame,
    candidates: Dict[str, List[str]],
    labels: Optional[Dict[str, set]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Build feature matrix X for all candidate pairs.
    Returns: X (N_pairs, n_features), y (N_pairs,) or None, pair_ids list
    """
    # Build lookup maps
    s1_map = {r["entity_id"]: r.to_dict() for _, r in s1_df.iterrows()}
    so_map = {r["entity_id"]: r.to_dict() for _, r in s_other_df.iterrows()}

    X_rows   = []
    y_rows   = []
    pair_ids = []

    for s1_eid, cand_list in candidates.items():
        s1_row = s1_map.get(s1_eid, {})
        for cand_eid in cand_list:
            s2_row = so_map.get(cand_eid, {})
            if not s2_row:
                continue
            feats = compute_pair_features(s1_row, s2_row)
            X_rows.append(feats)
            pair_ids.append((s1_eid, cand_eid))
            if labels is not None:
                y_rows.append(int(cand_eid in labels.get(s1_eid, set())))

    X = np.vstack(X_rows) if X_rows else np.empty((0, len(FEATURE_NAMES)))
    y = np.array(y_rows, dtype=np.int8) if labels else np.array([])
    return X, y, pair_ids
