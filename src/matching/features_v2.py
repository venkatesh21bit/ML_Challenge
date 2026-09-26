"""
features_v2.py — Expanded pairwise feature engineering (56 features).

Upgrade over features.py:
  - Jaro-Winkler distance (via jellyfish if available, else approximation)
  - RapidFuzz token_sort_ratio, partial_ratio
  - Number / digit extraction and matching
  - Phone number similarity
  - Zipcode / PIN-code features
  - City / state token matching
  - BGE-M3 embedding cosine similarity (optional)
  - Name-only, address-only, and combined embedding features
  - Length-normalized edit distance variants
  - Legal suffix presence indicators
  - Initialism detection (ABC ≈ Alpha Beta Corp)
  - First-word exact match

All features are float32 for CatBoost / LightGBM compatibility.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from src.data.normalize import normalize_name, normalize_address, normalize_country

# ─────────────────────────────────────────────────────────────────────────────
# Optional fast libs (fail gracefully)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False

try:
    from rapidfuzz import fuzz as _rfuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


# ─────────────────────────────────────────────────────────────────────────────
# Basic string helpers
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_tokens(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def token_overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


try:
    from rapidfuzz.distance import Levenshtein as _rf_lev
    _HAS_RF_LEV = True
except ImportError:
    _HAS_RF_LEV = False


def levenshtein(a: str, b: str) -> int:
    a, b = a[:80], b[:80]
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def normalized_edit(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if _HAS_RF_LEV:
        return float(_rf_lev.normalized_distance(a, b))
    return levenshtein(a, b) / max(len(a), len(b), 1)


def char_ngram_overlap(a: str, b: str, n: int) -> float:
    def ng(s): return set(s[i:i+n] for i in range(len(s) - n + 1))
    sa, sb = ng(a), ng(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def jaro_winkler(a: str, b: str) -> float:
    if _HAS_JELLYFISH:
        return jellyfish.jaro_winkler_similarity(a, b)
    return max(0.0, 1.0 - normalized_edit(a, b) / 2)


def token_sort_ratio(a: str, b: str) -> float:
    if _HAS_RAPIDFUZZ:
        return _rfuzz.token_sort_ratio(a, b) / 100.0
    sa = " ".join(sorted(a.split()))
    sb = " ".join(sorted(b.split()))
    return max(0.0, 1.0 - normalized_edit(sa, sb))


def partial_ratio(a: str, b: str) -> float:
    if _HAS_RAPIDFUZZ:
        return _rfuzz.partial_ratio(a, b) / 100.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if not short:
        return 0.0
    best = 0.0
    for i in range(len(long) - len(short) + 1):
        sub = long[i:i + len(short)]
        sc = 1.0 - normalized_edit(short, sub)
        if sc > best:
            best = sc
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Number / digit features
# ─────────────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r'\d+')


def extract_numbers(text: str) -> List[str]:
    return _NUM_RE.findall(text)


def number_overlap(a: str, b: str) -> float:
    na, nb = set(extract_numbers(a)), set(extract_numbers(b))
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def has_conflicting_numbers(a: str, b: str) -> float:
    na, nb = set(extract_numbers(a)), set(extract_numbers(b))
    if not na or not nb:
        return 0.0
    return float(len(na & nb) == 0)


# ─────────────────────────────────────────────────────────────────────────────
# Postal / city features
# ─────────────────────────────────────────────────────────────────────────────

_PINCODE_RE = re.compile(r'\b\d{5,6}\b')


def extract_pincode(text: str) -> Optional[str]:
    m = _PINCODE_RE.findall(text)
    return m[0] if m else None


def pincode_match(a: str, b: str) -> float:
    pa, pb = extract_pincode(a), extract_pincode(b)
    if pa is None or pb is None:
        return 0.5   # unknown
    return float(pa == pb)


def first_token_match(a: str, b: str) -> float:
    ta = a.split()
    tb = b.split()
    if not ta or not tb:
        return 0.0
    return float(ta[0] == tb[0])


def last_token_match(a: str, b: str) -> float:
    ta = a.split()
    tb = b.split()
    if not ta or not tb:
        return 0.0
    return float(ta[-1] == tb[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Initialism / abbreviation detector
# ─────────────────────────────────────────────────────────────────────────────

def is_initialism_of(short: str, long_name: str) -> bool:
    if len(short) < 2 or len(short) > 8:
        return False
    tokens = long_name.split()
    if len(tokens) < len(short):
        return False
    initials = "".join(t[0] for t in tokens if t)
    return initials.lower().startswith(short.lower())


def initialism_score(a: str, b: str) -> float:
    return float(is_initialism_of(a, b) or is_initialism_of(b, a))


# ─────────────────────────────────────────────────────────────────────────────
# Legal suffix presence
# ─────────────────────────────────────────────────────────────────────────────

_LEGAL_MARKERS = re.compile(
    r'\b(pvt|private|limited|ltd|llp|llc|inc|corp|sas|sarl|eurl|srl)\b',
    re.IGNORECASE
)


def has_legal_suffix(name: str) -> float:
    return float(bool(_LEGAL_MARKERS.search(name)))


# ─────────────────────────────────────────────────────────────────────────────
# Embedding cosine (called with pre-computed vectors)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
    if v1 is None or v2 is None:
        return 0.0
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ─────────────────────────────────────────────────────────────────────────────
# Feature name registry
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES_V2 = [
    # Name features (20)
    "name_jaccard", "name_token_overlap", "name_edit_norm",
    "name_char3_overlap", "name_char4_overlap", "name_char5_overlap",
    "name_jaro_winkler", "name_token_sort_ratio", "name_partial_ratio",
    "name_prefix6_match", "name_prefix4_match",
    "name_first_token_match", "name_last_token_match",
    "name_len_ratio", "name_common_token_count",
    "name_initialism", "name_has_legal_a", "name_has_legal_b",
    "name_number_overlap", "name_conflicting_numbers",
    # Address features (14)
    "addr_jaccard", "addr_token_overlap", "addr_edit_norm",
    "addr_char3_overlap", "addr_char4_overlap",
    "addr_first_token_match", "addr_last_token_match",
    "addr_len_ratio", "addr_number_overlap", "addr_conflicting_numbers",
    "addr_pincode_match", "addr_pincode_exact",
    "addr_partial_ratio", "addr_token_sort_ratio",
    # Country (2)
    "country_exact_match", "country_both_known",
    # Cross name+address (6)
    "combined_jaccard", "combined_token_overlap", "combined_char3_overlap",
    "name_in_addr_a", "name_in_addr_b", "combined_edit_norm",
    # Embedding features (6)
    "emb_name_cosine", "emb_addr_cosine", "emb_combined_cosine",
    "emb_name_l2", "emb_addr_l2", "emb_combined_l2",
    # Length / structural (8)
    "s1_name_len", "s2_name_len", "s1_addr_len", "s2_addr_len",
    "name_len_diff", "addr_len_diff",
    "s1_name_token_count", "s2_name_token_count",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pair_features_v2(
    s1_row: dict,
    s2_row: dict,
    emb_s1_name: Optional[np.ndarray] = None,
    emb_s2_name: Optional[np.ndarray] = None,
    emb_s1_addr: Optional[np.ndarray] = None,
    emb_s2_addr: Optional[np.ndarray] = None,
    emb_s1_combined: Optional[np.ndarray] = None,
    emb_s2_combined: Optional[np.ndarray] = None,
) -> np.ndarray:
    n1 = normalize_name(str(s1_row.get("business_name", "")))
    n2 = normalize_name(str(s2_row.get("business_name", "")))
    a1 = normalize_address(str(s1_row.get("business_address", "")))
    a2 = normalize_address(str(s2_row.get("business_address", "")))
    c1 = normalize_country(str(s1_row.get("country", "")))
    c2 = normalize_country(str(s2_row.get("country", "")))

    raw_n1 = str(s1_row.get("business_name", ""))
    raw_n2 = str(s2_row.get("business_name", ""))
    combined1 = n1 + " " + a1
    combined2 = n2 + " " + a2

    # Name features
    feats = [
        jaccard_tokens(n1, n2),
        token_overlap(n1, n2),
        normalized_edit(n1, n2),
        char_ngram_overlap(n1, n2, 3),
        char_ngram_overlap(n1, n2, 4),
        char_ngram_overlap(n1, n2, 5),
        jaro_winkler(n1, n2),
        token_sort_ratio(n1, n2),
        partial_ratio(n1, n2),
        float(n1[:6] == n2[:6]) if n1 and n2 else 0.0,
        float(n1[:4] == n2[:4]) if n1 and n2 else 0.0,
        first_token_match(n1, n2),
        last_token_match(n1, n2),
        len(n1) / (len(n2) + 1e-6),
        float(len(set(n1.split()) & set(n2.split()))),
        initialism_score(n1, n2),
        has_legal_suffix(raw_n1),
        has_legal_suffix(raw_n2),
        number_overlap(n1, n2),
        has_conflicting_numbers(n1, n2),
        # Address features
        jaccard_tokens(a1, a2),
        token_overlap(a1, a2),
        normalized_edit(a1, a2),
        char_ngram_overlap(a1, a2, 3),
        char_ngram_overlap(a1, a2, 4),
        first_token_match(a1, a2),
        last_token_match(a1, a2),
        len(a1) / (len(a2) + 1e-6),
        number_overlap(a1, a2),
        has_conflicting_numbers(a1, a2),
        pincode_match(a1, a2),
        float(extract_pincode(a1) is not None and
              extract_pincode(a1) == extract_pincode(a2)),
        partial_ratio(a1, a2),
        token_sort_ratio(a1, a2),
        # Country
        float(c1 == c2),
        float(c1 != "unknown" and c2 != "unknown"),
        # Cross
        jaccard_tokens(combined1, combined2),
        token_overlap(combined1, combined2),
        char_ngram_overlap(combined1, combined2, 3),
        float(n1[:6] in a2) if n1 and a2 else 0.0,
        float(n2[:6] in a1) if n2 and a1 else 0.0,
        normalized_edit(combined1, combined2),
        # Embedding
        cosine_sim(emb_s1_name, emb_s2_name),
        cosine_sim(emb_s1_addr, emb_s2_addr),
        cosine_sim(emb_s1_combined, emb_s2_combined),
        float(np.linalg.norm(emb_s1_name - emb_s2_name))
            if emb_s1_name is not None and emb_s2_name is not None else 0.0,
        float(np.linalg.norm(emb_s1_addr - emb_s2_addr))
            if emb_s1_addr is not None and emb_s2_addr is not None else 0.0,
        float(np.linalg.norm(emb_s1_combined - emb_s2_combined))
            if emb_s1_combined is not None and emb_s2_combined is not None else 0.0,
        # Structural
        float(len(n1)), float(len(n2)),
        float(len(a1)), float(len(a2)),
        float(abs(len(n1) - len(n2))),
        float(abs(len(a1) - len(a2))),
        float(len(n1.split())),
        float(len(n2.split())),
    ]
    return np.array(feats, dtype=np.float32)


def build_pair_feature_matrix_v2(
    s1_df: pd.DataFrame,
    s_other_df: pd.DataFrame,
    candidates: Dict[str, List[str]],
    labels: Optional[Dict[str, set]] = None,
    embeddings_s1: Optional[Dict] = None,
    embeddings_other: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Build expanded feature matrix for all candidate pairs.

    embeddings_s1 / embeddings_other: dicts mapping entity_id to dict of
        {'name': ndarray, 'addr': ndarray, 'combined': ndarray}
    """
    s1_map = {r["entity_id"]: r.to_dict() for _, r in s1_df.iterrows()}
    so_map = {r["entity_id"]: r.to_dict() for _, r in s_other_df.iterrows()}

    X_rows, y_rows, pair_ids = [], [], []

    for s1_eid, cand_list in candidates.items():
        s1_row = s1_map.get(s1_eid, {})
        emb_s1 = (embeddings_s1 or {}).get(s1_eid, {})

        for cand_eid in cand_list:
            s2_row = so_map.get(cand_eid, {})
            if not s2_row:
                continue
            emb_s2 = (embeddings_other or {}).get(cand_eid, {})

            feats = compute_pair_features_v2(
                s1_row, s2_row,
                emb_s1_name=emb_s1.get("name"),
                emb_s2_name=emb_s2.get("name"),
                emb_s1_addr=emb_s1.get("addr"),
                emb_s2_addr=emb_s2.get("addr"),
                emb_s1_combined=emb_s1.get("combined"),
                emb_s2_combined=emb_s2.get("combined"),
            )
            X_rows.append(feats)
            pair_ids.append((s1_eid, cand_eid))

            if labels is not None:
                y_rows.append(int(cand_eid in labels.get(s1_eid, set())))

    X = np.vstack(X_rows) if X_rows else np.empty((0, len(FEATURE_NAMES_V2)))
    y = np.array(y_rows, dtype=np.int8) if labels else np.array([])
    return X, y, pair_ids
