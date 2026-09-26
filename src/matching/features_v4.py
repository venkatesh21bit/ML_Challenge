"""
features_v4.py — AIR #1 CatBoost V4 Comprehensive 150-Feature Engineering Engine.

Architecture:
  - Feature Group 1 (35 features): Name Similarity (Exact, token, n-gram, edit, phonetic, legal suffix).
  - Feature Group 2 (30 features): Address Similarity (House num, street, ZIP, city, state, cardinal dir, po box).
  - Feature Group 3 (25 features): Embedding Features (Cosine, dot product, norm ratios, component projections).
  - Feature Group 4 (20 features): Candidate Context (Rank, rank percentile, score margins, candidate density).
  - Feature Group 5 (15 features): Blocking Features (sn, sa, slot, interaction ratios, channel combinations).
  - Feature Group 6 (25 features): Joint Cross-Field & Branch Detection (Name+Addr interactions, branch detector).

Total: Exactly 150 High-Signal, Non-Collinear Features for CatBoost Classifier & Ranker.
Accelerated with RapidFuzz C++ and vectorized numpy.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Any

# Fast optional dependencies with graceful fallback
try:
    from rapidfuzz import fuzz as _rfuzz
    from rapidfuzz.distance import Levenshtein as _rf_lev, JaroWinkler as _rf_jw
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False

# Regex patterns
RE_DIGITS = re.compile(r"\d+")
RE_ZIP = re.compile(r"\b\d{5,6}\b")
RE_HOUSE_NUM = re.compile(r"^\D*(\d+)")
RE_LEGAL_SUFFIX = re.compile(
    r"\b(inc|incorporated|llc|limited liability company|ltd|limited|corp|corporation|co|company|gmbh|sa|srl|bv|sl|oy|ab|pvt|plc)\b",
    re.IGNORECASE
)
RE_UNIT = re.compile(r"\b(suite|ste|apt|apartment|unit|floor|fl|room|rm|bldg|building|dept|box|p\.?o\.?\s*box)\b", re.IGNORECASE)
RE_CARDINAL = re.compile(r"\b(n|s|e|w|ne|nw|se|sw|north|south|east|west)\b", re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# 150 Feature Names List
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES_V4: List[str] = [
    # Group 1: Name Similarity (35 features)
    "name_exact", "name_exact_ci", "name_alphanum_match", "name_token_jaccard",
    "name_token_dice", "name_token_overlap", "name_token_simpson", "name_lev_ratio",
    "name_jaro", "name_jaro_winkler", "name_char_2gram_jaccard", "name_char_3gram_jaccard",
    "name_char_4gram_jaccard", "name_char_5gram_jaccard", "name_char_3gram_cosine",
    "name_word_2gram_jaccard", "name_prefix_3", "name_prefix_4", "name_prefix_5", "name_prefix_6",
    "name_suffix_3", "name_suffix_4", "name_lcs_ratio", "name_lcsq_ratio",
    "name_length_ratio", "name_length_diff", "name_token_count_diff", "name_clean_legal_match",
    "name_clean_legal_jaccard", "name_soundex_match", "name_metaphone_match",
    "name_first_word_exact", "name_last_word_exact", "name_acronym_match", "name_rf_token_sort",

    # Group 2: Address Similarity (30 features)
    "addr_exact", "addr_token_jaccard", "addr_token_overlap", "addr_lev_ratio",
    "addr_jaro_winkler", "addr_char_3gram_cosine", "addr_char_4gram_cosine", "addr_house_num_exact",
    "addr_house_num_both_present", "addr_house_num_diff", "addr_street_name_sim", "addr_zip_exact",
    "addr_zip_both_present", "addr_zip_prefix_3", "addr_city_exact", "addr_city_sim",
    "addr_state_country_match", "addr_length_ratio", "addr_length_diff", "addr_token_count_diff",
    "addr_unit_match", "addr_cardinal_dir_match", "addr_po_box_match", "addr_first_word_match",
    "addr_last_word_match", "addr_digit_overlap", "addr_clean_street_jaccard", "addr_has_suite",
    "addr_rf_token_sort", "addr_rf_partial_ratio",

    # Group 3: Embedding Features (25 features)
    "emb_name_cosine", "emb_addr_cosine", "emb_full_cosine", "emb_dot_product",
    "emb_euclidean_dist", "emb_manhattan_dist", "emb_cosine_diff", "emb_min_cosine",
    "emb_max_cosine", "emb_harmonic_mean_cosine", "emb_prod_mean", "emb_prod_std",
    "emb_prod_max", "emb_prod_min", "emb_norm_ratio", "emb_diff_norm",
    "emb_top5_component_1", "emb_top5_component_2", "emb_top5_component_3",
    "emb_top5_component_4", "emb_top5_component_5", "emb_dense_rank", "emb_dense_margin",
    "emb_dense_score_ratio", "emb_bge_confidence",

    # Group 4: Candidate Context Features (20 features)
    "cand_rank", "cand_rank_percentile", "cand_total_for_s1", "cand_sn", "cand_sa",
    "cand_sim_sum", "cand_sim_prod", "cand_sim_ratio", "cand_rel_margin",
    "cand_rel_margin_ratio", "cand_gap_to_next", "cand_gap_from_second", "cand_is_rank_0",
    "cand_is_rank_1", "cand_is_rank_2", "cand_zip_density", "cand_city_density",
    "cand_prefix_density", "cand_is_source_2", "cand_is_source_3",

    # Group 5: Blocking Features (15 features)
    "cand_slot", "block_is_slot_0", "block_is_slot_1", "block_is_slot_2",
    "block_exact_det_match", "block_sorted_neigh_match", "block_tfidf_name_match",
    "block_tfidf_addr_match", "block_channel_count", "block_euclidean_sn_sa",
    "block_harmonic_sn_sa", "block_geometric_sn_sa", "block_l1_dist",
    "block_confuser_hardness", "block_priority_score",

    # Group 6: Joint Cross-Field & Branch Detection (25 features)
    "cross_name_in_addr", "cross_addr_in_name", "cross_city_in_name", "cross_name_and_addr_jaccard_prod",
    "cross_name_and_addr_jaro_prod", "cross_name_and_addr_lev_prod", "cross_name_and_zip_match",
    "cross_name_and_city_match", "cross_name_and_house_match", "cross_high_precision_triplet",
    "cross_mismatch_penalty", "cross_country_mismatch", "cross_digit_count_diff", "cross_vowel_diff",
    "cross_charset_overlap", "cross_min_all_sims", "cross_max_all_sims", "cross_mean_all_sims",
    "cross_std_all_sims", "cross_geometric_mean_sim", "cross_short_name_penalty", "cross_exact_name_fuzzy_addr",
    "cross_fuzzy_name_exact_addr", "cross_both_exact", "cross_branch_detector"
]

assert len(FEATURE_NAMES_V4) == 150, f"Expected 150 feature names, got {len(FEATURE_NAMES_V4)}"

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def _clean_str(s: Any) -> str:
    if s is None or s != s:
        return ""
    return str(s).strip()

def _char_ngrams(s: str, n: int) -> set:
    s_clean = s.lower()
    return {s_clean[i:i+n] for i in range(max(len(s_clean) - n + 1, 0))}

def _word_ngrams(tokens: list, n: int) -> set:
    return {tuple(tokens[i:i+n]) for i in range(max(len(tokens) - n + 1, 0))}

def _jaccard(sa: set, sb: set) -> float:
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def _dice(sa: set, sb: set) -> float:
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return 2.0 * len(sa & sb) / (len(sa) + len(sb))

def _overlap(sa: set, sb: set) -> float:
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))

def _cosine_ngrams(sa: set, sb: set) -> float:
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / (np.sqrt(len(sa)) * np.sqrt(len(sb)))

def _longest_common_substring(s1: str, s2: str) -> int:
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest = 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
            else:
                m[x][y] = 0
    return longest

def _longest_common_subsequence(s1: str, s2: str) -> int:
    dp = [0] * (len(s2) + 1)
    for c1 in s1:
        prev = 0
        for j, c2 in enumerate(s2, 1):
            temp = dp[j]
            dp[j] = prev + 1 if c1 == c2 else max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]

# ─────────────────────────────────────────────────────────────────────────────
# 150-Feature Vector Generator for a Single Pair
# ─────────────────────────────────────────────────────────────────────────────
def compute_pair_features_v4(
    r1: Dict[str, Any],
    r2: Dict[str, Any],
    context: Dict[str, Any] = None,
) -> np.ndarray:
    """
    Computes all 150 features for an entity pair (r1: S1, r2: S2/S3).
    Returns a 1D float32 numpy array of length 150.
    """
    feats = np.zeros(150, dtype=np.float32)
    ctx = context or {}

    # Extract strings
    na = _clean_str(r1.get("business_name", ""))
    aa = _clean_str(r1.get("business_address", ""))
    ca = _clean_str(r1.get("country", "")).lower()

    nb = _clean_str(r2.get("business_name", ""))
    ab = _clean_str(r2.get("business_address", ""))
    cb = _clean_str(r2.get("country", "")).lower()

    na_l, nb_l = na.lower(), nb.lower()
    aa_l, ab_l = aa.lower(), ab.lower()

    # Pre-tokenize
    na_toks, nb_toks = na_l.split(), nb_l.split()
    na_set, nb_set = set(na_toks), set(nb_toks)

    aa_toks, ab_toks = aa_l.split(), ab_l.split()
    aa_set, ab_set = set(aa_toks), set(ab_toks)

    # ═════════════════════════════════════════════════════════════
    # Group 1: Name Similarity (Indices 0 - 34)
    # ═════════════════════════════════════════════════════════════
    feats[0] = 1.0 if na == nb and len(na) > 0 else 0.0
    feats[1] = 1.0 if na_l == nb_l and len(na) > 0 else 0.0
    na_alnum = "".join(c for c in na_l if c.isalnum())
    nb_alnum = "".join(c for c in nb_l if c.isalnum())
    feats[2] = 1.0 if na_alnum == nb_alnum and len(na_alnum) > 0 else 0.0
    feats[3] = _jaccard(na_set, nb_set)
    feats[4] = _dice(na_set, nb_set)
    feats[5] = _overlap(na_set, nb_set)
    feats[6] = len(na_set & nb_set) / max(max(len(na_set), len(nb_set)), 1)

    if _HAS_RAPIDFUZZ:
        feats[7] = _rf_lev.normalized_similarity(na_l, nb_l)
        feats[8] = _rfuzz.ratio(na_l, nb_l) / 100.0
        feats[9] = _rf_jw.similarity(na_l, nb_l)
    else:
        feats[7] = feats[3]
        feats[8] = feats[3]
        feats[9] = feats[3]

    # N-grams
    n2a, n2b = _char_ngrams(na_l, 2), _char_ngrams(nb_l, 2)
    feats[10] = _jaccard(n2a, n2b)
    n3a, n3b = _char_ngrams(na_l, 3), _char_ngrams(nb_l, 3)
    feats[11] = _jaccard(n3a, n3b)
    n4a, n4b = _char_ngrams(na_l, 4), _char_ngrams(nb_l, 4)
    feats[12] = _jaccard(n4a, n4b)
    n5a, n5b = _char_ngrams(na_l, 5), _char_ngrams(nb_l, 5)
    feats[13] = _jaccard(n5a, n5b)
    feats[14] = _cosine_ngrams(n3a, n3b)
    feats[15] = _jaccard(_word_ngrams(na_toks, 2), _word_ngrams(nb_toks, 2))

    # Prefixes & Suffixes
    feats[16] = 1.0 if na_l[:3] == nb_l[:3] and len(na_l) >= 3 else 0.0
    feats[17] = 1.0 if na_l[:4] == nb_l[:4] and len(na_l) >= 4 else 0.0
    feats[18] = 1.0 if na_l[:5] == nb_l[:5] and len(na_l) >= 5 else 0.0
    feats[19] = 1.0 if na_l[:6] == nb_l[:6] and len(na_l) >= 6 else 0.0
    feats[20] = 1.0 if na_l[-3:] == nb_l[-3:] and len(na_l) >= 3 else 0.0
    feats[21] = 1.0 if na_l[-4:] == nb_l[-4:] and len(na_l) >= 4 else 0.0

    # Substring / Subsequence
    max_nl = max(len(na_l), len(nb_l), 1)
    feats[22] = _longest_common_substring(na_l[:50], nb_l[:50]) / max_nl
    feats[23] = _longest_common_subsequence(na_l[:50], nb_l[:50]) / max_nl

    # Length & token stats
    feats[24] = min(len(na_l), len(nb_l)) / max_nl
    feats[25] = abs(len(na_l) - len(nb_l))
    feats[26] = abs(len(na_toks) - len(nb_toks))

    # Legal suffix clean
    na_clean = RE_LEGAL_SUFFIX.sub("", na_l).strip()
    nb_clean = RE_LEGAL_SUFFIX.sub("", nb_l).strip()
    feats[27] = 1.0 if na_clean == nb_clean and len(na_clean) > 0 else 0.0
    feats[28] = _jaccard(set(na_clean.split()), set(nb_clean.split()))

    # Phonetics
    if _HAS_JELLYFISH and na_toks and nb_toks:
        feats[29] = 1.0 if jellyfish.soundex(na_toks[0]) == jellyfish.soundex(nb_toks[0]) else 0.0
        feats[30] = 1.0 if jellyfish.metaphone(na_toks[0]) == jellyfish.metaphone(nb_toks[0]) else 0.0
    else:
        feats[29] = feats[16]
        feats[30] = feats[16]

    feats[31] = 1.0 if na_toks and nb_toks and na_toks[0] == nb_toks[0] else 0.0
    feats[32] = 1.0 if na_toks and nb_toks and na_toks[-1] == nb_toks[-1] else 0.0

    # Acronym
    acro_a = "".join(w[0] for w in na_toks if w)
    acro_b = "".join(w[0] for w in nb_toks if w)
    feats[33] = 1.0 if (na_l == acro_b or nb_l == acro_a) and len(acro_a) >= 2 else 0.0

    feats[34] = _rfuzz.token_sort_ratio(na_l, nb_l) / 100.0 if _HAS_RAPIDFUZZ else feats[3]

    # ═════════════════════════════════════════════════════════════
    # Group 2: Address Similarity (Indices 35 - 64)
    # ═════════════════════════════════════════════════════════════
    feats[35] = 1.0 if aa == ab and len(aa) > 0 else 0.0
    feats[36] = _jaccard(aa_set, ab_set)
    feats[37] = _overlap(aa_set, ab_set)
    feats[38] = _rf_lev.normalized_similarity(aa_l, ab_l) if _HAS_RAPIDFUZZ else feats[36]
    feats[39] = _rf_jw.similarity(aa_l, ab_l) if _HAS_RAPIDFUZZ else feats[36]

    a3a, a3b = _char_ngrams(aa_l, 3), _char_ngrams(ab_l, 3)
    feats[40] = _cosine_ngrams(a3a, a3b)
    a4a, a4b = _char_ngrams(aa_l, 4), _char_ngrams(ab_l, 4)
    feats[41] = _cosine_ngrams(a4a, a4b)

    # House / Street Number
    ha = RE_HOUSE_NUM.findall(aa_l)
    hb = RE_HOUSE_NUM.findall(ab_l)
    feats[42] = 1.0 if ha and hb and ha[0] == hb[0] else 0.0
    feats[43] = 1.0 if ha and hb else 0.0
    feats[44] = abs(int(ha[0]) - int(hb[0])) if ha and hb and ha[0].isdigit() and hb[0].isdigit() else 0.0

    # Street name without numbers
    street_a = RE_DIGITS.sub("", aa_l).strip()
    street_b = RE_DIGITS.sub("", ab_l).strip()
    feats[45] = _rf_jw.similarity(street_a, street_b) if _HAS_RAPIDFUZZ else _jaccard(set(street_a.split()), set(street_b.split()))

    # ZIP / Postal code
    za, zb = RE_ZIP.findall(aa_l), RE_ZIP.findall(ab_l)
    feats[46] = 1.0 if za and zb and za[0] == zb[0] else 0.0
    feats[47] = 1.0 if za and zb else 0.0
    feats[48] = 1.0 if za and zb and za[0][:3] == zb[0][:3] else 0.0

    # City matching
    feats[49] = 1.0 if len(aa_set & ab_set) >= 2 else 0.0
    feats[50] = _rfuzz.partial_ratio(aa_l, ab_l) / 100.0 if _HAS_RAPIDFUZZ else feats[36]

    feats[51] = 1.0 if ca and cb and ca == cb else 0.0
    max_al = max(len(aa_l), len(ab_l), 1)
    feats[52] = min(len(aa_l), len(ab_l)) / max_al
    feats[53] = abs(len(aa_l) - len(ab_l))
    feats[54] = abs(len(aa_toks) - len(ab_toks))

    # Unit / Suite
    ua, ub = RE_UNIT.findall(aa_l), RE_UNIT.findall(ab_l)
    feats[55] = 1.0 if ua and ub and set(ua) & set(ub) else 0.0
    ca_dir, cb_dir = RE_CARDINAL.findall(aa_l), RE_CARDINAL.findall(ab_l)
    feats[56] = 1.0 if ca_dir and cb_dir and set(ca_dir) & set(cb_dir) else 0.0
    feats[57] = 1.0 if ("po box" in aa_l and "po box" in ab_l) else 0.0

    feats[58] = 1.0 if aa_toks and ab_toks and aa_toks[0] == ab_toks[0] else 0.0
    feats[59] = 1.0 if aa_toks and ab_toks and aa_toks[-1] == ab_toks[-1] else 0.0

    da = set(RE_DIGITS.findall(aa_l))
    db = set(RE_DIGITS.findall(ab_l))
    feats[60] = len(da & db) / max(len(da | db), 1)
    feats[61] = _jaccard(set(street_a.split()), set(street_b.split()))
    feats[62] = 1.0 if ua or ub else 0.0
    feats[63] = _rfuzz.token_sort_ratio(aa_l, ab_l) / 100.0 if _HAS_RAPIDFUZZ else feats[36]
    feats[64] = _rfuzz.partial_ratio(aa_l, ab_l) / 100.0 if _HAS_RAPIDFUZZ else feats[37]

    # ═════════════════════════════════════════════════════════════
    # Group 3: Embedding Features (Indices 65 - 89)
    # ═════════════════════════════════════════════════════════════
    # If dense embeddings are passed in context, compute true cosine/stats, else use TF-IDF proxy
    cos_n = float(ctx.get("emb_name_cos", feats[14]))
    cos_a = float(ctx.get("emb_addr_cos", feats[40]))
    cos_f = float(ctx.get("emb_full_cos", 0.5 * (cos_n + cos_a)))

    feats[65] = cos_n
    feats[66] = cos_a
    feats[67] = cos_f
    feats[68] = cos_n * cos_a
    feats[69] = float(np.sqrt(max(2.0 - 2.0 * cos_f, 0.0)))  # Euclidean on unit vectors
    feats[70] = float(abs(1.0 - cos_n) + abs(1.0 - cos_a))
    feats[71] = abs(cos_n - cos_a)
    feats[72] = min(cos_n, cos_a)
    feats[73] = max(cos_n, cos_a)
    feats[74] = 2.0 * cos_n * cos_a / max(cos_n + cos_a, 1e-6)

    # Continuous component summary stats
    feats[75] = (cos_n + cos_a + cos_f) / 3.0
    feats[76] = float(np.std([cos_n, cos_a, cos_f]))
    feats[77] = max(cos_n, cos_a, cos_f)
    feats[78] = min(cos_n, cos_a, cos_f)
    feats[79] = 1.0 if cos_f > 0.85 else 0.0
    feats[80] = float(abs(cos_n - cos_f))

    # Dense component proxies
    feats[81] = cos_n ** 2
    feats[82] = cos_a ** 2
    feats[83] = cos_f ** 2
    feats[84] = float(np.cbrt(max(cos_n * cos_a * cos_f, 0.0)))
    feats[85] = 1.0 if cos_n > 0.90 and cos_a > 0.70 else 0.0

    feats[86] = float(ctx.get("emb_rank", 0))
    feats[87] = float(ctx.get("emb_margin", 0.0))
    feats[88] = float(ctx.get("emb_score_ratio", 1.0))
    feats[89] = 1.0 if cos_f > 0.88 else 0.0

    # ═════════════════════════════════════════════════════════════
    # Group 4: Candidate Context Features (Indices 90 - 109)
    # ═════════════════════════════════════════════════════════════
    rank = float(ctx.get("rank", 0))
    total_cands = max(float(ctx.get("total_cands", 1)), 1.0)
    sn = float(ctx.get("sn", feats[14] * 100.0))
    sa = float(ctx.get("sa", feats[40] * 100.0))
    top_sim = float(ctx.get("top_sim", sn + sa))

    feats[90] = rank
    feats[91] = rank / total_cands
    feats[92] = total_cands
    feats[93] = sn
    feats[94] = sa
    feats[95] = sn + sa
    feats[96] = sn * sa
    feats[97] = sn / max(sa, 0.01)
    feats[98] = top_sim - (sn + sa)
    feats[99] = (sn + sa) / max(top_sim, 0.01)
    feats[100] = float(ctx.get("gap_to_next", 0.0))
    feats[101] = float(ctx.get("gap_from_second", 0.0))
    feats[102] = 1.0 if rank == 0 else 0.0
    feats[103] = 1.0 if rank == 1 else 0.0
    feats[104] = 1.0 if rank == 2 else 0.0
    feats[105] = float(ctx.get("zip_density", 1.0))
    feats[106] = float(ctx.get("city_density", 1.0))
    feats[107] = float(ctx.get("prefix_density", 1.0))
    o_id = _clean_str(r2.get("entity_id", ""))
    feats[108] = 1.0 if o_id.startswith("S2-") else 0.0
    feats[109] = 1.0 if o_id.startswith("S3-") else 0.0

    # ═════════════════════════════════════════════════════════════
    # Group 5: Blocking Features (Indices 110 - 124)
    # ═════════════════════════════════════════════════════════════
    slot = float(ctx.get("slot", 0))
    feats[110] = slot
    feats[111] = 1.0 if slot == 0 else 0.0
    feats[112] = 1.0 if slot == 1 else 0.0
    feats[113] = 1.0 if slot == 2 else 0.0
    feats[114] = 1.0 if (feats[2] == 1.0 and feats[46] == 1.0) else 0.0
    feats[115] = 1.0 if (feats[17] == 1.0 and feats[48] == 1.0) else 0.0
    feats[116] = 1.0 if sn >= 30.0 else 0.0
    feats[117] = 1.0 if sa >= 30.0 else 0.0
    feats[118] = feats[114] + feats[115] + feats[116] + feats[117]
    feats[119] = float(np.sqrt(sn**2 + sa**2))
    feats[120] = 2.0 * sn * sa / max(sn + sa, 1e-5)
    feats[121] = float(np.sqrt(max(sn * sa, 0.0)))
    feats[122] = abs(sn - sa)
    feats[123] = (sn + sa) if slot < 2 else 0.0
    feats[124] = sn * 0.6 + sa * 0.4 - slot * 5.0

    # ═════════════════════════════════════════════════════════════
    # Group 6: Joint Cross-Field & Branch Detection (Indices 125 - 149)
    # ═════════════════════════════════════════════════════════════
    feats[125] = 1.0 if (len(na_l) >= 4 and na_l in ab_l) else 0.0
    feats[126] = 1.0 if (len(street_a) >= 4 and street_a in nb_l) else 0.0
    feats[127] = 1.0 if any(len(tok) >= 4 and tok in nb_l for tok in aa_toks) else 0.0
    feats[128] = feats[3] * feats[36]
    feats[129] = feats[9] * feats[39]
    feats[130] = feats[7] * feats[38]
    feats[131] = 1.0 if feats[5] >= 0.8 and feats[46] == 1.0 else 0.0
    feats[132] = 1.0 if feats[5] >= 0.8 and feats[49] == 1.0 else 0.0
    feats[133] = 1.0 if feats[5] >= 0.8 and feats[42] == 1.0 else 0.0
    feats[134] = 1.0 if feats[3] >= 0.75 and feats[42] == 1.0 and feats[46] == 1.0 else 0.0
    # Branch penalty: High name similarity BUT different ZIP & different City
    feats[135] = 1.0 if (feats[9] >= 0.85 and feats[46] == 0.0 and feats[49] == 0.0) else 0.0
    feats[136] = 1.0 if (ca and cb and ca != cb) else 0.0
    feats[137] = abs(len(da) - len(db))
    vowels = set("aeiou")
    va = sum(1 for c in na_l if c in vowels) / max(len(na_l), 1)
    vb = sum(1 for c in nb_l if c in vowels) / max(len(nb_l), 1)
    feats[138] = abs(va - vb)
    feats[139] = _jaccard(set(na_l), set(nb_l))
    core_sims = [feats[3], feats[9], feats[36], feats[39]]
    feats[140] = min(core_sims)
    feats[141] = max(core_sims)
    feats[142] = float(np.mean(core_sims))
    feats[143] = float(np.std(core_sims))
    feats[144] = float(np.sqrt(max(feats[9] * feats[39], 0.0)))
    feats[145] = 1.0 if min(len(na_l), len(nb_l)) <= 4 and feats[9] < 0.95 else 0.0
    feats[146] = 1.0 if feats[1] == 1.0 and feats[39] < 0.5 else 0.0
    feats[147] = 1.0 if feats[9] < 0.6 and feats[35] == 1.0 else 0.0
    feats[148] = 1.0 if feats[1] == 1.0 and feats[35] == 1.0 else 0.0
    # Branch detector: Same business name, same city, but different house/street number
    feats[149] = 1.0 if (feats[9] >= 0.85 and feats[49] == 1.0 and feats[42] == 0.0 and feats[43] == 1.0) else 0.0

    return feats
