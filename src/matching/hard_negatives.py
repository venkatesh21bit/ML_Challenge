"""
hard_negatives.py — Hard negative mining for training set construction.

Instead of random negatives, we mine negatives that are DIFFICULT:
  - High name similarity but different businesses
  - Same city but different entities
  - Very close TF-IDF / embedding scores but wrong match

This dramatically improves model precision, which is critical for F0.5.

Usage:
    python -m src.matching.hard_negatives --help
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple, Optional
import random


def mine_hard_negatives(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    s_other_df: pd.DataFrame,
    ratio: int = 3,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    For each S1 entity, select hard negatives from its candidate set.

    Hard negatives = candidates that are NOT true matches.
    These are naturally hard because the blocking system thought they were similar.

    Args:
        candidates:    {s1_eid → [cand_eid, ...]}   (blocking output)
        ground_truth:  {s1_eid → {true_match_eid, ...}}
        s_other_df:    Source2/3 dataframe
        ratio:         How many negatives per positive (default 3)

    Returns:
        {s1_eid → [selected_negative_eids]}
    """
    rng = random.Random(seed)
    result: Dict[str, List[str]] = {}

    for s1_eid, cand_list in candidates.items():
        true_matches = ground_truth.get(s1_eid, set())
        negatives = [c for c in cand_list if c not in true_matches]

        n_pos = len(true_matches)
        n_neg_target = max(n_pos * ratio, ratio)  # at least `ratio` negatives

        if len(negatives) > n_neg_target:
            selected = rng.sample(negatives, n_neg_target)
        else:
            selected = negatives

        result[s1_eid] = selected

    total_neg = sum(len(v) for v in result.values())
    print(f"  Hard negatives: {total_neg:,} selected from blocking candidates")
    return result


def build_balanced_training_pairs(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    hard_neg_ratio: int = 3,
    include_all_positives: bool = True,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Build (s1_eid, cand_eid) pairs and labels for training.

    Positives:  all true matches found in the candidate set (blocking recall matters)
    Negatives:  hard negatives from blocking output

    Returns:
        pairs:  [(s1_eid, cand_eid), ...]
        labels: [0 or 1, ...]
    """
    pairs: List[Tuple[str, str]] = []
    labels: List[int] = []
    rng = random.Random(seed)

    for s1_eid, cand_list in candidates.items():
        true_matches = ground_truth.get(s1_eid, set())
        cand_set = set(cand_list)

        # Positives (in candidate set)
        pos_in_cands = [c for c in true_matches if c in cand_set]
        neg_candidates = [c for c in cand_list if c not in true_matches]

        for p in pos_in_cands:
            pairs.append((s1_eid, p))
            labels.append(1)

        # Hard negatives
        n_neg = max(len(pos_in_cands) * hard_neg_ratio, hard_neg_ratio)
        if len(neg_candidates) > n_neg:
            selected_neg = rng.sample(neg_candidates, n_neg)
        else:
            selected_neg = neg_candidates

        for n in selected_neg:
            pairs.append((s1_eid, n))
            labels.append(0)

    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"  Training pairs: {len(pairs):,} | pos={pos_count:,} | neg={neg_count:,} | ratio={neg_count/(pos_count+1e-6):.1f}x")
    return pairs, labels


def get_super_hard_negatives(
    s1_df: pd.DataFrame,
    s_other_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    name_sim_threshold: float = 0.6,
    n_per_entity: int = 5,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Generate SUPER-hard negatives: entities with high name similarity
    in the same country but are NOT matches.

    These are the most challenging cases:
      "Sri Lakshmi Electronics" vs "Sri Lakshmi Electricals"
      "ABC Tech Pvt Ltd" vs "ABC Technologies Pvt Ltd"

    Requires importing from features_v2 for similarity scoring.
    This is a slower pass — run once and cache results.
    """
    from src.matching.features_v2 import jaro_winkler
    from src.data.normalize import normalize_name, normalize_country

    rng = random.Random(seed)
    result: Dict[str, List[str]] = {}

    # Build a name → entity_id lookup per country
    country_name_idx: Dict[str, List[Tuple[str, str]]] = {}
    for _, row in s_other_df.iterrows():
        eid = row["entity_id"]
        country = normalize_country(str(row.get("country", "")))
        name = normalize_name(str(row.get("business_name", "")))
        country_name_idx.setdefault(country, []).append((name, eid))

    total = 0
    for _, s1_row in s1_df.iterrows():
        s1_eid = s1_row["entity_id"]
        s1_country = normalize_country(str(s1_row.get("country", "")))
        s1_name = normalize_name(str(s1_row.get("business_name", "")))
        true_matches = ground_truth.get(s1_eid, set())

        candidates_in_country = country_name_idx.get(s1_country, [])
        scored = []
        for cname, ceid in candidates_in_country:
            if ceid in true_matches:
                continue
            sim = jaro_winkler(s1_name, cname)
            if sim >= name_sim_threshold:
                scored.append((sim, ceid))

        scored.sort(reverse=True)
        top_hard = [eid for _, eid in scored[:n_per_entity * 3]]
        selected = rng.sample(top_hard, min(n_per_entity, len(top_hard)))
        if selected:
            result[s1_eid] = selected
            total += len(selected)

    print(f"  Super-hard negatives: {total:,} pairs across {len(result):,} S1 entities")
    return result
