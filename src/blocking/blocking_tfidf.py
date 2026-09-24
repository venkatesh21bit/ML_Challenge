"""
blocking_tfidf.py — Layer 2: Character n-gram TF-IDF retrieval.

Primary fuzzy candidate generator. Handles:
  - Abbreviations (Pvt vs Private)
  - Spelling mistakes / typos
  - Transliteration variants
  - Punctuation variations
  - Reordered tokens

Two indexes are built per source:
  Index A — name only (identity)
  Index B — name + address (geographical discrimination)

With 5M records, uses sparse matrix batched retrieval to stay in memory.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

from src.data.normalize import normalize_name, normalize_address, build_combined_text


def _make_analyzer(ngram_range=(2, 4)):
    """Character n-gram analyzer that works on normalized text."""
    return "char_wb"   # char_wb pads word boundaries — better for names


def build_tfidf_indexes(
    df: pd.DataFrame,
    ngram_range: tuple = (2, 4),
    max_features: int = 300_000,
) -> tuple:
    """
    Build two TF-IDF vectorizers + sparse matrices for a source DataFrame.
    Returns: (vec_name, mat_name, vec_nameaddr, mat_nameaddr, entity_ids)
    """
    entity_ids = df["entity_id"].tolist()

    # Corpus A: name only
    corpus_name = [
        normalize_name(str(n)) for n in df["business_name"].fillna("")
    ]

    # Corpus B: name (×2) + address
    corpus_nameaddr = []
    for _, row in df.iterrows():
        corpus_nameaddr.append(build_combined_text(row, weight_name=2))

    print(f"  Fitting name TF-IDF ({len(corpus_name):,} docs)...")
    vec_name = TfidfVectorizer(
        analyzer=_make_analyzer(),
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    mat_name = vec_name.fit_transform(corpus_name)

    print(f"  Fitting name+addr TF-IDF ({len(corpus_nameaddr):,} docs)...")
    vec_nameaddr = TfidfVectorizer(
        analyzer=_make_analyzer(),
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    mat_nameaddr = vec_nameaddr.fit_transform(corpus_nameaddr)

    return vec_name, mat_name, vec_nameaddr, mat_nameaddr, entity_ids


def query_tfidf_batch(
    query_texts: List[str],
    vec: TfidfVectorizer,
    mat: sp.csr_matrix,
    entity_ids: List[str],
    top_k: int = 30,
    batch_size: int = 512,
) -> List[List[str]]:
    """
    Retrieve top-k candidates for a batch of query texts.
    Returns list of lists of entity_ids.
    Uses batched cosine similarity to avoid OOM on large matrices.
    """
    q_mat = vec.transform(query_texts)   # (Q, V)
    n_queries = q_mat.shape[0]
    results = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_batch = q_mat[start:end]                        # (B, V)
        sims = (q_batch @ mat.T).toarray()                 # (B, N)
        # Top-k indices per query
        top_indices = np.argpartition(sims, -min(top_k, sims.shape[1]), axis=1)[:, -top_k:]
        for row_idx in range(sims.shape[0]):
            idx = top_indices[row_idx]
            # Sort by score descending
            idx_sorted = idx[np.argsort(sims[row_idx, idx])[::-1]]
            results.append([entity_ids[i] for i in idx_sorted])

    return results


def run_tfidf_blocking(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    top_k_name: int = 20,
    top_k_nameaddr: int = 30,
    ngram_range: tuple = (2, 4),
    batch_size: int = 512,
    verbose: bool = True,
) -> Dict[str, Set[str]]:
    """
    Full TF-IDF blocking pass for one (S1, S_other) source pair.
    Returns: {s1_entity_id → set of candidate entity_ids}
    """
    if verbose:
        print(f"  Building TF-IDF indexes for {len(s_other):,} records...")

    (vec_name, mat_name,
     vec_nameaddr, mat_nameaddr,
     db_eids) = build_tfidf_indexes(s_other, ngram_range=ngram_range)

    # Build query texts for all S1 records
    s1_eids = s1["entity_id"].tolist()

    q_name = [
        normalize_name(str(n)) for n in s1["business_name"].fillna("")
    ]
    q_nameaddr = []
    for _, row in s1.iterrows():
        q_nameaddr.append(build_combined_text(row, weight_name=2))

    if verbose:
        print(f"  Querying name index (top_k={top_k_name}) for {len(s1):,} S1 records...")
    cands_name = query_tfidf_batch(q_name, vec_name, mat_name, db_eids, top_k_name, batch_size)

    if verbose:
        print(f"  Querying name+addr index (top_k={top_k_nameaddr})...")
    cands_nameaddr = query_tfidf_batch(q_nameaddr, vec_nameaddr, mat_nameaddr, db_eids, top_k_nameaddr, batch_size)

    result: Dict[str, Set[str]] = {}
    for i, eid in enumerate(s1_eids):
        result[eid] = set(cands_name[i]) | set(cands_nameaddr[i])

    if verbose:
        sizes = [len(v) for v in result.values()]
        import numpy as np
        print(f"  TF-IDF candidates: avg={np.mean(sizes):.1f}, "
              f"median={np.median(sizes):.0f}, max={max(sizes)}")
    return result
