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

Uses memory-isolated sequential index processing, character (3, 5) n-grams,
max_df filtering, zero-copy CSC matrix transpose, and small-batch sparse top-k
retrieval (zero dense array allocation) to fit within 6-8 GB RAM on 5M records.
"""

import gc
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

from src.data.normalize import normalize_name, normalize_address, build_combined_text


def _make_analyzer():
    """Character n-gram analyzer that works on normalized text."""
    return "char_wb"   # char_wb pads word boundaries — better for names


def query_tfidf_batch(
    query_texts: List[str],
    vec: TfidfVectorizer,
    mat: sp.csr_matrix,
    entity_ids: List[str],
    top_k: int = 30,
    batch_size: int = 32,
) -> List[List[str]]:
    """
    Retrieve top-k candidates for a batch of query texts.
    Returns list of lists of entity_ids.
    Uses sparse-sparse matrix multiplication without converting to dense arrays.
    mat.T is already a CSC matrix sharing the same data buffers (0 extra memory).
    Small batch_size (32) keeps intermediate product matrix small and fast.
    """
    q_mat = vec.transform(query_texts)   # (Q, V) in CSR format
    n_queries = q_mat.shape[0]
    results = []

    # In scipy, transposing CSR yields CSC without duplicating internal arrays
    mat_T = mat.T

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_batch = q_mat[start:end]                        # (B, V) CSR
        prod = q_batch @ mat_T                            # (B, N) sparse CSR

        # Extract top-k per query directly from CSR row slices
        for row_idx in range(prod.shape[0]):
            r_start = prod.indptr[row_idx]
            r_end = prod.indptr[row_idx + 1]
            row_indices = prod.indices[r_start:r_end]
            row_data = prod.data[r_start:r_end]

            if len(row_data) == 0:
                results.append([])
            elif len(row_data) <= top_k:
                top_order = np.argsort(-row_data)
                results.append([entity_ids[row_indices[i]] for i in top_order])
            else:
                top_part = np.argpartition(row_data, -top_k)[-top_k:]
                top_sorted = top_part[np.argsort(row_data[top_part])[::-1]]
                results.append([entity_ids[row_indices[i]] for i in top_sorted])

    return results


def build_tfidf_indexes(
    df: pd.DataFrame,
    ngram_range: tuple = (3, 5),
    max_features: int = 100_000,
    max_df: float = 0.8,
) -> tuple:
    """
    Build two TF-IDF vectorizers + sparse matrices for a source DataFrame.
    Returns: (vec_name, mat_name, vec_nameaddr, mat_nameaddr, entity_ids)
    """
    entity_ids = df["entity_id"].tolist()

    corpus_name = [
        normalize_name(str(n)) for n in df["business_name"].fillna("")
    ]

    names = df["business_name"].fillna("").astype(str).tolist()
    addrs = df["business_address"].fillna("").astype(str).tolist()
    corpus_nameaddr = [
        build_combined_text({"business_name": n, "business_address": a}, weight_name=2)
        for n, a in zip(names, addrs)
    ]
    del names, addrs

    print(f"  Fitting name TF-IDF ({len(corpus_name):,} docs, max_features={max_features:,})...")
    vec_name = TfidfVectorizer(
        analyzer=_make_analyzer(),
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=max_df,
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )
    mat_name = vec_name.fit_transform(corpus_name)
    print(f"  mat_name: {mat_name.shape}, nnz={mat_name.nnz:,}, "
          f"~{mat_name.data.nbytes/1e9:.2f}GB")

    print(f"  Fitting name+addr TF-IDF ({len(corpus_nameaddr):,} docs)...")
    vec_nameaddr = TfidfVectorizer(
        analyzer=_make_analyzer(),
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=max_df,
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )
    mat_nameaddr = vec_nameaddr.fit_transform(corpus_nameaddr)
    print(f"  mat_nameaddr: {mat_nameaddr.shape}, nnz={mat_nameaddr.nnz:,}, "
          f"~{mat_nameaddr.data.nbytes/1e9:.2f}GB")

    return vec_name, mat_name, vec_nameaddr, mat_nameaddr, entity_ids


def run_tfidf_blocking(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    top_k_name: int = 30,
    top_k_nameaddr: int = 40,
    ngram_range: tuple = (3, 5),
    max_features: int = 100_000,
    max_df: float = 0.8,
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict[str, Set[str]]:
    """
    Full TF-IDF blocking pass for one (S1, S_other) source pair.
    Runs sequentially in two phases with explicit garbage collection
    to minimize peak memory usage.
    Returns: {s1_entity_id → set of candidate entity_ids}
    """
    db_eids = s_other["entity_id"].tolist()
    s1_eids = s1["entity_id"].tolist()

    # ── Phase 1: Name-only TF-IDF index & query ─────────────────────────────
    if verbose:
        print(f"  [TF-IDF Phase 1] Building name index for {len(s_other):,} docs...")
    corpus_name = [
        normalize_name(str(n)) for n in s_other["business_name"].fillna("")
    ]
    vec_name = TfidfVectorizer(
        analyzer=_make_analyzer(),
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=max_df,
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )
    mat_name = vec_name.fit_transform(corpus_name)
    del corpus_name
    gc.collect()

    if verbose:
        print(f"  mat_name: {mat_name.shape}, nnz={mat_name.nnz:,}, "
              f"~{mat_name.data.nbytes/1e9:.2f}GB")
        print(f"  Querying name index (top_k={top_k_name}) for {len(s1):,} S1 records...")

    q_name = [
        normalize_name(str(n)) for n in s1["business_name"].fillna("")
    ]
    cands_name = query_tfidf_batch(
        q_name, vec_name, mat_name, db_eids, top_k=top_k_name, batch_size=batch_size
    )

    del vec_name, mat_name, q_name
    gc.collect()

    # ── Phase 2: Name+Address TF-IDF index & query ──────────────────────────
    if verbose:
        print(f"  [TF-IDF Phase 2] Building name+addr index for {len(s_other):,} docs...")
    names = s_other["business_name"].fillna("").astype(str).tolist()
    addrs = s_other["business_address"].fillna("").astype(str).tolist()
    corpus_nameaddr = [
        build_combined_text({"business_name": n, "business_address": a}, weight_name=2)
        for n, a in zip(names, addrs)
    ]
    del names, addrs
    gc.collect()

    vec_nameaddr = TfidfVectorizer(
        analyzer=_make_analyzer(),
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=max_df,
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )
    mat_nameaddr = vec_nameaddr.fit_transform(corpus_nameaddr)
    del corpus_nameaddr
    gc.collect()

    if verbose:
        print(f"  mat_nameaddr: {mat_nameaddr.shape}, nnz={mat_nameaddr.nnz:,}, "
              f"~{mat_nameaddr.data.nbytes/1e9:.2f}GB")
        print(f"  Querying name+addr index (top_k={top_k_nameaddr}) for {len(s1):,} S1 records...")

    q_names = s1["business_name"].fillna("").astype(str).tolist()
    q_addrs = s1["business_address"].fillna("").astype(str).tolist()
    q_nameaddr = [
        build_combined_text({"business_name": n, "business_address": a}, weight_name=2)
        for n, a in zip(q_names, q_addrs)
    ]
    del q_names, q_addrs
    gc.collect()

    cands_nameaddr = query_tfidf_batch(
        q_nameaddr, vec_nameaddr, mat_nameaddr, db_eids, top_k=top_k_nameaddr, batch_size=batch_size
    )

    del vec_nameaddr, mat_nameaddr, q_nameaddr
    gc.collect()

    # ── Union results ─────────────────────────────────────────────────────────
    result: Dict[str, Set[str]] = {}
    for i, eid in enumerate(s1_eids):
        result[eid] = set(cands_name[i]) | set(cands_nameaddr[i])

    if verbose:
        sizes = [len(v) for v in result.values()]
        print(f"  TF-IDF candidates: avg={np.mean(sizes):.1f}, "
              f"median={np.median(sizes):.0f}, max={max(sizes)}")
    return result
