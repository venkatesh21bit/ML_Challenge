"""
blocking_dense.py — Layer 3: Dense embedding retrieval via FAISS ANN.

Uses a sentence-transformers model to encode business name + address
into a dense vector, then performs approximate nearest-neighbor search.

Handles semantic equivalences that character-level TF-IDF misses:
  "Technologies" ≈ "Tech"
  "Mahatma Gandhi Road" ≈ "MG Road"

For 5M records at fp32 (384-dim):
  Memory: ~7.3 GB → use fp16 → ~3.6 GB
  FAISS IndexFlatIP is exact; IVF for >1M vectors.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, List, Optional
import faiss
import os

from src.data.normalize import normalize_name, normalize_address, build_combined_text


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # 384-dim, multilingual (Indian scripts + French), Apache 2.0
# Backup: "sentence-transformers/all-MiniLM-L6-v2" (English only)



def encode_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    device: str = "cpu",
    normalize: bool = True,
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Encode a list of texts into L2-normalized float32 embeddings.
    Returns shape (N, D).
    """
    from sentence_transformers import SentenceTransformer

    load_path = cache_dir if (cache_dir and os.path.exists(cache_dir)) else model_name
    print(f"  Loading embedding model: {load_path}")
    model = SentenceTransformer(load_path, device=device)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(
    embeddings: np.ndarray,
    use_ivf: bool = False,
    nlist: int = 4096,
) -> faiss.Index:
    """
    Build a FAISS index from embeddings (already L2-normalized → use Inner Product).

    For N > 500K, use IVF for faster search. Below that, exact FlatIP.
    """
    d = embeddings.shape[1]
    N = embeddings.shape[0]

    if use_ivf and N > 100_000:
        print(f"  Building IVF FAISS index (nlist={nlist}, N={N:,}, d={d})...")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(64, nlist)   # trade speed for recall
    else:
        print(f"  Building Flat FAISS index (N={N:,}, d={d})...")
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

    return index


def save_faiss_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)
    print(f"  FAISS index saved to {path}")


def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def query_faiss(
    query_embeddings: np.ndarray,
    index: faiss.Index,
    entity_ids: List[str],
    top_k: int = 30,
    batch_size: int = 1024,
) -> List[List[str]]:
    """
    Query FAISS index for top_k nearest neighbors for each query.
    Returns list of lists of entity_ids.
    """
    results = []
    N = query_embeddings.shape[0]
    k_actual = min(top_k, index.ntotal)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        q_batch = query_embeddings[start:end]
        _, indices = index.search(q_batch, k_actual)     # (B, k)
        for row in indices:
            results.append([entity_ids[i] for i in row if i >= 0])

    return results


def run_dense_blocking(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    top_k: int = 30,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    batch_size: int = 256,
    use_ivf: bool = True,
    cache_dir: Optional[str] = None,
    index_cache_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Set[str]]:
    """
    Full dense blocking pass.
    Returns: {s1_entity_id → set of candidate entity_ids}
    """
    db_eids = s_other["entity_id"].tolist()
    s1_eids = s1["entity_id"].tolist()

    # Build combined texts for encoding
    if verbose:
        print(f"  Building combined texts for {len(s_other):,} source records...")
    db_names = s_other["business_name"].fillna("").astype(str).tolist()
    db_addrs = s_other["business_address"].fillna("").astype(str).tolist()
    db_texts = [
        build_combined_text({"business_name": n, "business_address": a}, weight_name=2)
        for n, a in zip(db_names, db_addrs)
    ]
    del db_names, db_addrs

    if verbose:
        print(f"  Building combined texts for {len(s1):,} S1 records...")
    q_names = s1["business_name"].fillna("").astype(str).tolist()
    q_addrs = s1["business_address"].fillna("").astype(str).tolist()
    q_texts = [
        build_combined_text({"business_name": n, "business_address": a}, weight_name=2)
        for n, a in zip(q_names, q_addrs)
    ]
    del q_names, q_addrs

    # Encode database
    if index_cache_path and os.path.exists(index_cache_path):
        if verbose:
            print(f"  Loading cached FAISS index from {index_cache_path}")
        index = load_faiss_index(index_cache_path)
    else:
        if verbose:
            print(f"  Encoding {len(db_texts):,} database records...")
        db_embeddings = encode_texts(db_texts, model_name, batch_size, device, cache_dir=cache_dir)
        index = build_faiss_index(db_embeddings, use_ivf=use_ivf)
        if index_cache_path:
            save_faiss_index(index, index_cache_path)

    # Encode queries
    if verbose:
        print(f"  Encoding {len(q_texts):,} query records...")
    q_embeddings = encode_texts(q_texts, model_name, batch_size, device, cache_dir=cache_dir)

    # Search
    if verbose:
        print(f"  Searching FAISS (top_k={top_k})...")
    raw_cands = query_faiss(q_embeddings, index, db_eids, top_k)

    result: Dict[str, Set[str]] = {}
    for i, eid in enumerate(s1_eids):
        result[eid] = set(raw_cands[i])

    if verbose:
        sizes = [len(v) for v in result.values()]
        print(f"  Dense candidates: avg={np.mean(sizes):.1f}, "
              f"median={np.median(sizes):.0f}, max={max(sizes)}")
    return result
