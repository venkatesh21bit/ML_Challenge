"""
bge_reranker.py — BGE-M3 bi-encoder embeddings + BGE Reranker cross-encoder.

Two separate capabilities in one module:

1. BGE-M3 Bi-encoder (for blocking + feature embeddings)
   - Encodes name, address, combined text into dense vectors
   - Used in FAISS blocking (Layer 3) and as embedding features in CatBoost

2. BGE Reranker (for final re-ranking)
   - Cross-encoder that scores (query, candidate) pairs directly
   - More accurate than cosine similarity alone
   - Used in the ensemble as Model 3

Models:
   Bi-encoder:  BAAI/bge-m3          (~2.2GB, 1024-dim, multilingual)
   Reranker:    BAAI/bge-reranker-v2-m3   (~1.1GB)
   Lighter:     BAAI/bge-reranker-base    (~500MB)
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import torch


# ─────────────────────────────────────────────────────────────────────────────
# BGE-M3 Bi-encoder
# ─────────────────────────────────────────────────────────────────────────────

BGE_M3_MODEL = os.environ.get("BGE_M3_MODEL", "BAAI/bge-m3")
BGE_RERANKER_MODEL = os.environ.get("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def encode_with_bge_m3(
    texts: List[str],
    model_name: str = BGE_M3_MODEL,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cache_dir: Optional[str] = None,
    normalize: bool = True,
    max_length: int = 256,
) -> np.ndarray:
    """
    Encode texts with BGE-M3 (dense embeddings only, not colbert/sparse).

    Returns: float32 array of shape (N, 1024)
    """
    from FlagEmbedding import BGEM3FlagModel

    print(f"  Loading BGE-M3 from {cache_dir or model_name} on {device}...")
    model = BGEM3FlagModel(
        model_name,
        use_fp16=(device == "cuda"),
        cache_dir=cache_dir,
    )

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BGE-M3 encode"):
        batch = texts[i:i + batch_size]
        output = model.encode(
            batch,
            batch_size=len(batch),
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        emb = output["dense_vecs"]
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / (norms + 1e-8)
        all_embeddings.append(emb.astype(np.float32))

    return np.vstack(all_embeddings)


def build_entity_embeddings(
    df: pd.DataFrame,
    model_name: str = BGE_M3_MODEL,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cache_dir: Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build three embedding types for each entity:
      - 'name':     normalized business name
      - 'addr':     normalized address
      - 'combined': name + name + address (name weighted 2x)

    Returns: {entity_id → {'name': ndarray, 'addr': ndarray, 'combined': ndarray}}
    """
    from src.data.normalize import normalize_name, normalize_address, build_combined_text

    eids = df["entity_id"].tolist()
    names = [normalize_name(str(r.get("business_name", ""))) for _, r in df.iterrows()]
    addrs = [normalize_address(str(r.get("business_address", ""))) for _, r in df.iterrows()]
    combined = [build_combined_text(r.to_dict(), weight_name=2) for _, r in df.iterrows()]

    print("  Encoding names...")
    emb_names = encode_with_bge_m3(names, model_name, batch_size, device, cache_dir)
    print("  Encoding addresses...")
    emb_addrs = encode_with_bge_m3(addrs, model_name, batch_size, device, cache_dir)
    print("  Encoding combined texts...")
    emb_combined = encode_with_bge_m3(combined, model_name, batch_size, device, cache_dir)

    result = {}
    for i, eid in enumerate(eids):
        result[eid] = {
            "name":     emb_names[i],
            "addr":     emb_addrs[i],
            "combined": emb_combined[i],
        }
    return result


def save_embeddings(embeddings: Dict[str, Dict[str, np.ndarray]], path: str):
    """Save embeddings dict to .npz file for fast reload."""
    import pickle
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        import pickle
        pickle.dump(embeddings, f)
    print(f"  Embeddings saved to {path}")


def load_embeddings(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# BGE Reranker
# ─────────────────────────────────────────────────────────────────────────────

def score_with_bge_reranker(
    pairs: List[Tuple[str, str]],
    model_name: str = BGE_RERANKER_MODEL,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Score a list of (text_a, text_b) pairs with BGE reranker.

    Returns: float32 array of shape (N,) with relevance scores (logits, not probabilities).
    Use sigmoid to convert to probabilities: sigmoid(score) = P(match).

    pairs: [(query_text, candidate_text), ...]
    """
    from FlagEmbedding import FlagReranker

    print(f"  Loading BGE reranker from {cache_dir or model_name} on {device}...")
    reranker = FlagReranker(
        model_name,
        use_fp16=(device == "cuda"),
        cache_dir=cache_dir,
    )

    all_scores = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="BGE rerank"):
        batch = pairs[i:i + batch_size]
        scores = reranker.compute_score(batch, normalize=False)
        if isinstance(scores, float):
            scores = [scores]
        all_scores.extend(scores)

    return np.array(all_scores, dtype=np.float32)


def run_bge_reranker_scoring(
    s1_df: pd.DataFrame,
    s_other_df: pd.DataFrame,
    candidates: Dict[str, List[str]],
    model_name: str = BGE_RERANKER_MODEL,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
) -> Dict[Tuple[str, str], float]:
    """
    Score all candidate pairs with the BGE reranker.

    Returns: {(s1_eid, cand_eid) → reranker_score}
    Higher score → more likely to be a match.
    """
    from src.data.normalize import normalize_name, normalize_address

    s1_map = {r["entity_id"]: r.to_dict() for _, r in s1_df.iterrows()}
    so_map = {r["entity_id"]: r.to_dict() for _, r in s_other_df.iterrows()}

    pair_ids: List[Tuple[str, str]] = []
    pair_texts: List[Tuple[str, str]] = []

    for s1_eid, cand_list in candidates.items():
        s1_row = s1_map.get(s1_eid, {})
        s1_name = str(s1_row.get("business_name", ""))
        s1_addr = str(s1_row.get("business_address", ""))
        text_a = f"{s1_name} {s1_addr}".strip()

        for cand_eid in cand_list:
            s2_row = so_map.get(cand_eid, {})
            if not s2_row:
                continue
            s2_name = str(s2_row.get("business_name", ""))
            s2_addr = str(s2_row.get("business_address", ""))
            text_b = f"{s2_name} {s2_addr}".strip()

            pair_ids.append((s1_eid, cand_eid))
            pair_texts.append((text_a, text_b))

    print(f"  Scoring {len(pair_texts):,} pairs with BGE reranker...")
    raw_scores = score_with_bge_reranker(pair_texts, model_name, batch_size, device, cache_dir)

    # Convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-raw_scores))

    return {pid: float(p) for pid, p in zip(pair_ids, probs)}
