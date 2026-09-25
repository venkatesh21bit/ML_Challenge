"""
cross_encoder.py — DeBERTa / ModernBERT cross-encoder for pairwise entity matching.

A cross-encoder sees BOTH records simultaneously (not as separate embeddings),
making it far more accurate than bi-encoder cosine similarity for final ranking.

Architecture:
    [CLS] {name1} {addr1} [SEP] {name2} {addr2} [SEP]
         → linear classification head
         → P(match)

Supported base models (set in CROSS_ENCODER_MODEL):
    - microsoft/deberta-v3-large     (best accuracy, ~400MB)
    - microsoft/deberta-v3-base      (lighter, ~180MB)
    - answerdotai/ModernBERT-large   (faster, experimental)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (tiny, quick baseline)

Training is done with HuggingFace Trainer for Colab compatibility.
Inference batches are small to fit in Colab T4 (16GB).
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL",
    "microsoft/deberta-v3-base"   # change to deberta-v3-large for best results
)

MAX_LENGTH = 256   # tokens; increase to 384 for deberta-v3-large if VRAM allows


# ─────────────────────────────────────────────────────────────────────────────
# Text formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_record(name: str, address: str) -> str:
    """Format a single business record as a text string."""
    name = name.strip() if name else ""
    address = address.strip() if address else ""
    if address:
        return f"{name} | {address}"
    return name


def format_pair_text(name1: str, addr1: str, name2: str, addr2: str) -> str:
    """Format a candidate pair for cross-encoder input."""
    rec1 = format_record(name1, addr1)
    rec2 = format_record(name2, addr2)
    return rec1, rec2   # tokenizer will handle [SEP] automatically


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EntityPairDataset(Dataset):
    """
    PyTorch Dataset for entity pair classification.

    Each item: (text_a, text_b, label)
    where text_a = "name1 | addr1" and text_b = "name2 | addr2"
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, str, str]],  # (name1, addr1, name2, addr2)
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = MAX_LENGTH,
    ):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name1, addr1, name2, addr2 = self.pairs[idx]
        text_a = format_record(name1, addr1)
        text_b = format_record(name2, addr2)

        encoding = self.tokenizer(
            text_a, text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cross_encoder(
    model_name: str = CROSS_ENCODER_MODEL,
    num_labels: int = 2,
    cache_dir: Optional[str] = None,
    load_from_checkpoint: Optional[str] = None,
):
    """
    Load tokenizer + model for binary sequence classification.
    Returns (tokenizer, model).
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    load_path = load_from_checkpoint or model_name
    print(f"  Loading cross-encoder: {load_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,   # always use original tokenizer
        cache_dir=cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=num_labels,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_cross_encoder(
    train_pairs: List[Tuple[str, str, str, str]],
    train_labels: List[int],
    val_pairs: Optional[List[Tuple[str, str, str, str]]] = None,
    val_labels: Optional[List[int]] = None,
    model_name: str = CROSS_ENCODER_MODEL,
    output_dir: str = "outputs/cross_encoder",
    max_length: int = MAX_LENGTH,
    num_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    cache_dir: Optional[str] = None,
    fp16: bool = True,
):
    """
    Fine-tune a cross-encoder on entity pair classification.

    Train pairs format: [(name1, addr1, name2, addr2), ...]
    Labels: [0 or 1, ...]

    Tips for Colab:
      - Use deberta-v3-base (not large) with batch_size=16
      - Enable fp16=True for T4 GPU
      - Use gradient_checkpointing for memory efficiency
    """
    from transformers import (
        TrainingArguments, Trainer,
        AutoTokenizer, AutoModelForSequenceClassification,
        EarlyStoppingCallback,
    )
    import evaluate

    os.makedirs(output_dir, exist_ok=True)

    tokenizer, model = load_cross_encoder(model_name, cache_dir=cache_dir)

    train_dataset = EntityPairDataset(train_pairs, train_labels, tokenizer, max_length)

    val_dataset = None
    if val_pairs and val_labels:
        val_dataset = EntityPairDataset(val_pairs, val_labels, tokenizer, max_length)

    # Class weights for imbalanced data
    neg_count = train_labels.count(0)
    pos_count = train_labels.count(1)
    pos_weight = neg_count / max(pos_count, 1)
    print(f"  CE Training: {len(train_pairs):,} pairs | pos={pos_count:,} | neg={neg_count:,} | weight={pos_weight:.1f}x")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        fp16=fp16,
        evaluation_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        dataloader_num_workers=2,
        report_to="none",
        gradient_checkpointing=True,  # saves memory on T4
        gradient_accumulation_steps=2,  # effective batch = batch_size * 2
    )

    # Weighted loss trainer
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels", None)
            outputs = model(**inputs)
            logits = outputs.logits
            weights = torch.tensor(
                [1.0, pos_weight], dtype=torch.float32, device=logits.device
            )
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    callbacks = []
    if val_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Cross-encoder saved to {output_dir}")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_cross_encoder(
    pairs: List[Tuple[str, str, str, str]],
    tokenizer,
    model,
    device: str = "cuda",
    batch_size: int = 64,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """
    Run inference on a list of entity pairs.

    Returns: np.ndarray of shape (N,) with P(match) probabilities.
    """
    model.eval()
    model.to(device)

    dataset = EntityPairDataset(pairs, labels=None, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs)


def run_cross_encoder_scoring(
    s1_df: pd.DataFrame,
    s_other_df: pd.DataFrame,
    candidates: Dict[str, List[str]],
    checkpoint_path: str,
    model_name: str = CROSS_ENCODER_MODEL,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
) -> Dict[Tuple[str, str], float]:
    """
    Score all candidate pairs with the cross-encoder.

    Returns: {(s1_eid, cand_eid) → P(match)}
    """
    s1_map = {r["entity_id"]: r.to_dict() for _, r in s1_df.iterrows()}
    so_map = {r["entity_id"]: r.to_dict() for _, r in s_other_df.iterrows()}

    pair_ids: List[Tuple[str, str]] = []
    pair_texts: List[Tuple[str, str, str, str]] = []

    for s1_eid, cand_list in candidates.items():
        s1_row = s1_map.get(s1_eid, {})
        for cand_eid in cand_list:
            s2_row = so_map.get(cand_eid, {})
            if not s2_row:
                continue
            pair_ids.append((s1_eid, cand_eid))
            pair_texts.append((
                str(s1_row.get("business_name", "")),
                str(s1_row.get("business_address", "")),
                str(s2_row.get("business_name", "")),
                str(s2_row.get("business_address", "")),
            ))

    print(f"  Scoring {len(pair_texts):,} pairs with cross-encoder on {device}...")
    tokenizer, model = load_cross_encoder(model_name, load_from_checkpoint=checkpoint_path)
    probs = predict_cross_encoder(pair_texts, tokenizer, model, device, batch_size)

    return {pid: float(p) for pid, p in zip(pair_ids, probs)}
