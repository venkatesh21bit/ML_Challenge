"""
cross_encoder.py — AIR #1 DeBERTa v4 Cross-Encoder for Pairwise Entity Matching.

Key Architectural Upgrades:
  1. Backbone: microsoft/deberta-v3-large (or deberta-v3-base / ModernBERT-large).
  2. Structured Field Prompting: Explicit [BUSINESS_A], [ADDRESS_A], [COUNTRY_A] tokens.
  3. Dynamic Padding: DataCollatorWithPadding (max_length=320, dynamic batching).
  4. Loss Function: Precision-Weighted Focal Loss (gamma=2.0) with Label Smoothing (0.05).
  5. Multi-Sample Dropout: 5 parallel dropout heads (p=0.10..0.30) averaged for calibration.
  6. Test-Time Augmentation (TTA): Symmetric forward + swapped record order averaging.
"""

import os
import inspect
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config & Defaults
# ─────────────────────────────────────────────────────────────────────────────

CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL",
    "microsoft/deberta-v3-large"  # High capacity disentangled attention
)

MAX_LENGTH = 320  # Accommodates full premises, landmarks, and PIN codes


# ─────────────────────────────────────────────────────────────────────────────
# 1. Structured Field Prompting
# ─────────────────────────────────────────────────────────────────────────────

def format_structured_prompt(r1: dict, r2: dict, aux_hints: str = "") -> Tuple[str, str]:
    """
    Format entity pair with field-aware delimiters:
    text_a: [BUSINESS_A] name1 [ADDRESS_A] addr1 [COUNTRY_A] country1
    text_b: [BUSINESS_B] name2 [ADDRESS_B] addr2 [COUNTRY_B] country2 [HINTS] ...
    """
    name1 = str(r1.get("business_name", "")).strip() or "N/A"
    addr1 = str(r1.get("business_address", "")).strip() or "N/A"
    c1 = str(r1.get("country", "")).strip() or "N/A"

    name2 = str(r2.get("business_name", "")).strip() or "N/A"
    addr2 = str(r2.get("business_address", "")).strip() or "N/A"
    c2 = str(r2.get("country", "")).strip() or "N/A"

    text_a = f"[BUSINESS_A] {name1} [ADDRESS_A] {addr1} [COUNTRY_A] {c1}"
    text_b = f"[BUSINESS_B] {name2} [ADDRESS_B] {addr2} [COUNTRY_B] {c2}"
    if aux_hints:
        text_b += f" [HINTS] {aux_hints.strip()}"
    return text_a, text_b


def format_record(name: str, address: str, country: str = "") -> str:
    """Format single record backwards-compatibility helper."""
    parts = [f"[NAME] {name.strip() if name else 'N/A'}"]
    if address:
        parts.append(f"[ADDRESS] {address.strip()}")
    if country:
        parts.append(f"[COUNTRY] {country.strip()}")
    return " ".join(parts)


def format_pair_text(name1: str, addr1: str, name2: str, addr2: str) -> Tuple[str, str]:
    """Format pair text backwards-compatibility helper."""
    r1 = {"business_name": name1, "business_address": addr1, "country": ""}
    r2 = {"business_name": name2, "business_address": addr2, "country": ""}
    return format_structured_prompt(r1, r2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Sample Dropout & Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class MultiSampleDropoutHead(nn.Module):
    """
    Multi-Sample Dropout Head (5 parallel dropouts averaged).
    Stabilizes gradients and sharpens calibration for precision-heavy metrics.
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 2,
        drop_rates: Tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3),
    ):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in drop_rates])
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([self.classifier(drop(features)) for drop in self.dropouts], dim=0)
        return logits.mean(dim=0)


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Precision-weighted Focal Loss with Label Smoothing for F0.5 optimization.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        with torch.no_grad():
            smoothed_targets = torch.full_like(logits, self.label_smoothing / max(num_classes - 1, 1))
            smoothed_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.label_smoothing)

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        focal_weight = torch.pow(1.0 - probs, self.gamma)
        loss = -focal_weight * smoothed_targets * log_probs

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = loss * alpha

        return loss.sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dynamic Padding Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EntityPairDataset(Dataset):
    """
    Dynamic Sequence Length Dataset for Cross-Encoder.
    Tokens are returned as raw lists for DataCollatorWithPadding batching.
    """
    def __init__(
        self,
        pairs: List[Union[Tuple, dict]],
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
        item = self.pairs[idx]
        if isinstance(item, dict):
            text_a, text_b = format_structured_prompt(item["r1"], item["r2"], item.get("hints", ""))
        elif len(item) == 4:
            r1 = {"business_name": item[0], "business_address": item[1], "country": ""}
            r2 = {"business_name": item[2], "business_address": item[3], "country": ""}
            text_a, text_b = format_structured_prompt(r1, r2)
        elif len(item) >= 6:
            r1 = {"business_name": item[0], "business_address": item[1], "country": item[2]}
            r2 = {"business_name": item[3], "business_address": item[4], "country": item[5]}
            text_a, text_b = format_structured_prompt(r1, r2)
        else:
            text_a, text_b = str(item[0]), str(item[1])

        encoding = self.tokenizer(
            text_a, text_b,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,  # Lists for dynamic collator
        )
        if self.labels is not None:
            encoding["labels"] = int(self.labels[idx])
        return encoding


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cross_encoder(
    model_name: str = CROSS_ENCODER_MODEL,
    num_labels: int = 2,
    cache_dir: Optional[str] = None,
    load_from_checkpoint: Optional[str] = None,
    use_multi_sample_dropout: bool = True,
):
    """
    Load tokenizer + model for binary sequence classification.
    Optionally equips the classification head with Multi-Sample Dropout.
    """
    candidate_paths = [
        load_from_checkpoint,
        f"/kaggle/working/{load_from_checkpoint}" if load_from_checkpoint else None,
        os.path.join(os.getcwd(), load_from_checkpoint) if load_from_checkpoint else None,
    ]
    load_path = next((p for p in candidate_paths if p and os.path.exists(p)), model_name)
    print(f"  Loading cross-encoder: {load_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    config = AutoConfig.from_pretrained(
        load_path,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        config=config,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True,
    )

    if use_multi_sample_dropout and hasattr(model, "classifier"):
        hidden_size = config.hidden_size
        model.classifier = MultiSampleDropoutHead(hidden_size, num_labels=num_labels)

    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training with Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

def train_cross_encoder(
    train_pairs: List[Union[Tuple, dict]],
    train_labels: List[int],
    val_pairs: Optional[List[Union[Tuple, dict]]] = None,
    val_labels: Optional[List[int]] = None,
    model_name: str = CROSS_ENCODER_MODEL,
    output_dir: str = "models/deberta_v3_cross_encoder",
    max_length: int = MAX_LENGTH,
    num_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1.5e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
    cache_dir: Optional[str] = None,
    fp16: bool = False,
    bf16: bool = False,
):
    """
    Fine-tune cross-encoder with Focal Loss, Multi-Sample Dropout, and Dynamic Padding.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Detect hardware capabilities
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not fp16:
        bf16 = True

    tokenizer, model = load_cross_encoder(model_name, cache_dir=cache_dir)

    train_dataset = EntityPairDataset(train_pairs, train_labels, tokenizer, max_length)
    val_dataset = None
    if val_pairs and val_labels:
        val_dataset = EntityPairDataset(val_pairs, val_labels, tokenizer, max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # Precision-oriented class weights
    neg_count = train_labels.count(0)
    pos_count = train_labels.count(1)
    pos_weight = min(neg_count / max(pos_count, 1), 3.0)
    print(f"  Training: {len(train_pairs):,} pairs | Pos={pos_count:,} | Neg={neg_count:,} (weight: {pos_weight:.2f}x)")

    args_dict = dict(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        save_strategy="epoch",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        dataloader_num_workers=2,
        report_to="none",
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
    )

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        args_dict["eval_strategy"] = "epoch" if val_dataset else "no"
    elif "evaluation_strategy" in sig:
        args_dict["evaluation_strategy"] = "epoch" if val_dataset else "no"

    valid_args = {k: v for k, v in args_dict.items() if k in sig}
    training_args = TrainingArguments(**valid_args)

    class FocalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels", None)
            outputs = model(**inputs)
            logits = outputs.logits

            if use_focal_loss:
                alpha = torch.tensor([1.0, pos_weight], dtype=logits.dtype, device=logits.device)
                loss_fn = FocalLossWithLabelSmoothing(gamma=focal_gamma, alpha=alpha, label_smoothing=label_smoothing)
            else:
                weights = torch.tensor([1.0, pos_weight], dtype=logits.dtype, device=logits.device)
                loss_fn = nn.CrossEntropyLoss(weight=weights)

            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    callbacks = []
    if val_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = FocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Model successfully saved to: {output_dir}")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# 6. Inference with Test-Time Augmentation (TTA)
# ─────────────────────────────────────────────────────────────────────────────

def predict_cross_encoder(
    pairs: List[Union[Tuple, dict]],
    tokenizer,
    model,
    device: str = "cuda",
    batch_size: int = 64,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Run forward inference on a list of entity pairs with dynamic batch padding."""
    model.eval()
    model.to(device)

    dataset = EntityPairDataset(pairs, labels=None, tokenizer=tokenizer, max_length=max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=2)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            if outputs.logits.shape[-1] == 2:
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            else:
                probs = torch.sigmoid(outputs.logits.squeeze(-1)).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs)


def predict_cross_encoder_tta(
    pairs: List[Union[Tuple, dict]],
    tokenizer,
    model,
    device: str = "cuda",
    batch_size: int = 64,
    max_length: int = MAX_LENGTH,
    use_tta: bool = True,
) -> np.ndarray:
    """
    Test-Time Augmentation (TTA):
    Computes forward pass P(A, B) and reverse pass P(B, A) and returns the average.
    Eliminates directional record bias.
    """
    p_fwd = predict_cross_encoder(pairs, tokenizer, model, device=device, batch_size=batch_size, max_length=max_length)
    if not use_tta:
        return p_fwd

    # Invert record pairs
    swapped_pairs = []
    for item in pairs:
        if isinstance(item, dict):
            swapped_pairs.append({"r1": item["r2"], "r2": item["r1"], "hints": item.get("hints", "")})
        elif len(item) == 4:
            swapped_pairs.append((item[2], item[3], item[0], item[1]))
        elif len(item) >= 6:
            swapped_pairs.append((item[3], item[4], item[5], item[0], item[1], item[2]))
        else:
            swapped_pairs.append((item[1], item[0]))

    p_rev = predict_cross_encoder(swapped_pairs, tokenizer, model, device=device, batch_size=batch_size, max_length=max_length)
    return 0.5 * (p_fwd + p_rev)

