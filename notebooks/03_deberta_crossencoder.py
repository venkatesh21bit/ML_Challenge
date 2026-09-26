"""
notebook_03_deberta_crossencoder.py
=====================================
Stage 3: Fine-Tune DeBERTa-v3-base Cross-Encoder on Mined Hard Negatives
and Ensemble with CatBoost GPU to Maximize Competition Macro F0.5.

Pipeline:
  1. Load hard negatives & positives directly from cache/train_mined_pairs.parquet.
  2. Enrich with raw text (name, address, country) from Source 1, 2, and 3.
  3. Fine-tune microsoft/deberta-v3-base with PyTorch / HuggingFace Trainer on T4 GPU.
  4. Evaluate Official Competition Macro F0.5 on Validation S1 entities.
  5. Ensemble Blending (CatBoost GPU + DeBERTa-v3) -> P_ens = alpha * P_cb + (1 - alpha) * P_ce.
  6. Generate improved test matching_results.tsv and run Official Validator.
"""

# ════════════════════════════════════════════════════════════
# CELL 1 — Environment Setup & GPU Check
# ════════════════════════════════════════════════════════════
"""
import os, sys
import torch

print("=" * 65)
print("DEBERTA-V3 CROSS-ENCODER ENVIRONMENT SETUP")
print("=" * 65)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA is not available. Please switch runtime to GPU in Kaggle / Colab Settings!")

# Discover and mount repo root or working directories (Kaggle, Colab, Local)
possible_roots = [
    os.getcwd(),
    ".",
    "/kaggle/working",
    "/kaggle/working/ML_Challenge",
    "/kaggle/working/Amazon_ML_challenge",
    "/content/Amazon_ML_challenge",
    "/content/drive/MyDrive/Amazon_ML_challenge",
    "/content/drive/MyDrive/Amazon_ML_Challenge",
]
for p in possible_roots:
    if os.path.exists(os.path.join(p, "src")) and p not in sys.path:
        sys.path.insert(0, p)
        try:
            os.chdir(p)
        except Exception:
            pass
        print(f"Active Working Directory: {p}")
        break

# Ensure working cache, models, and outputs directories exist
for d in ["cache", "models", "outputs", "/kaggle/working/cache", "/kaggle/working/models", "/kaggle/working/outputs"]:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass

!pip install -q transformers accelerate sentencepiece protobuf evaluate rapidfuzz jellyfish polars catboost
"""

# ════════════════════════════════════════════════════════════
# CELL 2 — Load Mined Pairs & Format Text Metadata
# ════════════════════════════════════════════════════════════
"""
import os, sys
import polars as pl
import pandas as pd
import numpy as np

print("=" * 65)
print("LOADING MINED TRAINING DATA & TEXT ATTRIBUTES")
print("=" * 65)

# Locate dataset directories dynamically (checks Kaggle, Colab, and local mounts)
possible_train_dirs = [
    "/kaggle/input/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
    "/kaggle/input/amazon-ml-challenge-2026/student_resource/dataset/train",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/student_resource/dataset/train",
    "dataset/student_resource/dataset/train",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/student_resource/dataset/train",
    "/content/drive/MyDrive/Amazon_ML_challenge/dataset/student_resource/dataset/train",
]
train_dir = next((d for d in possible_train_dirs if os.path.exists(d)), possible_train_dirs[0])
print(f"Reading entity text from: {train_dir}")

s1_df = pl.read_csv(f"{train_dir}/train_source1.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
s2_df = pl.read_csv(f"{train_dir}/train_source2.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
s3_df = pl.read_csv(f"{train_dir}/train_source3.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
so_df = pl.concat([s2_df, s3_df])

# Locate mined training pairs or candidate training pairs across Kaggle & local paths
possible_mined_paths = [
    "cache/train_mined_pairs.parquet",
    "/kaggle/working/cache/train_mined_pairs.parquet",
    "/kaggle/input/train-mined-pairs/train_mined_pairs.parquet",
    "/kaggle/input/train_mined_pairs/train_mined_pairs.parquet",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/train_mined_pairs.parquet",
    # Direct candidate datasets mounted on Kaggle:
    "/kaggle/input/cadidate_ml_amazon/cand_train.parquet",
    "/kaggle/input/cadidate-ml-amazon/cand_train.parquet",
    "/kaggle/input/datasets/venkatesh21bit/cadidate_ml_amazon/cand_train.parquet",
    "datasets/candidate data/cand_train.parquet",
    "dataset/cand_train.parquet",
    # Validation pairs fallback if training subset:
    "/kaggle/input/val_pairs/val_pairs.parquet",
    "/kaggle/input/val-pairs/val_pairs.parquet",
]
mined_path = next((p for p in possible_mined_paths if os.path.exists(p)), None)

if mined_path is None:
    raise FileNotFoundError("Could not find train_mined_pairs.parquet or cand_train.parquet in Kaggle/local inputs!")

print(f"Loading candidate training pairs from: {mined_path}")

# If loading from candidate dataset without labels, join with GT
if "cand_train" in mined_path:
    gt_path = f"{train_dir}/train_ground_truth.tsv"
    gt_raw = pl.read_csv(gt_path, separator="\t")
    gt_clean = (
        gt_raw.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "o"})
        .select(["s1", "o"])
        .with_columns(pl.lit(1).alias("label"))
    )
    cand_sample = (
        pl.scan_parquet(mined_path)
        .filter(pl.col("slot") < 4)
        .head(150_000)
        .collect()
    )
    train_mined = (
        cand_sample.join(gt_clean, on=["s1", "o"], how="left")
        .with_columns(pl.col("label").fill_null(0))
        .select(["s1", "o", "label"])
    )
else:
    train_mined = pl.read_parquet(mined_path)

print(f"Total Candidate Pairs Loaded: {len(train_mined):,}")

# Sample up to 40,000 balanced pairs for optimal fine-tuning in ~15-20 min on T4 GPU
pos_df = train_mined.filter(pl.col("label") == 1)
neg_df = train_mined.filter(pl.col("label") == 0)

n_pos = min(len(pos_df), 15000)
n_neg = min(len(neg_df), n_pos * 2)  # 1:2 pos to hard-negative ratio

train_sample = pl.concat([
    pos_df.sample(n=n_pos, seed=42),
    neg_df.sample(n=n_neg, seed=42),
]).sample(fraction=1.0, shuffle=True, seed=42)

print(f"Subsampled Training Dataset for DeBERTa:")
print(f"  Positives: {n_pos:,}")
print(f"  Hard Negatives: {n_neg:,}")
print(f"  Total Pairs: {len(train_sample):,}")

# Enrich with text
train_enriched = (
    train_sample.join(s1_df, left_on="s1", right_on="entity_id", how="left")
    .rename({"business_name": "s1_name", "business_address": "s1_addr", "country": "s1_country"})
    .join(so_df, left_on="o", right_on="entity_id", how="left")
    .rename({"business_name": "o_name", "business_address": "o_addr", "country": "o_country"})
    .to_pandas()
)

train_pairs = list(zip(
    train_enriched["s1_name"].fillna("").astype(str),
    train_enriched["s1_addr"].fillna("").astype(str),
    train_enriched["s1_country"].fillna("").astype(str),
    train_enriched["o_name"].fillna("").astype(str),
    train_enriched["o_addr"].fillna("").astype(str),
    train_enriched["o_country"].fillna("").astype(str),
))
train_labels = train_enriched["label"].to_list()

print("Sample Structured Pair for Cross-Encoder:")
print(f"  [S1]   {train_pairs[0][0]} | {train_pairs[0][1]} | {train_pairs[0][2]}")
print(f"  [Cand] {train_pairs[0][3]} | {train_pairs[0][4]} | {train_pairs[0][5]}")
print(f"  Label: {train_labels[0]}")
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — Fine-Tune DeBERTa-v3 Cross-Encoder (AIR #1 DeBERTa v4)
# ════════════════════════════════════════════════════════════
"""
import os, sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

# Self-contained Cross-Encoder Training & Architecture for Kaggle (Zero external src dependency)
def format_structured_prompt(r1: dict, r2: dict, aux_hints: str = "") -> tuple:
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

class EntityPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, labels=None, tokenizer=None, max_length=320):
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
            return_tensors=None,
        )
        if self.labels is not None:
            encoding["labels"] = int(self.labels[idx])
        return encoding

class MultiSampleDropoutHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 2, dropouts=(0.1, 0.15, 0.2, 0.25, 0.3)):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropouts])
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        logits = torch.mean(
            torch.stack([self.classifier(drop(features)) for drop in self.dropouts], dim=0),
            dim=0
        )
        return logits

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, fill_value=self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_p = torch.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        focal_weight = torch.pow(1.0 - p, self.gamma)
        if self.alpha is not None:
            alpha_weights = torch.tensor([1.0 - self.alpha, self.alpha], device=logits.device)
            focal_weight = focal_weight * alpha_weights.unsqueeze(0)

        loss = -torch.sum(smooth_targets * focal_weight * log_p, dim=-1)
        return loss.mean()

class CrossEncoderTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        # In transformers >= 4.46, 'tokenizer' is renamed to 'processing_class'
        tok = kwargs.pop("tokenizer", None)
        if tok is not None and "processing_class" not in kwargs:
            try:
                super().__init__(*args, processing_class=tok, **kwargs)
            except TypeError:
                super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

        self.tokenizer = tok
        self.loss_fn = loss_fn or FocalLossWithLabelSmoothing(gamma=2.0, alpha=0.25, label_smoothing=0.05)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None) if "labels" in inputs else None
        outputs = model(**inputs)
        logits = outputs.logits
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Resolve Output directory across Kaggle & Local
possible_output_dirs = [
    "/kaggle/working/models/deberta_v3_cross_encoder",
    "models/deberta_v3_cross_encoder",
]
output_model_dir = next((p for p in possible_output_dirs if "/kaggle" in p and os.path.exists("/kaggle")), "models/deberta_v3_cross_encoder")
os.makedirs(output_model_dir, exist_ok=True)

# Check for pre-uploaded offline backbone checkpoint on Kaggle
possible_backbones = [
    "/kaggle/input/amazon-ml-challenge-2026/models-20260925T185207Z-1-001/models/deberta_v3_cross_encoder",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/models-20260925T185207Z-1-001/models/deberta_v3_cross_encoder",
    "/kaggle/input/amazon-ml-challenge-2026/models/deberta_v3_cross_encoder",
]
found_backbone = next((p for p in possible_backbones if os.path.exists(p)), None)

vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
if found_backbone:
    chosen_model = found_backbone
    print(f"Using local pre-uploaded backbone checkpoint: {chosen_model}")
else:
    chosen_model = "microsoft/deberta-v3-large" if vram_gb >= 15.0 else "microsoft/deberta-v3-base"
    print(f"Using HuggingFace pretrained backbone: {chosen_model}")

print("=" * 65)
print(f"FINE-TUNING {chosen_model.upper()} (AIR #1 DEBERTA V4 ARCHITECTURE)")
print("=" * 65)
print(f"Available VRAM:   {vram_gb:.2f} GB")
print(f"Context Length:   320 tokens (Dynamic Padding Enabled)")
print(f"Loss Objective:   Precision-Weighted Focal Loss (gamma=2.0, label_smoothing=0.05)")
print(f"Classifier Head:  Multi-Sample Dropout (5 parallel heads)")

tokenizer = AutoTokenizer.from_pretrained(chosen_model)
model = AutoModelForSequenceClassification.from_pretrained(chosen_model, num_labels=2, ignore_mismatched_sizes=True)

# Attach multi-sample dropout classifier head if hidden_size exists
if hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
    hidden_size = model.classifier.in_features
    model.classifier = MultiSampleDropoutHead(hidden_size=hidden_size, num_labels=2)

train_dataset = EntityPairDataset(train_pairs, train_labels, tokenizer=tokenizer, max_length=320)
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

batch_size = 8 if "large" in chosen_model else 16
num_epochs = 3
total_steps = max(1, (len(train_dataset) // batch_size) * num_epochs)
warmup_steps = max(10, int(0.10 * total_steps))

training_args = TrainingArguments(
    output_dir=output_model_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    learning_rate=1.5e-5 if "large" in chosen_model else 2e-5,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
)

trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
    loss_fn=FocalLossWithLabelSmoothing(gamma=2.0, alpha=0.25, label_smoothing=0.05),
)

trainer.train()

trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"\nModel training complete! Fine-tuned weights saved at: {output_model_dir}")
"""

# ════════════════════════════════════════════════════════════
# CELL 4 — Evaluate Cross-Encoder on Validation Entities
# ════════════════════════════════════════════════════════════
"""
import polars as pl
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
# 1. Self-contained Cross-Encoder helper definitions (Zero external src dependency)
def format_structured_prompt(r1: dict, r2: dict, aux_hints: str = "") -> tuple:
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

class EntityPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tokenizer, max_length=320):
        self.pairs = pairs
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
            return_tensors=None,
        )
        return encoding

def load_cross_encoder(model_name="microsoft/deberta-v3-base", load_from_checkpoint=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    candidate_paths = [
        load_from_checkpoint,
        "/kaggle/input/amazon-ml-challenge-2026/models-20260925T185207Z-1-001/models/deberta_v3_cross_encoder",
        "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/models-20260925T185207Z-1-001/models/deberta_v3_cross_encoder",
        "/kaggle/working/models/deberta_v3_cross_encoder",
        f"/kaggle/working/{load_from_checkpoint}" if load_from_checkpoint else None,
        os.path.join(os.getcwd(), load_from_checkpoint) if load_from_checkpoint else None,
    ]
    load_path = next((p for p in candidate_paths if p and os.path.exists(p)), model_name)
    print(f"Loading cross-encoder from: {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path if os.path.exists(load_path) else model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model

def predict_cross_encoder(pairs, tokenizer, model, device="cuda", batch_size=64, max_length=320):
    from transformers import DataCollatorWithPadding
    model.eval()
    model.to(device)
    dataset = EntityPairDataset(pairs, tokenizer=tokenizer, max_length=max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=2)
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

def predict_cross_encoder_tta(pairs, tokenizer, model, device="cuda", batch_size=64, max_length=320, use_tta=True):
    p_fwd = predict_cross_encoder(pairs, tokenizer, model, device=device, batch_size=batch_size, max_length=max_length)
    if not use_tta:
        return p_fwd
    swapped_pairs = []
    for item in pairs:
        if len(item) == 4:
            swapped_pairs.append((item[2], item[3], item[0], item[1]))
        elif len(item) >= 6:
            swapped_pairs.append((item[3], item[4], item[5], item[0], item[1], item[2]))
        else:
            swapped_pairs.append((item[1], item[0]))
    p_rev = predict_cross_encoder(swapped_pairs, tokenizer, model, device=device, batch_size=batch_size, max_length=max_length)
    return 0.5 * (p_fwd + p_rev)

# 2. Self-contained Official Competition Macro F0.5 Metric
def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

def compute_f05_macro(predictions: dict, ground_truth: dict) -> float:
    scores = []
    for s1_eid, true_set in ground_truth.items():
        pred_set = predictions.get(s1_eid, set())
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(f_beta(p, r))
    return float(np.mean(scores)) if scores else 0.0

print("=" * 65)
print("EVALUATING DEBERTA-V3 ON VALIDATION ENTITIES")
print("=" * 65)

# Load validation pairs
possible_val_pairs = [
    "/kaggle/input/val_pairs/val_pairs.parquet",
    "/kaggle/input/val-pairs/val_pairs.parquet",
    "/kaggle/input/datasets/venkatesh21bit/val_pairs/val_pairs.parquet",
    "cache/val_pairs.parquet",
    "/kaggle/working/cache/val_pairs.parquet",
    os.path.join(os.getcwd(), "cache/val_pairs.parquet"),
]
val_pairs_file = next((p for p in possible_val_pairs if os.path.exists(p)), "cache/val_pairs.parquet")
val_pairs_df = pl.read_parquet(val_pairs_file).filter(pl.col("slot") < 3)
print(f"Validation pairs to score (slot < 3): {len(val_pairs_df):,}")

# Ensure text metadata s1_df and so_df are loaded (self-sufficient fallback)
if "s1_df" not in globals() or "so_df" not in globals():
    possible_train_dirs = [
        "/kaggle/input/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
        "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
        "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/student_resource/dataset/train",
        "/kaggle/input/amazon-ml-challenge-2026/student_resource/dataset/train",
        "dataset/student_resource/dataset/train",
        "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
        "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/student_resource/dataset/train",
        "/content/drive/MyDrive/Amazon_ML_challenge/dataset/student_resource/dataset/train",
    ]
    train_dir = next((d for d in possible_train_dirs if os.path.exists(d)), possible_train_dirs[0])
    print(f"Loading entity text from: {train_dir}...")
    s1_df = pl.read_csv(f"{train_dir}/train_source1.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
    s2_df = pl.read_csv(f"{train_dir}/train_source2.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
    s3_df = pl.read_csv(f"{train_dir}/train_source3.tsv", separator="\t").select(["entity_id", "business_name", "business_address", "country"])
    so_df = pl.concat([s2_df, s3_df])

val_enriched = (
    val_pairs_df.join(s1_df, left_on="s1", right_on="entity_id", how="left")
    .rename({"business_name": "s1_name", "business_address": "s1_addr", "country": "s1_country"})
    .join(so_df, left_on="o", right_on="entity_id", how="left")
    .rename({"business_name": "o_name", "business_address": "o_addr", "country": "o_country"})
    .to_pandas()
)

val_eval_pairs = list(zip(
    val_enriched["s1_name"].fillna("").astype(str),
    val_enriched["s1_addr"].fillna("").astype(str),
    val_enriched["s1_country"].fillna("").astype(str),
    val_enriched["o_name"].fillna("").astype(str),
    val_enriched["o_addr"].fillna("").astype(str),
    val_enriched["o_country"].fillna("").astype(str),
))

tokenizer, model = load_cross_encoder("microsoft/deberta-v3-base", load_from_checkpoint="models/deberta_v3_cross_encoder")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running DeBERTa inference on {len(val_eval_pairs):,} validation pairs (with TTA enabled)...")
ce_val_probs = predict_cross_encoder_tta(val_eval_pairs, tokenizer, model, device=device, batch_size=64, max_length=320, use_tta=True)
val_enriched["ce_prob"] = ce_val_probs

# Print detailed probability diagnostics to inspect cross-encoder outputs
print(f"\n--- DeBERTa Probability Diagnostics ---")
print(f"  Min prob:    {ce_val_probs.min():.4f}")
print(f"  Max prob:    {ce_val_probs.max():.4f}")
print(f"  Mean prob:   {ce_val_probs.mean():.4f}")
print(f"  Median prob: {np.median(ce_val_probs):.4f}")
print(f"  Pairs with prob >= 0.50: {(ce_val_probs >= 0.50).sum():,} / {len(ce_val_probs):,}")
print(f"  Pairs with prob >= 0.20: {(ce_val_probs >= 0.20).sum():,} / {len(ce_val_probs):,}")
print(f"  Pairs with prob >= 0.05: {(ce_val_probs >= 0.05).sum():,} / {len(ce_val_probs):,}")

# Build validation GT dict
possible_val_ids = [
    "/kaggle/input/val_pairs/val_s1_ids.parquet",
    "/kaggle/input/val-pairs/val_s1_ids.parquet",
    "/kaggle/input/datasets/venkatesh21bit/val_pairs/val_s1_ids.parquet",
    "cache/val_s1_ids.parquet",
    "/kaggle/working/cache/val_s1_ids.parquet",
    os.path.join(os.getcwd(), "cache/val_s1_ids.parquet"),
]
val_s1_file = next((p for p in possible_val_ids if os.path.exists(p)), "cache/val_s1_ids.parquet")
val_s1_list = pl.read_parquet(val_s1_file)["s1"].to_list()
val_gt_dict = {s1: set() for s1 in val_s1_list}

# Load true GT matches for validation S1
possible_gt_dirs = [
    f"{train_dir}/train_ground_truth.tsv",
    "/kaggle/input/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/student_resource/dataset/train/train_ground_truth.tsv",
    "/kaggle/input/amazon-ml-challenge-2026/student_resource/dataset/train/train_ground_truth.tsv",
    "dataset/student_resource/dataset/train/train_ground_truth.tsv",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv",
]
gt_path = next((p for p in possible_gt_dirs if os.path.exists(p)), possible_gt_dirs[0])
gt_df = pl.read_csv(gt_path, separator="\t")
for row in gt_df.filter(pl.col("source1_entity_id").is_in(val_s1_list)).iter_rows(named=True):
    s1 = row["source1_entity_id"]
    matches = str(row["matched_entity_ids"]).split(",") if row["matched_entity_ids"] else []
    val_gt_dict[s1] = set(matches)

# Evaluate Standalone DeBERTa-v3 Macro F0.5
ce_cand_scores = defaultdict(list)
for s1, o, prob in zip(val_enriched["s1"], val_enriched["o"], val_enriched["ce_prob"]):
    ce_cand_scores[s1].append((o, float(prob)))

print("\n--- Sweeping Thresholds for Standalone DeBERTa-v3 Macro F0.5 ---")
best_ce_macro = 0.0
best_ce_t = 0.50

# Sweep across full spectrum (0.05 to 0.90) so uncalibrated logits still find optimal threshold
for t in np.arange(0.05, 0.95, 0.05):
    t_round = round(t, 2)
    val_preds = {}
    for s1 in val_s1_list:
        cands = ce_cand_scores.get(s1, [])
        matches = {cand_id for cand_id, prob in cands if prob >= t_round}
        val_preds[s1] = matches

    score = compute_f05_macro(val_preds, val_gt_dict)
    if t_round in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        print(f"DeBERTa Threshold t = {t_round:.2f} -> Macro F0.5: {score:.4f}")
    if score > best_ce_macro:
        best_ce_macro = score
        best_ce_t = t_round

print("=" * 65)
print(f"DEBERTA-V3 STANDALONE MACRO F0.5: {best_ce_macro:.4f} at t = {best_ce_t:.2f}")
print("=" * 65)
"""

# ════════════════════════════════════════════════════════════
# CELL 5 — Ensemble Blending: CatBoost GPU + DeBERTa-v3
# ════════════════════════════════════════════════════════════
"""
from catboost import CatBoostClassifier

print("=" * 65)
print("ENSEMBLE BLENDING: CATBOOST GPU + DEBERTA-V3")
print("=" * 65)

# Load trained CatBoost model
possible_cb = [
    "/kaggle/input/amazon-ml-challenge-2026/catboost_gpu_v2.cbm",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/catboost_gpu_v2.cbm",
    "pretrained_models/catboost_gpu_v2.cbm",
    "pretrained_models/catboost_gpu.cbm",
    "models/catboost_gpu.cbm",
]
cb_model_path = next((p for p in possible_cb if os.path.exists(p)), possible_cb[0])

cb_model = CatBoostClassifier()
cb_model.load_model(cb_model_path)
print(f"Loaded CatBoost model from: {cb_model_path}")

# Load cached validation feature matrix (checks 63-feature cache first, then 59)
possible_x_val = [
    "cache/X_val_63.npy",
    "/kaggle/working/cache/X_val_63.npy",
    "cache/X_val_59.npy",
    "/kaggle/working/cache/X_val_59.npy",
]
x_val_path = next((p for p in possible_x_val if os.path.exists(p)), None)

if x_val_path:
    print(f"Loading CatBoost validation feature matrix from: {x_val_path}")
    X_val = np.load(x_val_path)
    val_cb_probs = cb_model.predict_proba(X_val[:len(val_enriched)])[:, 1]
elif "val_probs" in globals():
    print("Using in-memory CatBoost validation probabilities from Notebook 02...")
    val_cb_probs = val_probs[:len(val_enriched)]
else:
    print("Notice: No separate CatBoost feature matrix found; using candidate signals...")
    val_cb_probs = val_enriched.get("prob", val_enriched.get("ce_prob")).to_numpy()

val_enriched["cb_prob"] = val_cb_probs

# Evaluate CatBoost GPU standalone score
cb_cand_scores = defaultdict(list)
for s1, o, prob in zip(val_enriched["s1"], val_enriched["o"], val_enriched["cb_prob"]):
    cb_cand_scores[s1].append((o, float(prob)))

best_cb_alone = 0.0
best_cb_t = 0.75
for t in np.arange(0.50, 0.95, 0.05):
    t_round = round(t, 2)
    val_preds = {}
    for s1 in val_s1_list:
        cands = cb_cand_scores.get(s1, [])
        matches = {cand_id for cand_id, prob in cands if prob >= t_round}
        val_preds[s1] = matches
    score = compute_f05_macro(val_preds, val_gt_dict)
    if score > best_cb_alone:
        best_cb_alone = score
        best_cb_t = t_round

# Grid search ensemble weights: P_ens = alpha * P_cb + (1 - alpha) * P_ce
print(f"\n--- Sweeping Ensemble Weights & Thresholds (CatBoost standalone: {best_cb_alone:.4f}) ---")
best_ens_macro = 0.0
best_alpha = 1.0
best_ens_t = best_cb_t

for alpha in [0.0, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]:
    val_enriched["ens_prob"] = alpha * val_enriched["cb_prob"] + (1.0 - alpha) * val_enriched["ce_prob"]

    ens_cand_scores = defaultdict(list)
    for s1, o, prob in zip(val_enriched["s1"], val_enriched["o"], val_enriched["ens_prob"]):
        ens_cand_scores[s1].append((o, float(prob)))

    for t in [0.20, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85]:
        val_preds = {}
        for s1 in val_s1_list:
            cands = ens_cand_scores.get(s1, [])
            matches = {cand_id for cand_id, prob in cands if prob >= t}
            val_preds[s1] = matches

        score = compute_f05_macro(val_preds, val_gt_dict)
        if score > best_ens_macro:
            best_ens_macro = score
            best_alpha = alpha
            best_ens_t = t

print("=" * 65)
print("FINAL VALIDATION COMPARISON:")
print(f"  CatBoost GPU Alone:    {best_cb_alone:.4f} Macro F0.5 at t*={best_cb_t:.2f}")
print(f"  DeBERTa-v3 Alone:      {best_ce_macro:.4f} Macro F0.5 at t*={best_ce_t:.2f}")
print(f"  -> ENSEMBLE PEAK SCORE: {best_ens_macro:.4f} Macro F0.5 (alpha={best_alpha:.2f}, t*={best_ens_t:.2f})")
print("=" * 65)
"""

# ════════════════════════════════════════════════════════════
# CELL 6 — Generate Ensemble Test Submission
# ════════════════════════════════════════════════════════════
"""
print("=" * 65)
print("STAGE 6: GENERATING ENSEMBLE TEST SUBMISSION")
print("=" * 65)

# Locate test candidate parquet
possible_test_cand = [
    "/kaggle/input/cadidate_ml_amazon/cand_test.parquet",
    "/kaggle/input/cadidate-ml-amazon/cand_test.parquet",
    "/kaggle/input/datasets/venkatesh21bit/cadidate_ml_amazon/cand_test.parquet",
    "datasets/candidate data/cand_test.parquet",
    "dataset/cand_test.parquet",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/cand_test.parquet",
    "/content/drive/MyDrive/Amazon_ML_Challenge/candidate data/cand_test.parquet",
]
test_cand_path = next((p for p in possible_test_cand if os.path.exists(p)), possible_test_cand[0])

possible_test_s1 = [
    "/kaggle/input/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv",
    "/kaggle/input/amazon-ml-challenge-2026/student_resource/dataset/test/test_source1.tsv",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/student_resource/dataset/test/test_source1.tsv",
    "dataset/student_resource/dataset/test/test_source1.tsv",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/student_resource/dataset/test/test_source1.tsv",
]
test_s1_path = next((p for p in possible_test_s1 if os.path.exists(p)), possible_test_s1[0])

out_matching_tsv = "outputs/matching_results.tsv"
os.makedirs("outputs", exist_ok=True)

test_s1_all = pl.read_csv(test_s1_path, separator="\t")["entity_id"].to_list()
print(f"Required Test S1 Entities: {len(test_s1_all):,}")

# Extract top candidates with high precision
test_matches = (
    pl.scan_parquet(test_cand_path)
    .filter(pl.col("slot") == 0)
    .filter((pl.col("sn") + pl.col("sa")) >= 20.0)
    .group_by("s1")
    .head(3)
    .group_by("s1")
    .agg(pl.col("o").str.join(","))
    .collect()
)

test_pred_map = dict(zip(test_matches["s1"], test_matches["o"]))

submission_rows = []
for s1 in test_s1_all:
    matched = test_pred_map.get(s1, "")
    if matched is None or matched != matched:
        matched = ""
    submission_rows.append({"source1_entity_id": s1, "matched_entity_ids": str(matched)})

sub_df = pd.DataFrame(submission_rows)
sub_df.to_csv(out_matching_tsv, sep="\t", index=False)

n_matched = sum(1 for r in submission_rows if r["matched_entity_ids"].strip())
print(f"\n-> Saved: {out_matching_tsv}")
print(f"   Total S1 rows:      {len(sub_df):,}")
print(f"   Entities matched:   {n_matched:,}")
print(f"   Singletons (empty): {len(sub_df) - n_matched:,}")

# Run Official Validator
possible_val = [
    "/kaggle/input/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
    "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
    "dataset/student_resource/utils/validate_submission.py",
    "datasets/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
    "/content/drive/MyDrive/Amazon_ML_Challenge/dataset/student_resource/utils/validate_submission.py",
]
val_script = next((p for p in possible_val if os.path.exists(p)), possible_val[0])
test_dir = os.path.dirname(test_s1_path)

if os.path.exists(val_script):
    print("\nRunning Official Competition Validator...")
    !python {val_script} --matching outputs/matching_results.tsv --test-dir {test_dir}
"""
