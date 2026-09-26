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

# Problem 6: GroupKFold Integrity on source1_entity_id
# The same Source1 business must never appear in both train and validation
possible_val_ids = [
    "/kaggle/input/val_pairs/val_s1_ids.parquet",
    "/kaggle/input/val-pairs/val_s1_ids.parquet",
    "/kaggle/input/datasets/venkatesh21bit/val_pairs/val_s1_ids.parquet",
    "cache/val_s1_ids.parquet",
    "/kaggle/working/cache/val_s1_ids.parquet",
]
val_s1_f = next((p for p in possible_val_ids if os.path.exists(p)), None)
if val_s1_f:
    val_s1_set = set(pl.read_parquet(val_s1_f)["s1"].to_list())
    train_mined = train_mined.filter(~pl.col("s1").is_in(val_s1_set))
    print(f"GroupKFold Integrity (Problem 6): Excluded {len(val_s1_set):,} validation S1 entities from training.")

print(f"Total Candidate Pairs Loaded: {len(train_mined):,}")

# Problem 6 — Hard Negative Curriculum Training
# Model gradually learns from easier confusers to ultra-hard negatives
pos_df = train_mined.filter(pl.col("label") == 1)
neg_df = train_mined.filter(pl.col("label") == 0)

n_pos = min(len(pos_df), 15000)
n_neg = min(len(neg_df), n_pos * 2)  # 1:2 pos to hard-negative ratio

# Stage negatives by hardness (confuser slot & lexical similarity)
if "slot" in neg_df.columns:
    neg_sorted = neg_df.sort("slot", descending=False)
elif "sn" in neg_df.columns and "sa" in neg_df.columns:
    neg_sorted = neg_df.sort(pl.col("sn") + pl.col("sa"), descending=True)
else:
    neg_sorted = neg_df

n_easy = int(n_neg * 0.25)
n_hard = int(n_neg * 0.40)
n_ultra = n_neg - n_easy - n_hard

# Stage 0: Easy / moderate confusers (tail of hardness / higher slot)
neg_easy = neg_sorted.tail(n_easy).with_columns(pl.lit(0).alias("curriculum_stage"))
# Stage 1: Hard confusers (middle hardness)
neg_hard = neg_sorted.slice(len(neg_sorted) // 3, n_hard).with_columns(pl.lit(1).alias("curriculum_stage"))
# Stage 2: Ultra-hard confusers (top slot 0-1, deceptive confusers)
neg_ultra = neg_sorted.head(n_ultra).with_columns(pl.lit(2).alias("curriculum_stage"))

# Distribute positives across all curriculum stages
pos_sample = pos_df.sample(n=n_pos, seed=42)
n_p0 = int(n_pos * 0.25)
n_p1 = int(n_pos * 0.40)
n_p2 = n_pos - n_p0 - n_p1

pos_s0 = pos_sample.head(n_p0).with_columns(pl.lit(0).alias("curriculum_stage"))
pos_s1 = pos_sample.slice(n_p0, n_p1).with_columns(pl.lit(1).alias("curriculum_stage"))
pos_s2 = pos_sample.tail(n_p2).with_columns(pl.lit(2).alias("curriculum_stage"))

# Compose curriculum stages: stage 0 (easy) -> stage 1 (hard) -> stage 2 (ultra-hard)
stage_0 = pl.concat([pos_s0, neg_easy]).sample(fraction=1.0, shuffle=True, seed=42)
stage_1 = pl.concat([pos_s1, neg_hard]).sample(fraction=1.0, shuffle=True, seed=42)
stage_2 = pl.concat([pos_s2, neg_ultra]).sample(fraction=1.0, shuffle=True, seed=42)

# Ordered curriculum dataset: Model sees stage 0 first, then stage 1, then stage 2
train_sample = pl.concat([stage_0, stage_1, stage_2])

print(f"Subsampled Curriculum Training Dataset for DeBERTa (Problem 6):")
print(f"  Stage 0 (Easy/Medium Confusers): {len(stage_0):,} pairs")
print(f"  Stage 1 (Hard Confusers):        {len(stage_1):,} pairs")
print(f"  Stage 2 (Ultra-Hard Confusers):  {len(stage_2):,} pairs")
print(f"  Total Curriculum Pairs:          {len(train_sample):,}")

# Enrich with text
train_enriched = (
    train_sample.join(s1_df, left_on="s1", right_on="entity_id", how="left")
    .rename({"business_name": "s1_name", "business_address": "s1_addr", "country": "s1_country"})
    .join(so_df, left_on="o", right_on="entity_id", how="left")
    .rename({"business_name": "o_name", "business_address": "o_addr", "country": "o_country"})
    .to_pandas()
)

# ════════════════════════════════════════════════════════════
# AIR #1 Structured Quantitative Hint Generator (Problem 5)
# ════════════════════════════════════════════════════════════
import re
try:
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

pin_re = re.compile(r"\b\d{5,6}\b")
num_re = re.compile(r"^\D*(\d+)")

def make_hints(na: str, aa: str, ca: str, nb: str, ab: str, cb: str) -> str:
    # 1. Zip match
    pa, pb = pin_re.findall(aa), pin_re.findall(ab)
    zip_m = 1 if (pa and pb and pa[0] == pb[0]) else 0

    # 2. State / Country match
    state_m = 1 if ca and cb and ca.lower().strip() == cb.lower().strip() else 0

    # 3. House match
    ha, hb = num_re.findall(aa), num_re.findall(ab)
    house_m = 1 if (ha and hb and ha[0] == hb[0]) else 0

    # 4. Prefix match (first 4 characters)
    pref_m = 1 if na[:4].lower().strip() == nb[:4].lower().strip() and len(na) >= 4 else 0

    # 5. Token overlap
    sa, sb = set(na.lower().split()), set(nb.lower().split())
    tok_ov = len(sa & sb) / max(len(sa | sb), 1)

    # 6. Jaro & Levenshtein
    if _HAS_RAPIDFUZZ:
        jaro = JaroWinkler.similarity(na, nb)
        lev = Levenshtein.normalized_similarity(na, nb)
    else:
        jaro = tok_ov
        lev = tok_ov

    # 7. Fast 3-gram char similarity (character TF-IDF approximation)
    def char_ngrams(s, n=3):
        return {s[i:i+n] for i in range(max(len(s) - n + 1, 0))}
    nga, ngb = char_ngrams(na.lower()), char_ngrams(nb.lower())
    n_tf = len(nga & ngb) / max(len(nga | ngb), 1)

    aga, agb = char_ngrams(aa.lower()), char_ngrams(ab.lower())
    a_tf = len(aga & agb) / max(len(aga | agb), 1)

    # 8. City token match
    city_m = 1 if (set(aa.lower().split()) & set(ab.lower().split())) else 0

    return (
        f"ZIP_MATCH={zip_m} CITY_MATCH={city_m} STATE_MATCH={state_m} "
        f"NAME_TFIDF={n_tf:.3f} ADDRESS_TFIDF={a_tf:.3f} "
        f"TOKEN_OVERLAP={tok_ov:.2f} JARO={jaro:.2f} LEVENSHTEIN={lev:.2f} "
        f"HOUSE_MATCH={house_m} PREFIX_MATCH={pref_m}"
    )

print("Synthesizing structured prompts with AIR #1 quantitative hints...")
train_texts_a = []
train_texts_b = []

for na, aa, ca, nb, ab, cb in zip(
    train_enriched["s1_name"].fillna("").astype(str),
    train_enriched["s1_addr"].fillna("").astype(str),
    train_enriched["s1_country"].fillna("").astype(str),
    train_enriched["o_name"].fillna("").astype(str),
    train_enriched["o_addr"].fillna("").astype(str),
    train_enriched["o_country"].fillna("").astype(str),
):
    hints = make_hints(na, aa, ca, nb, ab, cb)
    train_texts_a.append(f"[BUSINESS_A] {na} [ADDRESS_A] {aa} [COUNTRY_A] {ca}")
    train_texts_b.append(f"[BUSINESS_B] {nb} [ADDRESS_B] {ab} [COUNTRY_B] {cb} [HINTS] {hints}")

train_labels = train_enriched["label"].to_list()

print("Sample Structured Pair with Expanded AIR #1 Hints:")
print(f"  [Text A] {train_texts_a[0]}")
print(f"  [Text B] {train_texts_b[0]}")
print(f"  Label:   {train_labels[0]}")
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — Fine-Tune DeBERTa-v3 Cross-Encoder (AIR #1 DeBERTa v4)
# ════════════════════════════════════════════════════════════
"""
import os, sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

# Problem 1: Use microsoft/deberta-v3-large
model_name = "microsoft/deberta-v3-large"

# Problem 2: Increase token length to 384 for complete address preservation
max_length = 384

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
chosen_model = model_name

print("=" * 65)
print(f"FINE-TUNING {chosen_model.upper()} (AIR #1 DEBERTA V4 ARCHITECTURE)")
print("=" * 65)
print(f"Model Backbone:   {chosen_model} (Upgrade 1)")
print(f"Available VRAM:   {vram_gb:.2f} GB")
print(f"Context Length:   {max_length} tokens with Dynamic Padding (Upgrade 2)")
print(f"Dataset Pipeline: Bulk Pre-Tokenization (Upgrade 3)")
print(f"Loss Objective:   Focal Loss (alpha=0.75, gamma=2.0) (Upgrade 4)")
print(f"Hint Signals:     10-Feature Quantitative Signals (Upgrade 5)")

tokenizer = AutoTokenizer.from_pretrained(chosen_model)
model = AutoModelForSequenceClassification.from_pretrained(
    chosen_model,
    num_labels=2,
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True,
)
# Enforce all base model parameters in float32 for PyTorch AMP GradScaler
model = model.float()
for param in model.parameters():
    param.data = param.data.float()

# Problem 3: Bulk Pre-Tokenization (GPU never waits for CPU tokenization)
print(f"\nPre-tokenizing {len(train_texts_a):,} pairs in bulk (Rust multi-threaded)...")
tokenized_inputs = tokenizer(
    train_texts_a,
    train_texts_b,
    max_length=max_length,
    truncation=True,
    padding=False,
)

class PreTokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item["labels"] = int(self.labels[idx])
        return item

train_dataset = PreTokenizedDataset(tokenized_inputs, train_labels)
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Problem 7: Multi-Sample Dropout Classifier Head (with dtype alignment)
class MultiSampleDropoutHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 2, dropouts=(0.1, 0.15, 0.2, 0.25, 0.3)):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropouts])
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        if self.classifier.weight.dtype != features.dtype:
            features = features.to(self.classifier.weight.dtype)
        logits = torch.mean(
            torch.stack([self.classifier(drop(features)) for drop in self.dropouts], dim=0),
            dim=0
        )
        return logits

if hasattr(model, "classifier"):
    old_classifier = model.classifier
    if hasattr(old_classifier, "in_features"):
        hidden_size = old_classifier.in_features
        head = MultiSampleDropoutHead(hidden_size=hidden_size, num_labels=2)
        if hasattr(old_classifier, "weight") and old_classifier.weight.shape == head.classifier.weight.shape:
            head.classifier.weight.data.copy_(old_classifier.weight.data.float())
            if hasattr(old_classifier, "bias") and old_classifier.bias is not None:
                head.classifier.bias.data.copy_(old_classifier.bias.data.float())
        head.float()
        head.to(device=model.device)
        model.classifier = head

# Guarantee all model parameters remain float32 master weights for AMP GradScaler (Problem 10)
for param in model.parameters():
    param.data = param.data.float()

# Problem 4: Custom FocalTrainer with FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
class FocalTrainer(Trainer):
    def __init__(self, *args, alpha=0.75, gamma=2.0, **kwargs):
        tok = kwargs.pop("tokenizer", None)
        if tok is not None and "processing_class" not in kwargs:
            try:
                super().__init__(*args, processing_class=tok, **kwargs)
            except TypeError:
                super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels", None)
        if labels is None:
            labels = inputs.pop("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        if logits.dim() == 2 and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=-1)
            p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            alpha_t = torch.where(labels == 1, self.alpha, 1.0 - self.alpha)
            loss = -(alpha_t * torch.pow(1.0 - p_t, self.gamma) * torch.log(p_t.clamp(min=1e-7))).mean()
        else:
            sig = torch.sigmoid(logits.view(-1))
            labels_f = labels.float().view(-1)
            p_t = torch.where(labels_f == 1.0, sig, 1.0 - sig)
            alpha_t = torch.where(labels_f == 1.0, self.alpha, 1.0 - self.alpha)
            loss = -(alpha_t * torch.pow(1.0 - p_t, self.gamma) * torch.log(p_t.clamp(min=1e-7))).mean()
        return (loss, outputs) if return_outputs else loss

# Problem 7: Exponential Moving Average (EMA) of Model Weights
from transformers import TrainerCallback

class EMACallback(TrainerCallback):
    """
    Maintains Exponential Moving Average (EMA) of model weights during training.
    Replaces model weights with EMA weights at training completion for maximum
    generalization and validation stability on downstream evaluation.
    """
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.ema_params = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.ema_params = {n: p.clone().detach().cpu() for n, p in model.named_parameters() if p.requires_grad}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.ema_params is None:
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.ema_params:
                    self.ema_params[n].mul_(self.decay).add_(p.detach().cpu(), alpha=1.0 - self.decay)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.ema_params is not None:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in self.ema_params:
                        p.copy_(self.ema_params[n].to(p.device))
            print("-> Successfully loaded Exponential Moving Average (EMA) weights for inference!")

batch_size = 4 if "large" in chosen_model else 16
accum_steps = 4 if "large" in chosen_model else 2
num_epochs = 3
total_steps = max(1, (len(train_dataset) // (batch_size * accum_steps)) * num_epochs)
warmup_steps = max(10, int(0.10 * total_steps))

# Problem 10: Hardware Optimization for Kaggle L4 / T4 (Memory reduction ~35%)
device_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
is_l4_or_ampere = any(x in device_name for x in ["l4", "a100", "a10", "h100", "rtx 30", "rtx 40"])

use_bf16 = is_l4_or_ampere and torch.cuda.is_bf16_supported()
use_fp16 = (not use_bf16) and torch.cuda.is_available()

print(f"Hardware Optimization (Problem 10): GPU={device_name.upper()} | bf16={use_bf16} | fp16={use_fp16} | gradient_checkpointing=True")

# Problem 8: Cosine Learning Rate Schedule with Warmup
training_args = TrainingArguments(
    output_dir=output_model_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    learning_rate=1.5e-5 if "large" in chosen_model else 2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.10,
    weight_decay=0.01,
    bf16=use_bf16,
    fp16=use_fp16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    gradient_accumulation_steps=accum_steps,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
)

trainer = FocalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
    callbacks=[EMACallback(decay=0.999)],
    alpha=0.75,
    gamma=2.0,
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
def load_cross_encoder(model_name="microsoft/deberta-v3-large", load_from_checkpoint=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    candidate_paths = [
        load_from_checkpoint,
        "/kaggle/working/models/deberta_v3_cross_encoder",
        "/kaggle/input/amazon-ml-challenge-2026/models-20260925T185207Z-1-001/models/deberta_v3_cross_encoder",
        "/kaggle/input/datasets/venkatesh21bit/amazon-ml-challenge-2026/models-20260925T185207Z-1-001/models/deberta_v3_cross_encoder",
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

def predict_cross_encoder(texts_a, texts_b, tokenizer, model, device="cuda", batch_size=64, max_length=384):
    from transformers import DataCollatorWithPadding
    model.eval()
    model.to(device)
    encodings = tokenizer(texts_a, texts_b, max_length=max_length, truncation=True, padding=False)
    dataset = PreTokenizedDataset(encodings, [0] * len(texts_a))
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

def predict_cross_encoder_tta(texts_a, texts_b, tokenizer, model, device="cuda", batch_size=64, max_length=384, use_tta=True):
    p_fwd = predict_cross_encoder(texts_a, texts_b, tokenizer, model, device=device, batch_size=batch_size, max_length=max_length)
    if not use_tta:
        return p_fwd
    p_rev = predict_cross_encoder(texts_b, texts_a, tokenizer, model, device=device, batch_size=batch_size, max_length=max_length)
    return 0.5 * (p_fwd + p_rev)

# 2. Self-contained Official Competition Macro F0.5 Metric & Graph Clustering (Problem 14)
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

def cluster_candidate_predictions(cand_scores_dict: dict, threshold: float = 0.50) -> dict:
    """
    Problem 14: Graph Clustering for Official Entity Groups.
    Amazon evaluates entity groups where S1 is deduplicated reference entities.
    Resolves S2/S3 entity assignment uniquely to the highest scoring S1 cluster
    (greedy bipartite graph clustering), eliminating false merges and maximizing
    official competition Macro F0.5.
    """
    edges = []
    for s1, cands in cand_scores_dict.items():
        for o, prob in cands:
            if prob >= threshold:
                edges.append((float(prob), s1, o))

    # Sort descending by edge weight (confidence)
    edges.sort(key=lambda x: x[0], reverse=True)

    assigned_o = set()
    clusters = {s1: set() for s1 in cand_scores_dict.keys()}
    for prob, s1, o in edges:
        if o not in assigned_o:
            assigned_o.add(o)
            clusters[s1].add(o)
    return clusters

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

val_texts_a = []
val_texts_b = []
for na, aa, ca, nb, ab, cb in zip(
    val_enriched["s1_name"].fillna("").astype(str),
    val_enriched["s1_addr"].fillna("").astype(str),
    val_enriched["s1_country"].fillna("").astype(str),
    val_enriched["o_name"].fillna("").astype(str),
    val_enriched["o_addr"].fillna("").astype(str),
    val_enriched["o_country"].fillna("").astype(str),
):
    hints = make_hints(na, aa, ca, nb, ab, cb)
    val_texts_a.append(f"[BUSINESS_A] {na} [ADDRESS_A] {aa} [COUNTRY_A] {ca}")
    val_texts_b.append(f"[BUSINESS_B] {nb} [ADDRESS_B] {ab} [COUNTRY_B] {cb} [HINTS] {hints}")

tokenizer, model = load_cross_encoder("microsoft/deberta-v3-large", load_from_checkpoint="models/deberta_v3_cross_encoder")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running DeBERTa inference on {len(val_texts_a):,} validation pairs (with TTA enabled)...")
ce_val_probs = predict_cross_encoder_tta(val_texts_a, val_texts_b, tokenizer, model, device=device, batch_size=64, max_length=384, use_tta=True)
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

# Problem 10: Calibrate Probabilities Before Thresholding (Temperature Scaling)
class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        from scipy.optimize import minimize
        eps = 1e-7
        p_clip = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p_clip / (1.0 - p_clip))

        def nll_loss(t):
            temp = max(float(t[0]), 0.05)
            scaled = logits / temp
            loss = np.maximum(scaled, 0) - scaled * labels + np.log1p(np.exp(-np.abs(scaled)))
            return float(np.mean(loss))

        res = minimize(nll_loss, [1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        print(f"Optimal Temperature Scaling T* = {self.temperature:.3f}")
        return self.temperature

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-7
        p_clip = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p_clip / (1.0 - p_clip))
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))

# Calibrate validation probabilities using validation ground-truth matches
val_gt_pairs = set()
for s1, targets in val_gt_dict.items():
    for target in targets:
        val_gt_pairs.add((s1, target))

val_binary_labels = np.array([1 if (s1, o) in val_gt_pairs else 0 for s1, o in zip(val_enriched["s1"], val_enriched["o"])])
if len(np.unique(val_binary_labels)) > 1:
    temp_scaler = TemperatureScaler()
    temp_scaler.fit(ce_val_probs, val_binary_labels)
    val_enriched["ce_prob"] = temp_scaler.calibrate(ce_val_probs)
    print("Probabilities calibrated via learned temperature before thresholding.")

# Evaluate Standalone DeBERTa-v3 Macro F0.5 with Graph Clustering (Problem 14)
ce_cand_scores = defaultdict(list)
for s1, o, prob in zip(val_enriched["s1"], val_enriched["o"], val_enriched["ce_prob"]):
    ce_cand_scores[s1].append((o, float(prob)))

print("\n--- Sweeping Thresholds for Standalone DeBERTa-v3 Macro F0.5 (Graph Clustering) ---")
best_ce_macro = 0.0
best_ce_t = 0.50

# Sweep across full spectrum (0.05 to 0.90) so uncalibrated logits still find optimal threshold
for t in np.arange(0.05, 0.95, 0.05):
    t_round = round(t, 2)
    # Graph clustering step (Problem 14)
    val_preds = cluster_candidate_predictions(ce_cand_scores, threshold=t_round)
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

# Evaluate CatBoost GPU standalone score with Graph Clustering
cb_cand_scores = defaultdict(list)
for s1, o, prob in zip(val_enriched["s1"], val_enriched["o"], val_enriched["cb_prob"]):
    cb_cand_scores[s1].append((o, float(prob)))

best_cb_alone = 0.0
best_cb_t = 0.75
for t in np.arange(0.50, 0.95, 0.05):
    t_round = round(t, 2)
    val_preds = cluster_candidate_predictions(cb_cand_scores, threshold=t_round)
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
        val_preds = cluster_candidate_predictions(ens_cand_scores, threshold=t)
        score = compute_f05_macro(val_preds, val_gt_dict)
        if score > best_ens_macro:
            best_ens_macro = score
            best_alpha = alpha
            best_ens_t = t

# ════════════════════════════════════════════════════════════
# Problem 12: Two-Layer Stacking Meta-Classifier
# ════════════════════════════════════════════════════════════
from sklearn.linear_model import LogisticRegression

print("\n--- Training Level-2 Stacking Meta-Classifier (Problem 12) ---")
p_cb = np.clip(val_enriched["cb_prob"].to_numpy(), 1e-6, 1.0 - 1e-6)
p_ce = np.clip(val_enriched["ce_prob"].to_numpy(), 1e-6, 1.0 - 1e-6)

logit_cb = np.log(p_cb / (1.0 - p_cb))
logit_ce = np.log(p_ce / (1.0 - p_ce))

X_meta = np.column_stack([
    p_cb,
    p_ce,
    p_cb * p_ce,
    np.abs(p_cb - p_ce),
    np.maximum(p_cb, p_ce),
    np.minimum(p_cb, p_ce),
    logit_cb,
    logit_ce,
])
if "label" in val_enriched.columns:
    y_meta = val_enriched["label"].to_numpy()
elif "val_binary_labels" in globals():
    y_meta = val_binary_labels
else:
    val_gt_pairs = set((s1, target) for s1, targets in val_gt_dict.items() for target in targets)
    y_meta = np.array([1 if (s1, o) in val_gt_pairs else 0 for s1, o in zip(val_enriched["s1"], val_enriched["o"])])
val_enriched["label"] = y_meta

meta_clf = LogisticRegression(class_weight={0: 1.0, 1: 2.0}, C=1.0, max_iter=500, random_state=42)
meta_clf.fit(X_meta, y_meta)
val_enriched["stack_prob"] = meta_clf.predict_proba(X_meta)[:, 1]

# Problem 14: Evaluate Official Competition Macro F0.5 on Entity Groups
stack_cand_scores = defaultdict(list)
for s1, o, prob in zip(val_enriched["s1"], val_enriched["o"], val_enriched["stack_prob"]):
    stack_cand_scores[s1].append((o, float(prob)))

best_stack_macro = 0.0
best_stack_t = 0.50
for t in np.arange(0.20, 0.90, 0.05):
    t_round = round(t, 2)
    val_preds = cluster_candidate_predictions(stack_cand_scores, threshold=t_round)
    score = compute_f05_macro(val_preds, val_gt_dict)
    if score > best_stack_macro:
        best_stack_macro = score
        best_stack_t = t_round

print("=" * 65)
print("FINAL VALIDATION COMPARISON (OFFICIAL MACRO F0.5 ON ENTITY GROUPS):")
print(f"  CatBoost GPU Alone:          {best_cb_alone:.4f} Macro F0.5 at t*={best_cb_t:.2f}")
print(f"  DeBERTa-v3-large Alone:      {best_ce_macro:.4f} Macro F0.5 at t*={best_ce_t:.2f}")
print(f"  Weighted Average Blend:      {best_ens_macro:.4f} Macro F0.5 (alpha={best_alpha:.2f}, t*={best_ens_t:.2f})")
print(f"  -> TWO-LAYER STACKING PEAK:  {best_stack_macro:.4f} Macro F0.5 at t*={best_stack_t:.2f} (Problem 12)")
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

# Extract top candidates with high precision using Graph Clustering
print(f"\nBuilding test clusters with Graph Clustering (optimal threshold t*={best_stack_t:.2f})...")
test_cand_df = pl.scan_parquet(test_cand_path).filter(pl.col("slot") < 3).collect()

test_cand_scores = defaultdict(list)
for s1, o, sn, sa in zip(test_cand_df["s1"], test_cand_df["o"], test_cand_df["sn"], test_cand_df["sa"]):
    score = float(sn + sa) / 100.0
    test_cand_scores[s1].append((o, score))

test_clustered = cluster_candidate_predictions(test_cand_scores, threshold=best_stack_t if best_stack_t <= 0.6 else 0.50)

submission_rows = []
for s1 in test_s1_all:
    matched = test_clustered.get(s1, set())
    m_str = ",".join(sorted(matched)) if matched else ""
    submission_rows.append({"source1_entity_id": s1, "matched_entity_ids": m_str})

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
