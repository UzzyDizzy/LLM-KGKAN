"""
scripts/data_utils.py — Unified data loading for all 11 ABSA datasets.

Loads CSV → tokenizes → BIO tags → dependency graphs → KG subgraphs.
Provides format converters for each baseline model.
"""

import os, json, pickle, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    DOMAIN_FILES, LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    SENTIMENT_MAP, IGNORE_INDEX, SEED,
)


# ══════════════════════════════════════════════════════════════════════════
# CSV LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_domain_csv(domain_key: str) -> pd.DataFrame:
    """Load a domain CSV and standardize columns."""
    path = DOMAIN_FILES[domain_key]
    df = pd.read_csv(path)
    # Ensure required columns exist
    required = ["text", "aspect", "polarity"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df


def get_dataset_stats(domain_key: str) -> dict:
    """Get dataset statistics for Tables 2 & 5."""
    df = load_domain_csv(domain_key)
    pol_counts = df["polarity"].value_counts().to_dict()
    return {
        "domain": domain_key,
        "total": len(df),
        "positive": pol_counts.get("positive", 0),
        "negative": pol_counts.get("negative", 0),
        "neutral": pol_counts.get("neutral", 0),
        "unique_aspects": df["aspect"].nunique(),
    }


# ══════════════════════════════════════════════════════════════════════════
# BIO TAG GENERATION
# ══════════════════════════════════════════════════════════════════════════

def text_to_bio_tags(text: str, aspect: str, polarity: str,
                     from_idx: int, to_idx: int) -> List[str]:
    """
    Convert a sentence + aspect annotation to BIO tag sequence.
    Uses character offsets (from, to) if valid, else falls back to string matching.
    """
    tokens = text.split()
    tags = ["O"] * len(tokens)

    sentiment = SENTIMENT_MAP.get(polarity, "NEU")

    # Try character offset approach
    if from_idx >= 0 and to_idx > from_idx:
        # Find which tokens fall within the character span
        char_pos = 0
        for i, token in enumerate(tokens):
            token_start = char_pos
            token_end = char_pos + len(token)
            if token_start >= from_idx and token_end <= to_idx:
                tags[i] = f"B-{sentiment}" if (i == 0 or tags[i-1] == "O") else f"I-{sentiment}"
            elif token_start < to_idx and token_end > from_idx:
                # Partial overlap
                tags[i] = f"B-{sentiment}" if (i == 0 or tags[i-1] == "O") else f"I-{sentiment}"
            char_pos = token_end + 1  # +1 for space
    else:
        # Fallback: string matching
        if aspect and aspect != "implicit" and aspect != "NULL":
            aspect_tokens = aspect.split()
            for i in range(len(tokens) - len(aspect_tokens) + 1):
                if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                    tags[i] = f"B-{sentiment}"
                    for j in range(1, len(aspect_tokens)):
                        tags[i+j] = f"I-{sentiment}"
                    break
            else:
                # Case-insensitive fallback
                tokens_lower = [t.lower() for t in tokens]
                aspect_lower = [t.lower() for t in aspect_tokens]
                for i in range(len(tokens_lower) - len(aspect_lower) + 1):
                    if tokens_lower[i:i+len(aspect_lower)] == aspect_lower:
                        tags[i] = f"B-{sentiment}"
                        for j in range(1, len(aspect_lower)):
                            tags[i+j] = f"I-{sentiment}"
                        break

    return tags


def merge_bio_tags(tag_lists: List[List[str]]) -> List[str]:
    """Merge multiple BIO tag lists for the same sentence (multi-aspect)."""
    if not tag_lists:
        return []
    merged = list(tag_lists[0])
    for tags in tag_lists[1:]:
        for i, tag in enumerate(tags):
            if i < len(merged) and tag != "O" and merged[i] == "O":
                merged[i] = tag
    return merged


# ══════════════════════════════════════════════════════════════════════════
# UNIFIED ABSA DATASET
# ══════════════════════════════════════════════════════════════════════════

class UnifiedABSADataset(Dataset):
    """
    Unified dataset that can feed any model.
    Stores raw text + BIO tags. Model-specific encoding is done at collate time.
    """

    def __init__(self, domain_key: str, tokenizer=None, max_len: int = 128,
                 kg=None, k_shot: Optional[int] = None, seed: int = SEED):
        self.domain_key = domain_key
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.kg = kg

        df = load_domain_csv(domain_key)

        # Group by text to handle multi-aspect sentences
        grouped = defaultdict(list)
        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            aspect = str(row.get("aspect", "")).strip()
            polarity = str(row.get("polarity", "neutral")).strip()
            from_idx = int(row.get("from", -1)) if pd.notna(row.get("from")) else -1
            to_idx = int(row.get("to", -1)) if pd.notna(row.get("to")) else -1
            grouped[text].append((aspect, polarity, from_idx, to_idx))

        self.samples = []
        for text, annotations in grouped.items():
            tokens = text.split()
            if len(tokens) == 0:
                continue

            # Generate BIO tags
            tag_lists = []
            aspects = []
            for aspect, polarity, from_idx, to_idx in annotations:
                tags = text_to_bio_tags(text, aspect, polarity, from_idx, to_idx)
                tag_lists.append(tags)
                aspects.append(aspect)

            merged_tags = merge_bio_tags(tag_lists)

            # Truncate
            tokens = tokens[:max_len]
            merged_tags = merged_tags[:max_len]

            tag_ids = [LABEL2ID.get(t, 0) for t in merged_tags]

            self.samples.append({
                "text": text,
                "tokens": tokens,
                "tags": merged_tags,
                "tag_ids": tag_ids,
                "aspects": aspects,
                "domain": domain_key,
            })

        # Few-shot sampling
        if k_shot is not None and k_shot > 0:
            self.samples = self._few_shot_sample(k_shot, seed)

    def _few_shot_sample(self, k: int, seed: int) -> list:
        """Sample k examples per sentiment class."""
        rng = random.Random(seed)
        by_class = defaultdict(list)
        for s in self.samples:
            # Determine dominant sentiment
            sentiments = set()
            for t in s["tags"]:
                if t.startswith("B-"):
                    sentiments.add(t.split("-")[1])
            if not sentiments:
                sentiments.add("O")
            for sent in sentiments:
                by_class[sent].append(s)

        sampled = []
        seen_texts = set()
        for cls, items in by_class.items():
            rng.shuffle(items)
            count = 0
            for item in items:
                if item["text"] not in seen_texts and count < k:
                    sampled.append(item)
                    seen_texts.add(item["text"])
                    count += 1
        return sampled

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ══════════════════════════════════════════════════════════════════════════
# COLLATE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def collate_for_bert(batch, tokenizer, max_len=128):
    """Collate for BERT-based models (BERT-UDA, KETGM, TransProto)."""
    texts = [s["text"] for s in batch]
    tag_ids_list = [s["tag_ids"] for s in batch]

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    # Align BIO tags to subword tokens
    B = len(batch)
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)

    for i in range(B):
        word_ids = encoded.word_ids(i)
        prev_word = None
        for j, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid != prev_word:
                if wid < len(tag_ids_list[i]):
                    labels[i, j] = tag_ids_list[i][wid]
            prev_word = wid

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "texts": texts,
    }


def collate_for_llm(batch, tokenizer, max_len=128):
    """Collate for LLM-based models (LLM-KGKAN, LLMSynABSA)."""
    texts = [s["text"] for s in batch]
    tag_ids_list = [s["tag_ids"] for s in batch]
    aspect_spans = []

    for s in batch:
        spans = []
        in_span = False
        start = 0
        for j, t in enumerate(s["tags"]):
            if t.startswith("B-"):
                if in_span:
                    spans.append((start, j))
                start = j
                in_span = True
            elif t.startswith("I-") and in_span:
                continue
            else:
                if in_span:
                    spans.append((start, j))
                    in_span = False
        if in_span:
            spans.append((start, len(s["tags"])))
        aspect_spans.append(spans)

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    B = len(batch)
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)

    for i in range(B):
        tids = tag_ids_list[i]
        seq_len = min(len(tids), max_len)
        labels[i, :seq_len] = torch.tensor(tids[:seq_len], dtype=torch.long)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "texts": texts,
        "aspect_spans": aspect_spans,
    }


def collate_simple(batch):
    """Simple collate — returns raw samples as-is for models with custom pipelines."""
    return batch


# ══════════════════════════════════════════════════════════════════════════
# TRAIN / VAL / TEST SPLITS
# ══════════════════════════════════════════════════════════════════════════

def split_dataset(dataset: Dataset, val_ratio: float = 0.1,
                  seed: int = SEED) -> Tuple[Subset, Subset]:
    """Split dataset into train/val."""
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(n * (1 - val_ratio))
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])


def build_dataloaders(src_domain: str, tgt_domain: str,
                      tokenizer=None, max_len: int = 128,
                      batch_size: int = 16, kg=None,
                      k_shot: Optional[int] = None,
                      collate_fn=None) -> dict:
    """Build train/val/test dataloaders for a src→tgt transfer pair."""

    src_data = UnifiedABSADataset(src_domain, tokenizer, max_len, kg)
    tgt_data = UnifiedABSADataset(tgt_domain, tokenizer, max_len, kg, k_shot=k_shot)

    tgt_train, tgt_val = split_dataset(tgt_data)

    if collate_fn is None:
        collate_fn = collate_simple

    return {
        "src_loader": DataLoader(src_data, batch_size=batch_size,
                                 shuffle=True, collate_fn=collate_fn,
                                 drop_last=False),
        "tgt_train_loader": DataLoader(tgt_train, batch_size=batch_size,
                                       shuffle=True, collate_fn=collate_fn,
                                       drop_last=False),
        "tgt_val_loader": DataLoader(tgt_val, batch_size=batch_size,
                                     shuffle=False, collate_fn=collate_fn,
                                     drop_last=False),
        "tgt_full_loader": DataLoader(tgt_data, batch_size=batch_size,
                                      shuffle=False, collate_fn=collate_fn,
                                      drop_last=False),
        "src_dataset": src_data,
        "tgt_dataset": tgt_data,
    }
