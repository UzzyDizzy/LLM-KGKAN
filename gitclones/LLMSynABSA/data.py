# data.py — Data loading with proper grouping and train/test splitting
import torch
import pandas as pd
import numpy as np
from config import MAX_LEN


def parse_data(file_path):
    """
    Load CSV and group all aspects per unique sentence.
    Paper formulation: each text has ALL its aspect-sentiment labels simultaneously.
    Returns: list of (text, [aspects]) where aspects is list of dicts.
    """
    df = pd.read_csv(file_path)
    grouped = {}

    for _, row in df.iterrows():
        text = row["text"]
        if text not in grouped:
            grouped[text] = []
        if pd.notna(row.get("aspect")) and pd.notna(row.get("polarity")):
            grouped[text].append({
                "term": row["aspect"],
                "polarity": row["polarity"],
                "from": int(row["from"]),
                "to": int(row["to"]),
            })

    data = [(text, aspects) for text, aspects in grouped.items()]
    return data


def split_data(data, train_ratio=0.8, seed=42):
    """
    Split data into train/test by unique sentences.
    Paper Table 1 has roughly 75-80% train / 20-25% test.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    split = int(len(data) * train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]
    train_data = [data[i] for i in train_idx]
    test_data = [data[i] for i in test_idx]
    return train_data, test_data


def create_split_labels(text, aspects, tokenizer):
    """
    Create separate aspect (O/B/I) and sentiment (O/POS/NEG/NEU) label sequences
    for auxiliary classifiers (Eq 2, 3).
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    offsets = enc["offset_mapping"]
    aspect_labels = []
    sentiment_labels = []

    for s, e in offsets:
        if s == e == 0:
            aspect_labels.append("PAD")
            sentiment_labels.append("PAD")
        else:
            aspect_labels.append("O")
            sentiment_labels.append("O")

    for asp in aspects:
        start_char = asp["from"]
        end_char = asp["to"]
        polarity = asp["polarity"]

        if polarity == "positive":
            tag = "POS"
        elif polarity == "negative":
            tag = "NEG"
        else:
            tag = "NEU"

        started = False
        for i, (s, e) in enumerate(offsets):
            if s == e == 0:
                continue
            if e <= start_char:
                continue
            if s >= end_char:
                break
            if not started:
                aspect_labels[i] = "B"
                sentiment_labels[i] = tag
                started = True
            else:
                aspect_labels[i] = "I"
                sentiment_labels[i] = tag

    return aspect_labels, sentiment_labels


def create_unified_labels(text, aspects, tokenizer):
    """
    Create unified 7-class label sequence (paper Section 3.1):
    {O, B-POS, I-POS, B-NEG, I-NEG, B-NEU, I-NEU}
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    offsets = enc["offset_mapping"]
    labels = []

    for s, e in offsets:
        if s == e == 0:
            labels.append("PAD")
        else:
            labels.append("O")

    for asp in aspects:
        start_char = asp["from"]
        end_char = asp["to"]
        polarity = asp["polarity"]

        if polarity == "positive":
            tag = "POS"
        elif polarity == "negative":
            tag = "NEG"
        else:
            tag = "NEU"

        started = False
        for i, (s, e) in enumerate(offsets):
            if s == e == 0:
                continue
            if e <= start_char:
                continue
            if s >= end_char:
                break
            if not started:
                labels[i] = f"B-{tag}"
                started = True
            else:
                labels[i] = f"I-{tag}"

    return labels


class ABSADataset(torch.utils.data.Dataset):
    """Source domain dataset with full labels."""

    def __init__(self, data, tokenizer, label_map, aspect_label_map, sentiment_label_map):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.aspect_label_map = aspect_label_map
        self.sentiment_label_map = sentiment_label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, aspects = self.data[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Unified labels for L_seq (Eq 26)
        unified = create_unified_labels(text, aspects, self.tokenizer)
        unified_ids = []
        for l in unified:
            if l == "PAD":
                unified_ids.append(-100)
            else:
                unified_ids.append(self.label_map[l])

        # Separate labels for auxiliary classifiers (Eq 2, 3)
        aspect_labels, sentiment_labels = create_split_labels(text, aspects, self.tokenizer)
        asp_ids, sent_ids = [], []
        for a, s in zip(aspect_labels, sentiment_labels):
            if a == "PAD":
                asp_ids.append(-100)
                sent_ids.append(-100)
            else:
                asp_ids.append(self.aspect_label_map[a])
                sent_ids.append(self.sentiment_label_map[s])

        # Pad/truncate to MAX_LEN
        def pad(seq):
            return seq[:MAX_LEN] + [-100] * max(0, MAX_LEN - len(seq))

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(pad(unified_ids), dtype=torch.long),
            "aspect_labels": torch.tensor(pad(asp_ids), dtype=torch.long),
            "sentiment_labels": torch.tensor(pad(sent_ids), dtype=torch.long),
            "text": text,
        }


class UnlabeledDataset(torch.utils.data.Dataset):
    """Target domain dataset (no labels during training)."""

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, _ = self.data[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "text": text,
        }
