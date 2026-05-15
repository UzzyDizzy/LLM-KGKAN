# data.py (FIXED)

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import spacy
from kg_utils import ConceptNet
import pandas as pd
from config import LLMKGKANConfig
from difflib import SequenceMatcher

nlp = spacy.load("en_core_web_sm")

DEP2ID = {}

class ABSADataset(Dataset):
    def __init__(self, file_path, cfg, kg, domain_id=0):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        # 🔥 ADD THIS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_len = cfg.max_len
        self.domain_id = domain_id
        self.kg = kg
        self.label2id = cfg.LABEL2ID

        df = pd.read_csv(file_path)

        # ---- GROUP BY SENTENCE ----
        self.data = []
        for text, group in df.groupby("text"):
            aspects = group["aspect"].tolist()
            sentiments = group["polarity"].tolist()
            starts = group["from"].tolist()
            ends = group["to"].tolist()

            self.data.append((text, aspects, sentiments, starts, ends))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize_text(text):
        """Normalize text for KG entity matching."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def build_dep_graph(self, doc):
        # 🔥 truncate safely
        doc = doc[:self.max_len]

        n = len(doc)

        adj = torch.zeros(n, n).bool()
        rel_ids = torch.zeros(n, n).long()

        # 🔥 remap old indices → new indices
        idx_map = {token.i: new_i for new_i, token in enumerate(doc)}

        for new_i, token in enumerate(doc):

            head_i = token.head.i

            # 🔥 skip if head is outside truncated doc
            if head_i not in idx_map:
                continue

            new_head_i = idx_map[head_i]

            if new_head_i != new_i:
                adj[new_i][new_head_i] = 1

                dep = token.dep_
                if dep not in DEP2ID:
                    DEP2ID[dep] = len(DEP2ID)

                rel_ids[new_i][new_head_i] = DEP2ID[dep]

        # ---- pad ----
        adj_padded = torch.zeros(self.max_len, self.max_len)
        rel_padded = torch.zeros(self.max_len, self.max_len).long()

        adj_padded[:n, :n] = adj
        rel_padded[:n, :n] = rel_ids

        return adj_padded, rel_padded

    def __getitem__(self, idx):
        text, aspects, sentiments, starts, ends = self.data[idx]

        # ---- TOKENIZATION (SPACY) ----
        doc = nlp(text if isinstance(text, str) else " ".join(text))

        # 🔥 TRUNCATE DOC TO max_len
        doc = doc[:self.max_len]

        tokens = [t.text for t in doc]

        # ---- HF TOKENIZER ----
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # ---- CHAR → TOKEN SPANS ----
        aspect_spans = []

        sentiment_map = {
            "positive": "POS",
            "negative": "NEG",
            "neutral": "NEU"
        }

        for aspect, sentiment, start_char, end_char in zip(aspects, sentiments, starts, ends):
            if aspect == "NULL":
                continue

            sentiment = sentiment_map.get(sentiment.lower(), "NEU")

            token_start, token_end = None, None

            for i, token in enumerate(doc):
                if token.idx <= start_char < token.idx + len(token):
                    token_start = i
                if token.idx < end_char <= token.idx + len(token):
                    token_end = i + 1

            if token_start is not None and token_end is not None:
                aspect_spans.append((token_start, token_end, sentiment))

        # ---- LABELS (BIO) ----
        word_ids = enc.word_ids()
        labels = torch.full((self.max_len,), -100)

        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            for start, end, sent in aspect_spans:
                if word_id == start:
                    labels[i] = self.label2id[f"B-{sent}"]
                elif start < word_id < end:
                    labels[i] = self.label2id[f"I-{sent}"]

        # fill O
        for i in range(len(labels)):
            if labels[i] == -100:
                labels[i] = self.label2id["O"]

        # ---- DEP GRAPH ----
        adj, rel_ids = self.build_dep_graph(doc)

        # ---- KG ----
        aspect_graphs = []

        for aspect in aspects:
            if aspect == "NULL":
                continue

            aspect_tokens = aspect.lower().split()
            h, r, t = self.kg.get_triples_for_tokens(aspect_tokens)

            if len(h) == 0:
                continue

            aspect_graphs.append((h, r, t))

        heads, rels, tails = [], [], []
        aspect_ids = []

        for i, (h, r, t) in enumerate(aspect_graphs):
            heads.extend(h)
            rels.extend(r)
            tails.extend(t)
            aspect_ids.extend([i] * len(h))

        if len(heads) == 0:
            heads, rels, tails = [0], [0], [0]

        kg_heads = torch.tensor(heads).long()
        kg_rels = torch.tensor(rels).long()
        kg_tails = torch.tensor(tails).long()
        kg_mask = torch.ones(len(heads), dtype=torch.float)
        kg_token_map = torch.zeros(self.max_len, len(heads)).float()
        kg_aspect_ids = torch.tensor(aspect_ids if len(aspect_ids) > 0 else [0]).long()

        # ---- TOKEN ↔ ENTITY MATCH ----
        from difflib import SequenceMatcher

        def token_entity_match(token, entity):
            token = token.lower()
            entity = entity.lower()

            if token == entity:
                return True
            if token in entity or entity in token:
                return True
            if token in entity.split():
                return True
            if SequenceMatcher(None, token, entity).ratio() > 0.7:
                return True
            return False

        for i, tok in enumerate(tokens[:self.max_len]):
            tok_norm = self.normalize_text(tok)

            for j, h_id in enumerate(heads):
                ent = self.normalize_text(self.kg.id2ent[h_id])

                if token_entity_match(tok_norm, ent):
                    kg_token_map[i][j] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "domain_ids": torch.tensor(self.domain_id).long(),
            "dep_adj": adj,
            "dep_rel_ids": rel_ids,
            "kg_heads": kg_heads,
            "kg_rels": kg_rels,
            "kg_tails": kg_tails,
            "kg_mask": kg_mask,
            "kg_token_map": kg_token_map,
            "kg_aspect_ids": kg_aspect_ids,
            "aspect_spans": aspect_spans
        }


def sentiment_to_id(s):
    return {"positive": 0, "negative": 1, "neutral": 2}.get(s, 2)


def pad_tensor_list(tensor_list, pad_value=0):
    max_len = max(t.size(0) for t in tensor_list)
    padded = []

    for t in tensor_list:
        pad_size = max_len - t.size(0)
        if pad_size > 0:
            pad = torch.full((pad_size,), pad_value, dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)

    return torch.stack(padded)


def collate_fn(batch):
    out = {}

    for key in ["input_ids", "attention_mask", "labels", "domain_ids", "dep_adj", "dep_rel_ids"]:
        out[key] = torch.stack([b[key] for b in batch])

    out["kg_heads"] = pad_tensor_list([b["kg_heads"] for b in batch], 0)
    out["kg_rels"] = pad_tensor_list([b["kg_rels"] for b in batch], 0)
    out["kg_tails"] = pad_tensor_list([b["kg_tails"] for b in batch], 0)
    out["kg_mask"] = pad_tensor_list([b["kg_mask"] for b in batch], 0)

    max_k = out["kg_heads"].size(1)
    T = batch[0]["kg_token_map"].size(0)

    maps = []
    for b in batch:
        m = b["kg_token_map"]
        pad_k = max_k - m.size(1)

        if pad_k > 0:
            pad = torch.zeros(T, pad_k)
            m = torch.cat([m, pad], dim=1)

        maps.append(m)

    out["kg_token_map"] = torch.stack(maps)
    out["kg_aspect_ids"] = pad_tensor_list([b["kg_aspect_ids"] for b in batch], 0)

    # ⚠️ KEEP AS LIST (DO NOT STACK)
    out["aspect_spans"] = [b["aspect_spans"] for b in batch]

    return out

def build_transfer_dataset(cfg, source_file, target_file):
    src = ABSADataset(source_file, cfg, domain_id=0)
    tgt = ABSADataset(target_file, cfg, domain_id=1)
    return src, tgt