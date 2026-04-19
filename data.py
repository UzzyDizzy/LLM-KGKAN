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
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        self.max_len = cfg.max_len
        self.domain_id = domain_id
        self.kg = kg

        self.label2id = cfg.LABEL2ID

        df = pd.read_csv(file_path)

        self.data = []

        for sent_id, group in df.groupby("sentence_id"):
            tokens = group["token"].tolist()
            aspects = group["aspect"].tolist()
            sentiments = group["sentiment"].tolist()

            self.data.append((tokens, aspects, sentiments))

    def __len__(self):
        return len(self.data)

    def build_dep_graph(self, doc):
        n = len(doc)

        adj = torch.zeros(n, n).bool()
        rel_ids = torch.zeros(n, n).long()

        for token in doc:
            if token.head.i != token.i:
                adj[token.i][token.head.i] = 1

                dep = token.dep_
                if dep not in DEP2ID:
                    DEP2ID[dep] = len(DEP2ID)

                rel_ids[token.i][token.head.i] = DEP2ID[dep]

        # PAD
        adj_padded = torch.zeros(self.max_len, self.max_len)
        rel_padded = torch.zeros(self.max_len, self.max_len).long()

        adj_padded[:n, :n] = adj
        rel_padded[:n, :n] = rel_ids

        return adj_padded, rel_padded

    def normalize_text(s):
        return s.lower().replace("_", " ").strip()

    def __getitem__(self, idx):
        tokens, aspects, sentiments = self.data[idx]

        doc = nlp(" ".join(tokens))
        tokens = [t.text for t in doc]

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

        # ---- LABELS (convert triplets → token tags) ----
        word_ids = enc.word_ids()
        labels = torch.full((self.max_len,), -100)

        aspect_spans = []

        for aspect, sentiment in zip(aspects, sentiments):
            if aspect == "NULL":
                continue

            sentiment = sentiment.upper()
            aspect_tokens = [self.normalize_text(t) for t in aspect.split()]

            token_norm = [self.normalize_text(t) for t in tokens]

            for i in range(len(token_norm)):
                if token_norm[i:i+len(aspect_tokens)] == aspect_tokens:
                    aspect_spans.append((i, i+len(aspect_tokens), sentiment))

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            for start, end, sent in aspect_spans:
                if word_id == start:
                    labels[idx] = self.label2id[f"B-{sent}"]
                elif start < word_id < end:
                    labels[idx] = self.label2id[f"I-{sent}"]
        # for Non-aspect tokens
        for i in range(len(labels)):
            if labels[i] == -100:
                labels[i] = self.label2id["O"]

        # ---- dependency ----
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

        k = len(heads)

        if k == 0:
            k = 1
            heads = [0]
            rels = [0]
            tails = [0]

        kg_heads = torch.tensor(heads).long()
        kg_rels = torch.tensor(rels).long()
        kg_tails = torch.tensor(tails).long()
        kg_mask = torch.ones(len(heads), dtype=torch.float)
        kg_token_map = torch.zeros(self.max_len, len(heads)).float()
        kg_aspect_ids = torch.tensor(aspect_ids).long()


        def token_entity_match(token, entity):
            """
            Strong matching:
            - exact
            - substring
            - token overlap
            - fuzzy (lightweight)
            """
            token = token.lower()
            entity = entity.lower()

            # 1. exact
            if token == entity:
                return True

            # 2. substring
            if token in entity or entity in token:
                return True

            # 3. multi-token overlap
            ent_tokens = entity.split()
            if token in ent_tokens:
                return True

            # 4. simple fuzzy (character overlap ratio)
            # overlap = len(set(token) & set(entity))
            # ratio = overlap / max(len(token), len(entity))

            # if ratio > 0.6:
            #     return True
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
            "aspect_spans": torch.tensor(aspect_spans) if len(aspect_spans) > 0 else torch.zeros(1,3),
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

    # standard tensors
    for key in ["input_ids", "attention_mask", "labels", "domain_ids", "dep_adj", "dep_rel_ids"]:
        out[key] = torch.stack([b[key] for b in batch])

    # KG variable tensors
    out["kg_heads"] = pad_tensor_list([b["kg_heads"] for b in batch], 0)
    out["kg_rels"] = pad_tensor_list([b["kg_rels"] for b in batch], 0)
    out["kg_tails"] = pad_tensor_list([b["kg_tails"] for b in batch], 0)

    # mask must match padded size
    out["kg_mask"] = pad_tensor_list([b["kg_mask"] for b in batch], 0)

    # kg_token_map → (T, K)
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
    out["aspect_spans"] = [b["aspect_spans"] for b in batch]

    return out

def build_transfer_dataset(cfg, source_file, target_file):
    src = ABSADataset(source_file, cfg, domain_id=0)
    tgt = ABSADataset(target_file, cfg, domain_id=1)
    return src, tgt