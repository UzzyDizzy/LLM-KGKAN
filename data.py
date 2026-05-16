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
    def __init__(self, file_path, cfg, kg, domain_id=0, use_labels=True, use_gold_aspects=True):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_len = cfg.max_len
        self.domain_id = domain_id
        self.use_labels = use_labels
        self.use_gold_aspects = use_gold_aspects
        self.kg = kg
        self.label2id = cfg.LABEL2ID
        self.ignore_index = cfg.ignore_index
        self.num_dep_relations = cfg.num_dep_relations

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

    def build_dep_graph(self, doc, word_ids=None, attention_mask=None):
        """Build a dependency graph in tokenizer-position space.

        spaCy gives word-level dependency indices, while the LLM receives
        subword token positions. We attach each word-level dependency edge to
        the first tokenizer position for that word so syntax, labels, and hidden
        states all share the same axis.
        """
        doc = doc[:self.max_len]
        n = len(doc)

        adj_padded = torch.zeros(self.max_len, self.max_len).bool()
        rel_padded = torch.zeros(self.max_len, self.max_len).long()

        idx_map = {token.i: new_i for new_i, token in enumerate(doc)}
        word_to_token = {}

        if word_ids is None:
            word_to_token = {i: i for i in range(min(n, self.max_len))}
        else:
            for token_pos, word_id in enumerate(word_ids[:self.max_len]):
                if word_id is None or word_id >= n:
                    continue
                if attention_mask is not None and int(attention_mask[token_pos].item()) == 0:
                    continue
                word_to_token.setdefault(word_id, token_pos)

        for new_i, token in enumerate(doc):
            head_i = token.head.i
            if head_i not in idx_map:
                continue

            new_head_i = idx_map[head_i]
            if new_head_i != new_i:
                child_pos = word_to_token.get(new_i)
                head_pos = word_to_token.get(new_head_i)
                if child_pos is None or head_pos is None:
                    continue

                adj_padded[child_pos][head_pos] = 1

                dep = token.dep_
                if dep not in DEP2ID:
                    DEP2ID[dep] = len(DEP2ID)

                dep_id = DEP2ID[dep]
                rel_padded[child_pos][head_pos] = dep_id if dep_id < self.num_dep_relations else 0

        return adj_padded, rel_padded

    def detect_candidate_aspects(self, doc):
        """Aspect candidates for unlabeled/eval inputs.

        Gold aspect spans are valid for labeled source/few-shot training, but
        not at target inference time. We use noun chunks plus KG-covered content
        words as detected candidates for relational retrieval.
        """
        candidates = []
        seen = set()

        def add_span(start, end, text):
            if end <= start:
                return
            key = (start, end)
            if key in seen:
                return
            seen.add(key)
            candidates.append((start, end, text))

        try:
            for chunk in doc.noun_chunks:
                if chunk.start < self.max_len:
                    add_span(chunk.start, min(chunk.end, self.max_len), chunk.text)
        except Exception:
            pass

        for i, token in enumerate(doc[:self.max_len]):
            text = token.text.strip()
            if not text:
                continue
            pos = getattr(token, "pos_", "")
            is_content = pos in {"NOUN", "PROPN", "ADJ", "VERB"}
            is_stop = bool(getattr(token, "is_stop", False))
            is_punct = bool(getattr(token, "is_punct", False))
            norm = self.normalize_text(text)
            in_kg = self.kg is not None and norm in getattr(self.kg, "graph", {})
            if (is_content and not is_stop and not is_punct) or in_kg:
                add_span(i, i + 1, text)

        return candidates

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
        word_ids = enc.word_ids()

        # ---- CHAR → TOKEN SPANS ----
        gold_word_aspect_spans = []
        gold_word_aspect_terms = []

        sentiment_map = {
            "positive": "POS",
            "negative": "NEG",
            "neutral": "NEU"
        }

        for aspect, sentiment, start_char, end_char in zip(aspects, sentiments, starts, ends):
            if not isinstance(aspect, str) or aspect == "NULL":
                continue

            sentiment = sentiment_map.get(sentiment.lower(), "NEU")
            start_char, end_char = int(start_char), int(end_char)

            token_start, token_end = None, None

            for i, token in enumerate(doc):
                if token.idx <= start_char < token.idx + len(token):
                    token_start = i
                if token.idx < end_char <= token.idx + len(token):
                    token_end = i + 1

            if token_start is not None and token_end is not None:
                gold_word_aspect_spans.append((token_start, token_end, sentiment))
                gold_word_aspect_terms.append(aspect)

        # ---- LABELS (BIO) ----
        word_label_ids = [self.label2id["O"]] * len(doc)
        for start, end, sent in gold_word_aspect_spans:
            if start >= len(word_label_ids):
                continue
            end = min(end, len(word_label_ids))
            word_label_ids[start] = self.label2id[f"B-{sent}"]
            for word_id in range(start + 1, end):
                word_label_ids[word_id] = self.label2id[f"I-{sent}"]

        labels = torch.full((self.max_len,), self.ignore_index)
        prev_word_id = None
        for token_pos, word_id in enumerate(word_ids[:self.max_len]):
            if word_id is None or word_id >= len(word_label_ids):
                continue
            if int(attention_mask[token_pos].item()) == 0:
                continue
            if word_id != prev_word_id:
                labels[token_pos] = word_label_ids[word_id]
            prev_word_id = word_id

        if not self.use_labels:
            labels.fill_(self.ignore_index)

        # Tokenizer-position spans used by the relational stream.
        if self.use_gold_aspects:
            rel_word_aspect_spans = gold_word_aspect_spans
            rel_word_aspect_terms = gold_word_aspect_terms
        else:
            detected = self.detect_candidate_aspects(doc)
            rel_word_aspect_spans = [(start, end, "UNK") for start, end, _ in detected]
            rel_word_aspect_terms = [text for _, _, text in detected]

        aspect_spans = []
        aspect_terms = []
        for (start, end, sent), aspect in zip(rel_word_aspect_spans, rel_word_aspect_terms):
            positions = [
                token_pos
                for token_pos, word_id in enumerate(word_ids[:self.max_len])
                if word_id is not None
                and start <= word_id < end
                and int(attention_mask[token_pos].item()) == 1
            ]
            if positions:
                aspect_spans.append((min(positions), max(positions) + 1, sent))
                aspect_terms.append(aspect)

        # ---- DEP GRAPH ----
        adj, rel_ids = self.build_dep_graph(doc, word_ids, attention_mask)

        # ---- KG ----
        aspect_graphs = []

        if self.kg is not None:
            for aspect in aspect_terms:
                aspect_tokens = aspect.lower().split()
                h, r, t = self.kg.get_triples_for_tokens(aspect_tokens)

                if len(h) == 0:
                    aspect_graphs.append(([], [], []))
                else:
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
            kg_mask = torch.zeros(1, dtype=torch.float)
            aspect_ids = [0]
        else:
            kg_mask = torch.ones(len(heads), dtype=torch.float)

        kg_heads = torch.tensor(heads).long()
        kg_rels = torch.tensor(rels).long()
        kg_tails = torch.tensor(tails).long()
        kg_token_map = torch.zeros(self.max_len, len(heads)).float()
        kg_aspect_ids = torch.tensor(aspect_ids if len(aspect_ids) > 0 else [0]).long()
        kg_num_aspects = torch.tensor(len(aspect_spans)).long()

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

        if self.kg is not None:
            for token_pos, word_id in enumerate(word_ids[:self.max_len]):
                if word_id is None or word_id >= len(tokens):
                    continue
                if int(attention_mask[token_pos].item()) == 0:
                    continue

                tok_norm = self.normalize_text(tokens[word_id])

                for j, h_id in enumerate(heads):
                    if kg_mask[j].item() == 0:
                        continue
                    ent = self.normalize_text(self.kg.id2ent.get(h_id, ""))

                    if token_entity_match(tok_norm, ent):
                        kg_token_map[token_pos][j] = 1

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
            "kg_num_aspects": kg_num_aspects,
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
    out["kg_num_aspects"] = torch.stack([b["kg_num_aspects"] for b in batch])

    # ⚠️ KEEP AS LIST (DO NOT STACK)
    out["aspect_spans"] = [b["aspect_spans"] for b in batch]

    return out

def build_transfer_dataset(cfg, source_file, target_file, kg=None):
    src = ABSADataset(source_file, cfg, kg, domain_id=0)
    tgt = ABSADataset(target_file, cfg, kg, domain_id=1)
    return src, tgt
