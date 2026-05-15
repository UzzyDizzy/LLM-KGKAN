"""
ABSA data parsers  →  token-level BIO-sentiment tags.

Supports two input formats:
  1. SemEval-2014 XML  (parse_semeval_xml)
  2. CSV with columns: id, text, aspect, category, polarity, from, to
     (parse_absa_csv)

Both produce a list of dicts, each with:
  tokens, bio_tags, pos_tags, dep_tags, text
"""
import os, csv, pickle, xml.etree.ElementTree as ET
from typing import List, Dict
from collections import OrderedDict
import spacy

from config import Config

# ── polarity string → tag suffix ──────────────────────────────────────
_POL = {"positive": "POS", "negative": "NEG", "neutral": "NEU", "conflict": "NEU"}


def _char_spans(doc) -> List[tuple]:
    """Return (start_char, end_char) for each spaCy token."""
    return [(t.idx, t.idx + len(t.text)) for t in doc]


def _assign_bio(bio, spans, a_start, a_end, suffix):
    """Assign BIO tags for one aspect span to token-level bio list."""
    first = True
    for idx, (cs, ce) in enumerate(spans):
        # token overlaps with aspect span
        if cs >= a_start and ce <= a_end:
            bio[idx] = f"B-{suffix}" if first else f"I-{suffix}"
            first = False


# ── CSV parser (restaurant.csv, laptop.csv, device.csv, service.csv) ──

def parse_absa_csv(csv_path: str, nlp=None) -> List[Dict]:
    """
    Parse an ABSA CSV file with columns:
        id, text, aspect, category, polarity, from, to

    Multiple rows may share the same text (one row per aspect term).
    Groups rows by text, assigns BIO tags for all aspects in each sentence.

    Returns list of dicts identical to parse_semeval_xml output:
        {tokens, bio_tags, pos_tags, dep_tags, text}
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    # Group rows by text (preserve order)
    text_aspects: OrderedDict = OrderedDict()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"]
            if text not in text_aspects:
                text_aspects[text] = []
            text_aspects[text].append(row)

    samples = []
    for text, aspects in text_aspects.items():
        if not text or not text.strip():
            continue

        # ── spaCy tokenize ────────────────────────────────────────────
        doc = nlp(text)
        tokens = [t.text for t in doc]
        pos_tags = [t.tag_ for t in doc]
        dep_tags = [t.dep_ for t in doc]
        spans = _char_spans(doc)

        # ── default: all O ────────────────────────────────────────────
        bio = ["O"] * len(tokens)

        for row in aspects:
            polarity = row.get("polarity", "neutral").lower()
            if polarity == "conflict":
                polarity = "neutral"
            if polarity not in _POL:
                continue
            suffix = _POL[polarity]

            try:
                a_start = int(row["from"])
                a_end   = int(row["to"])
            except (ValueError, KeyError):
                continue

            _assign_bio(bio, spans, a_start, a_end, suffix)

        samples.append(
            dict(tokens=tokens, bio_tags=bio, pos_tags=pos_tags,
                 dep_tags=dep_tags, text=text)
        )

    return samples


# ── XML parser (SemEval-2014) ─────────────────────────────────────────

def parse_semeval_xml(xml_path: str, nlp=None) -> List[Dict]:
    """
    Parse the SemEval-2014 Restaurants XML and produce token-level
    BIO-sentiment tags aligned with spaCy tokenization.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    tree = ET.parse(xml_path)
    root = tree.getroot()
    samples = []

    for sent_el in root.iter("sentence"):
        text = sent_el.find("text").text
        if text is None:
            continue

        # ── spaCy tokenize ────────────────────────────────────────────
        doc = nlp(text)
        tokens = [t.text for t in doc]
        pos_tags = [t.tag_ for t in doc]
        dep_tags = [t.dep_ for t in doc]
        spans = _char_spans(doc)

        # ── default: all O ────────────────────────────────────────────
        bio = ["O"] * len(tokens)

        at_el = sent_el.find("aspectTerms")
        if at_el is not None:
            for a in at_el.findall("aspectTerm"):
                polarity = a.get("polarity", "neutral")
                if polarity == "conflict":
                    polarity = "neutral"
                suffix = _POL[polarity]
                a_start = int(a.get("from"))
                a_end   = int(a.get("to"))
                _assign_bio(bio, spans, a_start, a_end, suffix)

        samples.append(
            dict(tokens=tokens, bio_tags=bio, pos_tags=pos_tags,
                 dep_tags=dep_tags, text=text)
        )

    return samples


# ── Build tag-to-id maps for POS and DEP ──────────────────────────────
def build_tag_vocabs(samples: List[Dict]):
    """Return {pos_tag: id}, {dep_tag: id} covering all tags in *samples*."""
    pos_set, dep_set = set(), set()
    for s in samples:
        pos_set.update(s["pos_tags"])
        dep_set.update(s["dep_tags"])
    pos2id = {t: i for i, t in enumerate(sorted(pos_set))}
    dep2id = {t: i for i, t in enumerate(sorted(dep_set))}
    return pos2id, dep2id


# ── PyTorch Dataset ───────────────────────────────────────────────────
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class ABSADataset(Dataset):
    """
    Each item returns:
      input_ids, attention_mask     – BERT subword tensors  (MAX_SEQ_LEN)
      word_ids_tensor               – maps subword → word index (-1=special)
      bio_ids                       – tag per word          (max_words)
      pos_ids, dep_ids              – POS/DEP per word      (max_words)
      num_words                     – actual word count
      tokens                        – raw token strings
    """

    def __init__(self, samples, pos2id, dep2id, tokenizer=None,
                 max_len=Config.MAX_SEQ_LEN):
        self.samples = samples
        self.pos2id = pos2id
        self.dep2id = dep2id
        self.max_len = max_len
        self.tokenizer = tokenizer or BertTokenizerFast.from_pretrained(
            Config.BERT_MODEL_NAME
        )
        self.tag2id = Config.TAG2IDX

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tokens = s["tokens"]
        bio    = s["bio_tags"]
        pos    = s["pos_tags"]
        dep    = s["dep_tags"]

        # ── BERT tokenisation (word-aligned) ─────────────────────────
        enc = self.tokenizer(
            tokens, is_split_into_words=True,
            max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)        # (max_len,)
        attention_mask  = enc["attention_mask"].squeeze(0)   # (max_len,)

        # word_ids: maps each subword position → original word index (None for special tokens)
        word_ids = enc.word_ids(batch_index=0)

        # Build word_ids tensor (-1 for special / padding tokens)
        word_ids_tensor = torch.full((self.max_len,), -1, dtype=torch.long)
        for i, wid in enumerate(word_ids):
            if wid is not None:
                word_ids_tensor[i] = wid

        # ── Per-word labels (truncated to fit) ───────────────────────
        n_words = min(len(tokens), self.max_len - 2)   # leave room for [CLS],[SEP]
        max_words = self.max_len  # upper bound
        bio_ids = torch.full((max_words,), -100, dtype=torch.long)
        pos_ids = torch.full((max_words,), -100, dtype=torch.long)
        dep_ids = torch.full((max_words,), -100, dtype=torch.long)

        for w in range(n_words):
            bio_ids[w] = self.tag2id.get(bio[w], 0)
            pos_ids[w] = self.pos2id.get(pos[w], 0)
            dep_ids[w] = self.dep2id.get(dep[w], 0)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids_tensor,
            bio_ids=bio_ids,
            pos_ids=pos_ids,
            dep_ids=dep_ids,
            num_words=n_words,
            tokens=tokens[:n_words],
        )


def collate_fn(batch):
    """Stack tensors; keep tokens as list of lists."""
    keys = ["input_ids", "attention_mask", "word_ids", "bio_ids", "pos_ids", "dep_ids"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["num_words"] = torch.tensor([b["num_words"] for b in batch], dtype=torch.long)
    out["tokens"] = [b["tokens"] for b in batch]
    return out


"""
Helpers — seed setting, logging, device detection.
"""
import os, random, logging
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_logger(name: str = "KETGM", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s %(name)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def download_file(url: str, dest: str, desc: str = None):
    """Download *url* to *dest* with a tqdm progress bar."""
    import urllib.request
    from tqdm import tqdm

    if os.path.exists(dest):
        print(f"  ✓ Already exists: {os.path.basename(dest)}")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    desc = desc or os.path.basename(dest)
    print(f"  ↓ Downloading {desc} …")

    class _Progress(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with _Progress(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)

    print(f"  ✓ Saved to {dest}")


"""
TWLDA-style topic extraction.

Approximation of the Term-Weighting LDA (Yang et al. 2016):
  1. Run initial LDA on the corpus.
  2. Compute per-word topic-discriminating power (TF-IDF style).
  3. Re-weight the BoW and run LDA again.
  4. Extract top-k words per topic.
"""
import re, pickle, os
from typing import List, Set
from collections import Counter

import numpy as np
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS

from config import Config


def _preprocess(texts: List[str]) -> List[List[str]]:
    """Lowercase, keep alpha tokens ≥ 3 chars, remove stopwords."""
    docs = []
    for t in texts:
        toks = re.findall(r"[a-z]{3,}", t.lower())
        toks = [w for w in toks if w not in STOPWORDS]
        docs.append(toks)
    return docs


def _tw_weights(corpus, dictionary, lda_model, num_topics):
    """
    Compute term-weighting: for each word, reduce weight if it
    appears uniformly across all topics (low discriminating power).
    We use a simple entropy-based weight:
        w(t) = 1 − H(t) / log(K)
    where H(t) is the entropy of word t across topics.
    """
    V = len(dictionary)
    # topic-word matrix  (K x V)
    tw = np.zeros((num_topics, V))
    for k in range(num_topics):
        for wid, prob in lda_model.get_topic_terms(k, topn=V):
            tw[k, wid] = prob
    # normalise per word
    tw_sum = tw.sum(axis=0, keepdims=True) + 1e-12
    tw_norm = tw / tw_sum
    # entropy per word
    H = -np.sum(tw_norm * np.log(tw_norm + 1e-12), axis=0)
    max_H = np.log(num_topics) + 1e-12
    weights = 1.0 - H / max_H          # high = discriminating
    weights = np.clip(weights, 0.1, 1.0)
    return weights                      # shape (V,)


def extract_topics(texts: List[str],
                   num_topics: int = Config.NUM_TOPICS,
                   words_per_topic: int = Config.WORDS_PER_TOPIC,
                   ) -> List[str]:
    """
    Return a flat list of Ntopic = num_topics × words_per_topic topic words.
    """
    docs = _preprocess(texts)
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=3, no_above=0.7)
    corpus = [dictionary.doc2bow(d) for d in docs]

    # ── Phase 1: initial LDA ──────────────────────────────────────────
    lda1 = models.LdaModel(
        corpus, id2word=dictionary, num_topics=num_topics,
        passes=10, random_state=Config.SEED, alpha="auto", eta="auto",
    )

    # ── Compute term weights ──────────────────────────────────────────
    weights = _tw_weights(corpus, dictionary, lda1, num_topics)

    # ── Phase 2: re-weighted corpus ───────────────────────────────────
    weighted_corpus = []
    for doc in corpus:
        weighted_corpus.append(
            [(wid, cnt * weights[wid]) for wid, cnt in doc]
        )

    lda2 = models.LdaModel(
        weighted_corpus, id2word=dictionary, num_topics=num_topics,
        passes=15, random_state=Config.SEED, alpha="auto", eta="auto",
    )

    # ── Extract topic words ───────────────────────────────────────────
    seen = set()
    topic_words: List[str] = []
    for k in range(num_topics):
        for word, _ in lda2.show_topic(k, topn=words_per_topic + 20):
            if word not in seen and len(word) > 2:
                topic_words.append(word)
                seen.add(word)
            if len(topic_words) >= (k + 1) * words_per_topic:
                break

    # pad if some topics overlapped
    topic_words = topic_words[: num_topics * words_per_topic]
    return topic_words
