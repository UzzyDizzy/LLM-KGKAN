# data.py

import os
import pickle

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from gensim.models import Word2Vec

from config import Config
from preprocess import char_to_token_indices, normalize_polarity

cfg = Config()


# --------------------------------------------------
# build sentence-level grouped rows
# --------------------------------------------------

def build_sentence_rows(csv_path):

    cache_file = f"./cache/{os.path.basename(csv_path)}.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(csv_path)

    grouped = defaultdict(list)

    for _, r in df.iterrows():
        txt = str(r["text"])
        grouped[(txt, r["split"])].append(r)

    rows = []

    for (text, split), items in grouped.items():

        toks = None
        poss = None
        aspect_spans = []

        for r in items:

            start = int(r["from"])
            end   = int(r["to"])

            toks, poss, inds = char_to_token_indices(
                text, start, end
            )

            pol = normalize_polarity(r["polarity"])

            if len(inds) > 0:
                aspect_spans.append((inds, pol))

        labels = make_bio_labels(len(toks), aspect_spans)

        rows.append({
            "text": text,
            "tokens": toks,
            "pos": poss,
            "labels": labels,
            "split": split
        })

    with open(cache_file, "wb") as f:
        pickle.dump(rows, f)

    return rows


# --------------------------------------------------
# BIO labels
# --------------------------------------------------

def make_bio_labels(n, spans):

    y = ["O"] * n

    for inds, pol in spans:

        if len(inds) == 0:
            continue

        y[inds[0]] = f"B-{pol}"

        for j in inds[1:]:
            y[j] = f"I-{pol}"

    return y


# --------------------------------------------------
# official split
# --------------------------------------------------

def load_domain(domain_name):

    path = f"{cfg.data_dir}/{cfg.datasets[domain_name]}"
    rows = build_sentence_rows(path)

    train = [r for r in rows if r["split"] == "train"]
    test  = [r for r in rows if r["split"] == "test"]

    return train, test


# --------------------------------------------------
# choose pretrained embedding source
# --------------------------------------------------

_EMBED_MODEL = None

def get_embedding_model():
    global _EMBED_MODEL

    if _EMBED_MODEL is None:
        print("Loading Word2Vec once...")
        _EMBED_MODEL = Word2Vec.load(
            "./embeddings/raw_corpus_w2v.model"
        )

    return _EMBED_MODEL

# --------------------------------------------------
# vocab build
# --------------------------------------------------

def build_vocab(src_train, src_test, tgt_train, tgt_test):

    wc = Counter()
    pc = Counter()

    all_rows = src_train + src_test + tgt_train + tgt_test

    for row in all_rows:
        for t in row["tokens"]:
            wc[t] += 1
        for p in row["pos"]:
            pc[p] += 1

    word2id = {"<PAD>":0, "<UNK>":1}
    pos2id  = {"<PAD>":0, "<UNK>":1}

    for w,_ in wc.items():
        word2id[w] = len(word2id)

    for p,_ in pc.items():
        pos2id[p] = len(pos2id)

    return word2id, pos2id


# --------------------------------------------------
# embedding matrix
# --------------------------------------------------

def build_embedding_matrix(word2id, src_domain, tgt_domain):

    dim = cfg.word_dim

    mat = np.random.uniform(
        -0.05, 0.05,
        (len(word2id), dim)
    ).astype(np.float32)

    model = get_embedding_model()

    for w, idx in word2id.items():

        if w in ["<PAD>", "<UNK>"]:
            continue

        if w in model.wv:
            mat[idx] = model.wv[w]

    mat[0] = 0.0

    return mat


# --------------------------------------------------
# encode rows
# --------------------------------------------------

def encode_rows(rows, word2id, pos2id, labeled=True):

    out = []

    for row in rows:

        x = [
            word2id.get(t,1)
            for t in row["tokens"]
        ]

        p = [
            pos2id.get(tag,1)
            for tag in row["pos"]
        ]

        if labeled:
            y = [
                cfg.label2id[z]
                for z in row["labels"]
            ]
        else:
            y = None

        out.append({
            "tokens": row["tokens"],
            "x": x,
            "p": p,
            "y": y
        })

    return out


# --------------------------------------------------
# final transfer loader
# --------------------------------------------------

def load_transfer_pair(src_domain, tgt_domain):

    # source
    src_train, src_test = load_domain(src_domain)

    # target
    tgt_train, tgt_test = load_domain(tgt_domain)

    # official protocol:
    # source train labeled
    # source test validation
    # target train unlabeled
    # target test labeled evaluation

    word2id, pos2id = build_vocab(
        src_train, src_test,
        tgt_train, tgt_test
    )

    emb = build_embedding_matrix(
        word2id,
        src_domain,
        tgt_domain
    )

    src_train = encode_rows(
        src_train, word2id, pos2id, labeled=True
    )

    src_val = encode_rows(
        src_test, word2id, pos2id, labeled=True
    )

    tgt_train = encode_rows(
        tgt_train, word2id, pos2id, labeled=False
    )

    tgt_test = encode_rows(
        tgt_test, word2id, pos2id, labeled=True
    )

    return (
        src_train,
        src_val,
        tgt_train,
        tgt_test,
        word2id,
        pos2id,
        emb
    )
