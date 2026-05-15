#preprocess.py

import stanza
import re
from functools import lru_cache

# Stanford official pipeline
nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos",
    tokenize_no_ssplit=True,
    use_gpu=False
)

@lru_cache(maxsize=50000)
def stanford_annotate(text):

    doc = nlp(text)

    tokens = []
    pos_tags = []
    spans = []

    for sent in doc.sentences:
        for w in sent.words:
            tok = w.text
            tokens.append(tok.lower())
            pos_tags.append(w.xpos if w.xpos else w.upos)

            start = w.start_char
            end   = w.end_char
            spans.append((start,end))

    return tokens, pos_tags, spans


def char_to_token_indices(text, start, end):
    """
    exact char span -> token ids
    """
    toks, pos, spans = stanford_annotate(text)

    ids = []

    for i,(s,e) in enumerate(spans):
        if not (e <= start or s >= end):
            ids.append(i)

    return toks, pos, ids


def normalize_polarity(p):
    p = str(p).lower()

    if "pos" in p:
        return "POS"
    elif "neg" in p:
        return "NEG"
    else:
        return "NEU"