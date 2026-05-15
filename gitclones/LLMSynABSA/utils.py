# utils.py — Utilities with proper word-to-token alignment for syntax matrix
import torch
import spacy
from config import MAX_LEN, ID_MAP, device

nlp = spacy.load("en_core_web_sm")


def extract_pairs(labels, offsets, text):
    """Extract (aspect_text, sentiment) pairs from BIO label sequence."""
    pairs = []
    i = 0
    while i < len(labels):
        if labels[i] == "O" or labels[i] == "PAD":
            i += 1
            continue
        if labels[i].startswith("B"):
            sent = labels[i].split("-")[1]
            start = i
            i += 1
            while (i < len(labels) and labels[i].startswith("I") and labels[i].split("-")[1] == sent):
                i += 1
            end = i
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            if char_end > char_start:
                aspect = text[char_start:char_end]
                pairs.append((aspect.strip(), sent))
        else:
            i += 1
    return set(pairs)


def extract_spans_from_preds(batch_labels, id_map):
    """Extract span indices from predicted label sequences (for aspect/sentiment classifiers)."""
    all_spans = []
    for labels in batch_labels:
        spans = []
        i = 0
        while i < len(labels):
            if isinstance(labels[i], int):
                if labels[i] == -100:
                    i += 1
                    continue
                lab = id_map[labels[i]]
            else:
                lab = labels[i]

            if lab.startswith("B") or lab in ["POS", "NEG", "NEU"]:
                start = i
                i += 1
                while i < len(labels):
                    if isinstance(labels[i], int):
                        if labels[i] == -100:
                            break
                        next_lab = id_map[labels[i]]
                    else:
                        next_lab = labels[i]
                    if lab.startswith("B"):
                        if not next_lab.startswith("I"):
                            break
                    else:
                        if next_lab != lab:
                            break
                    i += 1
                spans.append((start, i))
            else:
                i += 1
        all_spans.append(spans)
    return all_spans


def build_representation_batch(H, spans_batch):
    """
    Build mean-pooled representations from spans (Eq 4, 5).
    Returns: (batch, d) tensor — one representative vector per sample.
    """
    reps = []
    for i, spans in enumerate(spans_batch):
        h = H[i]  # (seq, d)
        cur = []
        for s, e in spans:
            if s < h.size(0) and e <= h.size(0):
                cur.append(h[s:e].mean(dim=0))
        if len(cur) == 0:
            cur_vec = torch.zeros(H.size(-1), device=H.device, dtype=H.dtype)
        else:
            cur_vec = torch.stack(cur).mean(dim=0)
        reps.append(cur_vec)
    return torch.stack(reps)  # (batch, d)


def get_relative_positions(seq_len):
    """Generate relative position indices for attention (Eq 8)."""
    pos = torch.arange(seq_len).unsqueeze(1)
    rel = pos - pos.T + seq_len  # shift to positive range
    return rel.long().to(device)


def _word_to_token_map(text, tokenizer):
    """
    Map spaCy word indices to LLaMA token indices.
    Returns dict: spacy_word_idx -> list of token indices.
    """
    doc = nlp(text)
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    offsets = enc["offset_mapping"]

    word_to_tokens = {}
    for word in doc:
        word_start = word.idx
        word_end = word.idx + len(word.text)
        token_indices = []
        for tok_idx, (ts, te) in enumerate(offsets):
            if ts == te == 0:
                continue  # special token
            if te <= word_start:
                continue
            if ts >= word_end:
                break
            token_indices.append(tok_idx)
        if token_indices:
            word_to_tokens[word.i] = token_indices

    return word_to_tokens, doc


def syntax_matrix(text, tokenizer):
    """
    Build syntax-aware dependency matrix (Eq 6) with proper word-to-token alignment.
    0 = visible, -1e9 = masked.
    """
    M = torch.full((MAX_LEN, MAX_LEN), -1e4)

    word_to_tokens, doc = _word_to_token_map(text, tokenizer)

    # Self-visibility for all real tokens
    for word_idx, tok_indices in word_to_tokens.items():
        for ti in tok_indices:
            M[ti][ti] = 0.0

    # Rule 2: Dependency edges — child can see parent and vice versa
    for token in doc:
        parent_toks = word_to_tokens.get(token.head.i, [])
        child_toks = word_to_tokens.get(token.i, [])
        for pt in parent_toks:
            for ct in child_toks:
                if pt < MAX_LEN and ct < MAX_LEN:
                    M[pt][ct] = 0.0
                    M[ct][pt] = 0.0

    # Rule 1 (intra-span visibility) is applied dynamically in model.py
    # using detected aspect/sentiment spans

    return M


def syntax_caching(train_data, test_data, tokenizer):
    """Pre-compute syntax matrices for all texts."""
    syntax_cache = {}
    all_texts = set()
    for t, _ in train_data:
        all_texts.add(t)
    for t, _ in test_data:
        all_texts.add(t)

    print(f"Building syntax cache for {len(all_texts)} texts...")
    for i, text in enumerate(all_texts):
        syntax_cache[text] = syntax_matrix(text, tokenizer).cpu()
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(all_texts)}")
    print("Syntax cache built.")
    return syntax_cache