"""
Data Loading — Step 1 of the KETGM pipeline.

Parses ABSA CSV data (restaurant.csv, laptop.csv, etc.),
splits source domain into train/eval, loads target domain unlabeled,
and builds POS/DEP tag vocabularies.
"""
import random
import time
import spacy

from config import Config
from utils import parse_absa_csv, build_tag_vocabs


def run(config=None, nlp=None):
    """
    Load source and target domain data from CSVs.

    Uses config.SOURCE_DOMAIN and config.TARGET_DOMAIN to select CSVs.
    Source domain is split into train/eval (labeled).
    Target domain is loaded as unlabeled for topic extraction + KG.

    Returns
    -------
    dict with keys:
        samples, train_samples, eval_samples,
        target_samples, pos2id, dep2id, nlp
    """
    if config is None:
        config = Config

    if nlp is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])

    src = config.SOURCE_DOMAIN
    tgt = config.TARGET_DOMAIN
    print(f'Source domain: {src}')
    print(f'Target domain: {tgt}')

    # ── Parse source domain CSV ───────────────────────────────────────
    src_csv = config.DOMAIN_CSV[src]
    t0 = time.time()
    samples = parse_absa_csv(src_csv, nlp=nlp)
    print(f'Parsed {len(samples)} source sentences from {src}.csv  ({time.time()-t0:.1f}s)')

    # ── Parse target domain CSV (unlabeled — used for topics + KG) ────
    tgt_csv = config.DOMAIN_CSV[tgt]
    t0 = time.time()
    target_samples = parse_absa_csv(tgt_csv, nlp=nlp)
    print(f'Parsed {len(target_samples)} target sentences from {tgt}.csv  ({time.time()-t0:.1f}s)')

    # ── Train / eval split on source domain ───────────────────────────
    random.seed(config.SEED)
    idx = list(range(len(samples)))
    random.shuffle(idx)
    split = int(len(samples) * config.TRAIN_RATIO)
    train_samples = [samples[i] for i in idx[:split]]
    eval_samples  = [samples[i] for i in idx[split:]]
    print(f'Train: {len(train_samples)}   Eval: {len(eval_samples)}')

    # ── POS / DEP vocabs (built from ALL data for full coverage) ──────
    all_samples = samples + target_samples
    pos2id, dep2id = build_tag_vocabs(all_samples)
    print(f'POS tags: {len(pos2id)}   DEP tags: {len(dep2id)}')

    # ── Quick verification ────────────────────────────────────────────
    s = next((s for s in train_samples if any(t != 'O' for t in s['bio_tags'])), None)
    if s:
        print(f"\nVerification — \"{s['text'][:80]}...\"")
        for tok, tag in list(zip(s['tokens'], s['bio_tags']))[:15]:
            print(f"  {tok:20s} {'→ '+tag if tag != 'O' else '  O'}")

    return dict(
        samples=samples,
        train_samples=train_samples,
        eval_samples=eval_samples,
        target_samples=target_samples,
        pos2id=pos2id,
        dep2id=dep2id,
        nlp=nlp,
    )
