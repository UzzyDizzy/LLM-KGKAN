"""
Knowledge Graph Construction — Step 3 of the KETGM pipeline.

Loads ConceptNet, collects seed tokens (nouns/adj/adv),
and builds the domain-topic aware knowledge graph.
Paper Section III-B1.
"""
import gc
import time

from config import Config
from conceptnet_loader import load_conceptnet_en, build_knowledge_graph


def run(samples, target_samples, topic_words, nlp, config=None):
    """
    Build the domain-topic aware knowledge graph from ConceptNet.

    Parameters
    ----------
    samples         : list[dict] — source domain samples (for seed token extraction)
    target_samples  : list[dict] — target domain samples (for seed token extraction)
    topic_words     : list[str]  — extracted topic words
    nlp             : spacy model

    Returns
    -------
    dict with keys: kg, seed_tokens
    """
    if config is None:
        config = Config

    # Load ConceptNet English adjacency list
    t0 = time.time()
    cn_adj = load_conceptnet_en(config.CONCEPTNET_CSV_GZ, config.CONCEPTNET_EN_PKL)
    print(f'English ConceptNet loaded  ({time.time()-t0:.1f}s)')

    # Collect seed tokens (nouns / adj / adv) from both domains
    seed_tokens = set()
    for s in samples + target_samples:
        text = s.get('text', ' '.join(s.get('tokens', [])))
        doc = nlp(text)
        for tok in doc:
            if tok.pos_ in ('NOUN', 'ADJ', 'ADV') and len(tok.text) > 2:
                seed_tokens.add(tok.text.lower())
    seed_tokens = list(seed_tokens)
    print(f'Seed tokens: {len(seed_tokens)}')

    # Build knowledge graph
    t0 = time.time()
    kg = build_knowledge_graph(cn_adj, seed_tokens, topic_words,
                               max_hops=config.MAX_HOPS)
    print(f'KG built in {time.time()-t0:.1f}s')
    print(f"  Nodes : {len(kg['node2id']):,}")
    print(f"  Edges : {len(kg['edges']):,}")
    print(f"  Rels  : {len(kg['rel2id'])}")
    print(f"  Token→node mappings : {len(kg['token2node']):,}")
    print(f"  Topic nodes in KG   : {len(kg['topic_node_ids'])}")

    # Free ConceptNet adjacency (large) — no longer needed
    del cn_adj
    gc.collect()

    return dict(kg=kg, seed_tokens=seed_tokens)
