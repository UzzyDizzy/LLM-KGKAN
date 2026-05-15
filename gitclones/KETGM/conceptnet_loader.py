"""
ConceptNet 5.7.0 CSV loader  →  domain-topic-aware knowledge graph.

Pipeline
--------
1. Stream the gzipped CSV, keep only English edges.
2. Build an in-memory adjacency list  concept → [(neighbour, relation)].
3. For every seed token (noun / adj / adv in the reviews) find ≤ 2-hop
   relational paths to domain topic words.
4. Merge all paths into graph  G = (V, E, R)  usable by R-GCN.
"""
import gzip, os, pickle, re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from tqdm import tqdm
from config import Config


# ── 1. Load / cache English-only adjacency list ──────────────────────

def _concept_text(uri: str) -> str:
    """'/c/en/good_food' → 'good_food'"""
    parts = uri.split("/")
    if len(parts) >= 4:
        return parts[3].lower()
    return uri.lower()


def _is_english(uri: str) -> bool:
    return uri.startswith("/c/en/")


def _relation_name(uri: str) -> str:
    """'/r/IsA' → 'IsA'"""
    return uri.split("/")[-1]


def load_conceptnet_en(csv_gz_path: str = Config.CONCEPTNET_CSV_GZ,
                       cache_path: str = Config.CONCEPTNET_EN_PKL,
                       ) -> Dict[str, List[Tuple[str, str]]]:
    """
    Return adjacency list  concept_text → [(neighbour_text, relation_name)].
    Builds from the CSV the first time; pickles the result.
    """
    if os.path.exists(cache_path):
        print("  ✓ Loading cached English ConceptNet …")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("  ⏳ Streaming ConceptNet CSV (English only) …")
    adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    count = 0
    if csv_gz_path.endswith('.gz'):
        f_in = gzip.open(csv_gz_path, "rt", encoding="utf-8")
    else:
        f_in = open(csv_gz_path, "r", encoding="utf-8")
        
    with f_in as f:
        for line in tqdm(f, desc="ConceptNet", unit=" edges"):
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            rel_uri, src_uri, tgt_uri = parts[1], parts[2], parts[3]
            if not (_is_english(src_uri) and _is_english(tgt_uri)):
                continue
            src = _concept_text(src_uri)
            tgt = _concept_text(tgt_uri)
            rel = _relation_name(rel_uri)
            adj[src].append((tgt, rel))
            adj[tgt].append((src, rel))   # undirected for path search
            count += 1

    adj = dict(adj)   # drop defaultdict wrapper
    print(f"  ✓ Kept {count:,} English edges, {len(adj):,} concepts")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return adj


# ── 2. Domain-topic-aware knowledge graph construction ────────────────

MAX_NEIGHBORS = 50   # cap neighbours explored per node for efficiency


def _find_paths_bfs(adj, seed: str, topics: Set[str],
                    max_hops: int = Config.MAX_HOPS
                    ) -> List[List[Tuple[str, str, str]]]:
    """
    BFS from *seed* up to *max_hops*.
    Returns list of paths, each path = [(src, rel, tgt), …]
    that end at a topic word.
    """
    # queue items: (current_node, path_so_far)
    queue = [(seed, [])]
    visited = {seed}
    found_paths = []

    for _hop in range(max_hops):
        next_queue = []
        for node, path in queue:
            neighbours = adj.get(node, [])[:MAX_NEIGHBORS]
            for neighbour, rel in neighbours:
                if neighbour in visited:
                    continue
                new_path = path + [(node, rel, neighbour)]
                if neighbour in topics:
                    found_paths.append(new_path)
                # keep expanding even if found, up to max_hops
                next_queue.append((neighbour, new_path))
                visited.add(neighbour)
        queue = next_queue
        if not queue:
            break

    return found_paths


def build_knowledge_graph(
    adj: Dict[str, List[Tuple[str, str]]],
    seed_tokens: List[str],
    topic_words: List[str],
    max_hops: int = Config.MAX_HOPS,
) -> Dict:
    """
    Build the domain-topic-aware knowledge graph G = (V, E, R).

    Returns dict with:
      node2id    – {concept_str: int}
      rel2id     – {relation_str: int}
      edges      – list of (src_id, rel_id, tgt_id)
      token2node – {token_str: node_id}  (for tokens present in G)
      topic_node_ids – list of node_ids for topic words
    """
    topics_set = set(topic_words)
    nodes: Dict[str, int] = {}
    rels:  Dict[str, int] = {}
    edges_set: Set[Tuple[int, int, int]] = set()

    def _nid(c):
        if c not in nodes:
            nodes[c] = len(nodes)
        return nodes[c]

    def _rid(r):
        if r not in rels:
            rels[r] = len(rels)
        return rels[r]

    token2node: Dict[str, int] = {}
    found_count = 0

    clean_seeds = list(set(t.lower().replace(" ", "_") for t in seed_tokens))

    for seed in tqdm(clean_seeds, desc="KG paths", unit=" tokens"):
        paths = _find_paths_bfs(adj, seed, topics_set, max_hops)
        if paths:
            token2node[seed] = _nid(seed)
            found_count += 1
            for path in paths:
                for src, rel, tgt in path:
                    sid, tid, rid = _nid(src), _nid(tgt), _rid(rel)
                    edges_set.add((sid, rid, tid))

    # make sure all topic words that appear as nodes are recorded
    topic_node_ids = [nodes[tw] for tw in topic_words if tw in nodes]

    edges = list(edges_set)
    print(f"  ✓ KG: {len(nodes):,} nodes, {len(edges):,} edges, "
          f"{len(rels)} relation types")
    print(f"  ✓ {found_count}/{len(clean_seeds)} seed tokens linked to topics")

    return dict(
        node2id=nodes, rel2id=rels, edges=edges,
        token2node=token2node, topic_node_ids=topic_node_ids,
    )
