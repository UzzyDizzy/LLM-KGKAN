# kg_utils.py

import csv
from collections import defaultdict

class ConceptNet:
    def __init__(self, path, max_edges=2_000_000):
        self.graph = defaultdict(list)
        self.rel2id = {}
        self.ent2id = {}
        self.id2rel = {}
        self.id2ent = {}

        self._load_csv(path, max_edges)

    def _normalize(self, term):
        return term.lower().replace("_", " ")

    def _get_ent_id(self, ent):
        if ent not in self.ent2id:
            idx = len(self.ent2id)
            self.ent2id[ent] = idx
            self.id2ent[idx] = ent
        return self.ent2id[ent]

    def _get_rel_id(self, rel):
        if rel not in self.rel2id:
            idx = len(self.rel2id)
            self.rel2id[rel] = idx
            self.id2rel[idx] = rel
        return self.rel2id[rel]

    def _parse_uri(self, uri):
        # /c/en/apple → apple
        parts = uri.split("/")
        if len(parts) >= 4:
            return self._normalize(parts[3])
        return None

    def _load_csv(self, path, max_edges):
        print("Loading ConceptNet...")

        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")

            for i, row in enumerate(reader):
                if i > max_edges:
                    break

                _, rel, head, tail, *_ = row

                if not (head.startswith("/c/en/") and tail.startswith("/c/en/")):
                    continue

                h = self._parse_uri(head)
                t = self._parse_uri(tail)
                r = rel.split("/")[-1]

                if h is None or t is None:
                    continue

                h_id = self._get_ent_id(h)
                t_id = self._get_ent_id(t)
                r_id = self._get_rel_id(r)

                self.graph[h].append((h_id, r_id, t_id))

        print(f"Loaded {len(self.ent2id)} entities, {len(self.rel2id)} relations")

    def get_triples_for_tokens(self, tokens, max_triples=16, hops=2):
        visited = set()
        triples = []

        queue = list(tokens)

        for _ in range(hops):
            new_queue = []

            for tok in queue:
                tok = tok.lower()
                if tok not in self.graph:
                    continue

                for (h_id, r_id, t_id) in self.graph[tok]:
                    key = (h_id, r_id, t_id)
                    if key in visited:
                        continue

                    visited.add(key)
                    triples.append(key)

                    new_queue.append(self.id2ent[t_id])

                    if len(triples) >= max_triples:
                        break

                if len(triples) >= max_triples:
                    break

            queue = new_queue

        if len(triples) == 0:
            return [], [], []

        h, r, t = zip(*triples[:max_triples])
        return list(h), list(r), list(t)

    def build_token_kg_map(tokens, triples, ent2id):
        t = len(tokens)
        k = len(triples)

        mapping = [[0]*k for _ in range(t)]

        for i, tok in enumerate(tokens):
            tok = tok.lower()

            for j, (h, r, ta) in enumerate(triples):
                if tok == ent2id.get(h, None):
                    mapping[i][j] = 1

        return mapping