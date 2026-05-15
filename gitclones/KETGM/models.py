"""
BERT Syntactic Knowledge Embedding — Section III-D of the KETGM paper.

Following Gong et al. 2020 (BERT-UDA), the BERT encoder is fine-tuned
with two self-supervised auxiliary tasks:
  1. POS-tag classification   (per token)
  2. Dependency-relation classification  (per token)

After pre-training the encoder produces syntactic-aware features  ts ∈ R^768.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel

from config import Config


class BERTSyntactic(nn.Module):
    """
    BERT encoder   +  POS classification head  +  DEP classification head
    """

    def __init__(self, num_pos_tags: int, num_dep_tags: int,
                 model_name: str = Config.BERT_MODEL_NAME,
                 hidden: int = Config.BERT_HIDDEN,
                 dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.pos_head = nn.Linear(hidden, num_pos_tags)
        self.dep_head = nn.Linear(hidden, num_dep_tags)

    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, word_ids=None):
        """
        Returns
        -------
        hidden       : (B, seq_len, 768)  full subword hidden states
        pos_logits   : (B, seq_len, num_pos)
        dep_logits   : (B, seq_len, num_dep)
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)        # (B, S, 768)
        return h, self.pos_head(h), self.dep_head(h)

    # ------------------------------------------------------------------
    def extract_word_features(self, input_ids, attention_mask, word_ids,
                              num_words):
        """
        Run BERT and gather **first-subword** hidden states as per-word
        syntactic features  ts.

        Parameters
        ----------
        word_ids   : (B, S) – for each subword, which word it belongs to
                      (-1 for [CLS], [SEP], [PAD])
        num_words  : (B,)

        Returns
        -------
        ts : (B, max_words, 768)
        """
        h, _, _ = self.forward(input_ids, attention_mask)  # (B, S, 768)
        B, S, D = h.shape
        max_w = num_words.max().item()
        ts = torch.zeros(B, max_w, D, device=h.device)

        for b in range(B):
            seen = set()
            for s in range(S):
                wid = word_ids[b, s].item()
                if wid < 0 or wid >= max_w or wid in seen:
                    continue
                ts[b, wid] = h[b, s]
                seen.add(wid)
        return ts


# ══════════════════════════════════════════════════════════════════════
# Topic-Guided Knowledge Attention — Section III-C (Eqs 6-7)
# ══════════════════════════════════════════════════════════════════════

class TopicGuidedAttention(nn.Module):
    """
    Input
    -----
    token_embeds  : (B, W, d)   mapped R-GCN embeddings  tm_c  per word
    topic_embeds  : (T, d)      topic-word embeddings from R-GCN

    Output
    ------
    trs           : (B, W, d)   relational structure representation
    """

    def __init__(self, hidden_dim: int = Config.RGCN_HIDDEN):
        super().__init__()
        self.scale = hidden_dim ** 0.5

    def forward(self, token_embeds, topic_embeds):
        """
        Eq 6:  w_{tp_i} = softmax( tm_xi · tp_i )
        Eq 7:  trs = Σ_i  w_{tp_i} · tp_i
        """
        # token_embeds : (B, W, d)
        # topic_embeds : (T, d)
        T = topic_embeds.size(0)
        if T == 0:
            # No topics found in KG — return zeros
            return torch.zeros_like(token_embeds)

        #   → attn scores : (B, W, T)
        scores = torch.matmul(token_embeds, topic_embeds.T) / self.scale
        weights = F.softmax(scores, dim=-1)           # (B, W, T)

        # weighted sum of topic embeddings
        trs = torch.matmul(weights, topic_embeds)     # (B, W, d)
        return trs


# ══════════════════════════════════════════════════════════════════════
# R-GCN Autoencoder  (Encoder + DistMult decoder)
#
# Implements Equations 1-5 from the KETGM paper:
#   Eq 1: R-GCN message passing
#   Eq 2: DistMult scoring
#   Eq 3: Link-prediction BCE loss
#   Eq 4: Feature mapping  tc → tm_c → trecon_c
#   Eq 5: Reconstruction loss  (cosine similarity)
# ══════════════════════════════════════════════════════════════════════

class RGCNLayer(nn.Module):
    """
    Relational Graph Convolution with basis decomposition.
    W_r = Σ_b  a_{r,b} · V_b
    """

    def __init__(self, in_dim, out_dim, num_rels, num_bases=None, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        num_bases = min(num_bases or num_rels, num_rels)
        self.num_bases = num_bases

        # basis matrices  V_b : (num_bases, in_dim, out_dim)
        self.bases = nn.Parameter(torch.Tensor(num_bases, in_dim, out_dim))
        # per-relation coefficients  a_{r,b} : (num_rels, num_bases)
        self.coeffs = nn.Parameter(torch.Tensor(num_rels, num_bases))
        # self-loop
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coeffs)

    def forward(self, x, edge_index, edge_type):
        """
        x          : (N, in_dim)
        edge_index : (2, E)  – [src ; tgt]
        edge_type  : (E,)
        """
        N = x.size(0)
        src, tgt = edge_index  # both (E,)

        # W_r = coeffs @ bases  →  (num_rels, in_dim, out_dim)
        W = torch.einsum("rb, bio -> rio", self.coeffs, self.bases)

        out = torch.zeros(N, self.out_dim, device=x.device)

        for r in range(self.num_rels):
            mask = (edge_type == r)
            if not mask.any():
                continue
            src_r = src[mask]
            tgt_r = tgt[mask]
            msg = x[src_r] @ W[r]  # (E_r, out_dim)

            # degree normalisation per relation
            deg = torch.zeros(N, device=x.device)
            deg.scatter_add_(0, tgt_r, torch.ones_like(tgt_r, dtype=torch.float))
            norm = (1.0 / deg.clamp(min=1.0))[tgt_r].unsqueeze(1)
            msg = msg * norm

            out.scatter_add_(0, tgt_r.unsqueeze(1).expand_as(msg), msg)

        out = out + self.W_self(x) + self.bias
        out = self.dropout(out)
        return out


class DistMultDecoder(nn.Module):
    """Score s(h, r, t) = σ( h^T · diag(R_r) · t )  (Eq 2)."""

    def __init__(self, embed_dim, num_rels):
        super().__init__()
        self.rel_embed = nn.Embedding(num_rels, embed_dim)
        nn.init.xavier_uniform_(self.rel_embed.weight)

    def forward(self, h, r, t):
        """h,t: (B, d)  r: (B,)  → scores: (B,)"""
        r_emb = self.rel_embed(r)          # (B, d)
        return torch.sigmoid((h * r_emb * t).sum(dim=1))


class RGCNAutoencoder(nn.Module):
    """
    • Encoder : multi-layer R-GCN  →  node embeddings  tc
    • Decoder : DistMult for link prediction
    • Feature mapping + reconstruction (Eqs 4-5)
    """

    def __init__(self, num_nodes, num_rels,
                 hidden_dim=Config.RGCN_HIDDEN,
                 num_layers=Config.RGCN_NUM_LAYERS,
                 num_bases=Config.RGCN_NUM_BASES,
                 dropout=0.2):
        super().__init__()
        self.node_embed = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_embed.weight)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                RGCNLayer(hidden_dim, hidden_dim, num_rels,
                          num_bases=num_bases, dropout=dropout)
            )

        self.decoder = DistMultDecoder(hidden_dim, num_rels)

        # Eq 4: feature mapping  &  reconstruction
        self.feat_map   = nn.Linear(hidden_dim, hidden_dim)
        self.feat_recon = nn.Linear(hidden_dim, hidden_dim)

    # ── encode ────────────────────────────────────────────────────────
    def encode(self, edge_index, edge_type):
        """Return tc for every node — shape (N, hidden_dim)."""
        x = self.node_embed.weight
        for layer in self.layers:
            x = F.relu(layer(x, edge_index, edge_type))
        return x  # tc

    # ── forward (training) ────────────────────────────────────────────
    def forward(self, edge_index, edge_type, pos_triplets, neg_triplets):
        """
        pos_triplets / neg_triplets : (B, 3)  columns = [src, rel, tgt]
        Returns: link_loss, recon_loss, tc, tm_c
        """
        tc = self.encode(edge_index, edge_type)   # (N, d)

        # ── link prediction loss (Eq 3) ──────────────────────────────
        def _score(triplets):
            h = tc[triplets[:, 0]]
            r = triplets[:, 1]
            t = tc[triplets[:, 2]]
            return self.decoder(h, r, t)

        pos_scores = _score(pos_triplets)
        neg_scores = _score(neg_triplets)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(pos_scores.size(0), device=tc.device),
            torch.zeros(neg_scores.size(0), device=tc.device),
        ])
        link_loss = F.binary_cross_entropy(scores, labels)

        # ── feature mapping + reconstruction (Eqs 4-5) ──────────────
        tm_c = self.feat_map(tc)                       # Eq 4a
        trecon_c = self.feat_recon(tm_c)               # Eq 4b
        # reconstruction loss = 1 − cosine_similarity (minimise)
        cos_sim = F.cosine_similarity(tc, trecon_c, dim=1).mean()
        recon_loss = 1.0 - cos_sim                     # Eq 5

        return link_loss, recon_loss, tc, tm_c


def sample_negatives(pos_triplets, num_nodes, ratio=Config.NEG_RATIO):
    """
    Corrupt head **or** tail of each positive triplet.
    Returns tensor of shape (len(pos)*ratio, 3).
    """
    negs = []
    for _ in range(ratio):
        corrupted = pos_triplets.clone()
        mask = torch.rand(len(pos_triplets)) < 0.5
        corrupted[mask, 0]  = torch.randint(0, num_nodes, (mask.sum(),), device=pos_triplets.device)
        corrupted[~mask, 2] = torch.randint(0, num_nodes, ((~mask).sum(),), device=pos_triplets.device)
        negs.append(corrupted)
    return torch.cat(negs, dim=0)


# ══════════════════════════════════════════════════════════════════════
# KETGM — Full Model (Eq 8-10)
#
# Paper-faithful: feat_map and feat_recon are TRAINABLE during KETGM
# training so that the reconstruction loss (Eq 5) provides meaningful
# gradients.  This matches Fig 5 which shows μ affects performance.
#
# vt = [ tm_c ;  trs ;  ts ]        (Eq 8)
#        200     200    768
# p_i = softmax( W · vt + b )       (Eq 9)
# L = L_softmax  +  μ · L_recon     (Eq 10)
# ══════════════════════════════════════════════════════════════════════

class KETGM(nn.Module):
    """
    End-to-end KETGM sequence labelling model.

    At training time, the caller supplies raw R-GCN node embeddings (tc)
    per token and topic tc embeddings.  The model applies trainable
    feat_map / feat_recon internally and computes reconstruction loss.
    """

    def __init__(self, num_pos_tags, num_dep_tags,
                 rgcn_dim=Config.RGCN_HIDDEN,
                 bert_dim=Config.BERT_HIDDEN,
                 num_tags=Config.NUM_TAGS,
                 mu=Config.MU,
                 dropout=Config.CLASSIFIER_DROPOUT):
        super().__init__()
        self.mu = mu
        self.rgcn_dim = rgcn_dim

        # ── syntactic encoder ─────────────────────────────────────────
        self.bert_enc = BERTSyntactic(num_pos_tags, num_dep_tags)

        # ── knowledge feature mapping (Eq 4) — trainable ─────────────
        self.feat_map   = nn.Linear(rgcn_dim, rgcn_dim)
        self.feat_recon = nn.Linear(rgcn_dim, rgcn_dim)

        # ── topic-guided attention (Eq 6-7) ───────────────────────────
        self.topic_attn = TopicGuidedAttention(hidden_dim=rgcn_dim)

        # ── classifier  (Eq 9) ────────────────────────────────────────
        concat_dim = rgcn_dim + rgcn_dim + bert_dim   # 200+200+768 = 1168
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(concat_dim, num_tags)

    # ─────────────────────────────────────────────────────────────────
    def forward(self, input_ids, attention_mask, word_ids, num_words,
                node_embeds_tc, topic_tc,
                bio_ids=None):
        """
        Parameters
        ----------
        node_embeds_tc : (B, W, rgcn_dim)   raw R-GCN embeddings tc per word (frozen input)
        topic_tc       : (T, rgcn_dim)      raw R-GCN embeddings for topic nodes (frozen input)
        bio_ids        : (B, W)  ground-truth tags  (-100 = ignore)

        Returns
        -------
        logits : (B, W, num_tags)
        loss   : scalar  (if bio_ids provided)
        """
        # ── feature mapping (Eq 4a): tc → tm_c  (trainable) ──────────
        tm_c = self.feat_map(node_embeds_tc)               # (B, W, d)
        topic_embeds = self.feat_map(topic_tc)              # (T, d)

        # ── topic-guided attention (Eq 6-7) ───────────────────────────
        trs = self.topic_attn(tm_c, topic_embeds)           # (B, W, d)

        # ── reconstruction (Eq 4b + Eq 5) — trainable ────────────────
        trecon_c = self.feat_recon(tm_c)                    # (B, W, d)
        cos_sim = F.cosine_similarity(
            node_embeds_tc.reshape(-1, self.rgcn_dim),
            trecon_c.reshape(-1, self.rgcn_dim),
            dim=1,
        ).mean()
        recon_loss = 1.0 - cos_sim                          # Eq 5

        # ── ts : per-word BERT features  (B, W, 768) ─────────────────
        ts = self.bert_enc.extract_word_features(
            input_ids, attention_mask, word_ids, num_words
        )
        W = ts.size(1)

        # ── align knowledge tensors to the same W ─────────────────────
        tm_c = tm_c[:, :W, :]
        trs  = trs[:, :W, :]

        # ── concatenate  (Eq 8) ──────────────────────────────────────
        vt = torch.cat([tm_c, trs, ts], dim=-1)            # (B, W, 1168)
        vt = self.dropout(vt)

        # ── classification  (Eq 9) ───────────────────────────────────
        logits = self.classifier(vt)                        # (B, W, num_tags)

        loss = None
        if bio_ids is not None:
            bio_ids_trunc = bio_ids[:, :W]
            softmax_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                bio_ids_trunc.reshape(-1),
                ignore_index=-100,
            )
            # Eq 10: L = L_softmax + μ * L_recon
            loss = softmax_loss + self.mu * recon_loss

        return logits, loss
