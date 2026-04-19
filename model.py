#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model

from config import LLMKGKANConfig
from utils import mmd_loss
from transformers import BitsAndBytesConfig


class SemanticEncoder(nn.Module):
    def __init__(self, cfg: LLMKGKANConfig):
        super().__init__()
        base_cfg = AutoConfig.from_pretrained(cfg.llm_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        self.backbone = AutoModel.from_pretrained(
            cfg.llm_name,
            config=base_cfg,
            device_map="auto",              # multi-GPU / auto placement
            quantization_config=bnb_config,  # QLoRA
            torch_dtype=torch.float16,
        )
        self.backbone.gradient_checkpointing_enable()

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if "deberta" in cfg.llm_name.lower():
            target_modules = ["query_proj", "key_proj", "value_proj"]
        elif "bert" in cfg.llm_name.lower():
            target_modules = ["query", "key", "value"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(self.backbone, peft_cfg)
        self.proj = nn.Linear(self.backbone.config.hidden_size, cfg.hidden_size)
        # === PREFIX MODULES (Eq 4) ===
        self.prefix_len = cfg.prefix_len
        d = self.backbone.config.hidden_size

        self.task_prefix = nn.Parameter(torch.randn(1, self.prefix_len, d))

        self.kg_prefix_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Linear(d, self.prefix_len * d)
        )
            

    def forward(self, input_ids, attention_mask, kg_summary=None):
        B = input_ids.size(0)
        d = self.backbone.config.hidden_size

        # --- token embeddings ---
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)

        # --- task prefix ---
        task_prefix = self.task_prefix.expand(B, -1, -1)  # (B, Lp, d)

        # --- KG prefix ---
        if kg_summary is not None:
            kg_prefix = self.kg_prefix_mlp(kg_summary)  # (B, Lp*d)
            kg_prefix = kg_prefix.view(B, self.prefix_len, d)
        else:
            kg_prefix = torch.zeros_like(task_prefix)

        # --- CONCAT (Eq 4) ---
        inputs_embeds = torch.cat([task_prefix, kg_prefix, inputs_embeds], dim=1)

        # --- attention mask fix ---
        prefix_mask = torch.ones(B, 2 * self.prefix_len).to(attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # --- forward ---
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        # --- REMOVE PREFIX TOKENS ---
        hidden = out.last_hidden_state[:, 2*self.prefix_len:, :]

        return self.proj(hidden)


class RGCNLayer(nn.Module):
    def __init__(self, hidden_size: int, num_relations: int, dropout: float):
        super().__init__()
        self.rel_weights = nn.Parameter(torch.randn(num_relations, hidden_size, hidden_size) * 0.02)
        self.self_weight = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.num_relations = num_relations

    def forward(self, h: torch.Tensor, dep_rel_ids: torch.Tensor, dep_adj: torch.Tensor) -> torch.Tensor:
        out = self.self_weight(h)
        for r in range(self.num_relations):
            rel_mask = (dep_rel_ids == r) & dep_adj.bool()
            if rel_mask.any():
                neigh = torch.matmul(rel_mask.float(), h)
                deg = rel_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                neigh = neigh / deg
                out = out + torch.einsum("btd,df->btf", neigh, self.rel_weights[r])
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(h + out)


class SyntaxEncoder(nn.Module):
    def __init__(self, cfg: LLMKGKANConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [RGCNLayer(cfg.hidden_size, cfg.num_dep_relations, cfg.dropout) for _ in range(cfg.rgcn_layers)]
        )

    def forward(self, token_features: torch.Tensor, dep_rel_ids: torch.Tensor, dep_adj: torch.Tensor) -> torch.Tensor:
        h = token_features
        for layer in self.layers:
            h = layer(h, dep_rel_ids, dep_adj)
        return h


class KGEncoder(nn.Module):
    def __init__(self, cfg: LLMKGKANConfig):
        super().__init__()
        self.cfg = cfg

        self.ent_emb = nn.Embedding(cfg.num_entities, cfg.kg_emb_dim)
        self.rel_emb = nn.Embedding(cfg.num_kg_relations, cfg.kg_emb_dim)

        self.proj = nn.Linear(cfg.kg_emb_dim, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size)

        # 🔥 NEW: attention projections
        self.query_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key_proj = nn.Linear(cfg.kg_emb_dim, cfg.hidden_size)
        self.value_proj = nn.Linear(cfg.kg_emb_dim, cfg.hidden_size)

    def triple_encode(self, heads, rels, tails):
        h = F.normalize(self.ent_emb(heads), dim=-1)
        r = F.normalize(self.rel_emb(rels), dim=-1)
        t = F.normalize(self.ent_emb(tails), dim=-1)

        if self.cfg.use_distmult:
            return h * r * t
        else:  # TransE-style
            return h + r - t

    def forward(self, kg_heads, kg_rels, kg_tails, kg_mask, kg_token_map, kg_aspect_ids):
        """
        STRICT Eq (10–12) implementation
        """

        B, K = kg_heads.shape
        device = kg_heads.device

        # ---- Eq (10): triple encoding ----
        triple_repr = self.triple_encode(kg_heads, kg_rels, kg_tails)  # (B,K,d)

        mask = kg_mask.unsqueeze(-1)
        triple_repr = triple_repr * mask

        # ---- Eq (11): aspect-level aggregation ----
        batch_aspect_vecs = []

        for b in range(B):
            aspect_vecs = []

            aspect_ids = kg_aspect_ids[b]
            unique_aspects = torch.unique(aspect_ids)

            for a in unique_aspects:
                idx = (aspect_ids == a) & (kg_mask[b] > 0)

                if idx.sum() == 0:
                    continue

                # mean over triples in Ga
                vec = triple_repr[b][idx].mean(dim=0)
                aspect_vecs.append(vec)

            if len(aspect_vecs) == 0:
                aspect_vecs = [torch.zeros(self.cfg.kg_emb_dim, device=device)]

            aspect_vecs = torch.stack(aspect_vecs)  # (A,d)
            batch_aspect_vecs.append(aspect_vecs)

        # ---- project to hidden size ----
        # we will broadcast later
        return batch_aspect_vecs


class KANLayer(nn.Module):
    """
    True Kolmogorov–Arnold style layer:
    z_i = sum_j phi_j(x_j)
    where phi_j are learnable univariate functions
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # Each dimension gets its own function
        self.phi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(input_dim)
        ])

    def forward(self, x):
        # x: (B, T, D)
        outs = []
        for i, fn in enumerate(self.phi):
            xi = x[..., i:i+1]  # (B,T,1)
            outs.append(fn(xi))
        return torch.cat(outs, dim=-1)  # (B,T,D)

class KANStyleFusion(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.input_dim = hidden_size * 3

        self.kan = KANLayer(self.input_dim, hidden_size // 2)

        self.proj = nn.Linear(self.input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sem, syn, rel):
        x = torch.cat([sem, syn, rel], dim=-1)

        x_kan = self.kan(x)

        z = self.proj(x_kan)
        z = self.dropout(F.gelu(z))

        return self.norm(z)


class ARGModule(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.g_tr = nn.Linear(hidden_size, hidden_size)
        self.g_pr = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, z: torch.Tensor, sem: torch.Tensor, syn: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        g_tr = torch.sigmoid(self.g_tr(z))
        g_pr = torch.sigmoid(self.g_pr(z))
        out = g_tr * rel + g_pr * sem + (1.0 - g_tr) * syn
        return self.norm(out)


class LLMKGKAN(nn.Module):
    def __init__(self, cfg: LLMKGKANConfig):
        super().__init__()
        self.cfg = cfg
        self.semantic = SemanticEncoder(cfg)
        self.syntax = SyntaxEncoder(cfg)
        self.kg = KGEncoder(cfg)
        self.fusion = KANStyleFusion(cfg.hidden_size, cfg.dropout)
        self.arg = ARGModule(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_labels)

    def encode(self, batch):
        # --- KG summary (Eq 11) ---
        triple_repr = self.kg.triple_encode(
            batch["kg_heads"],
            batch["kg_rels"],
            batch["kg_tails"]
        )

        mask = batch["kg_mask"].unsqueeze(-1).to(batch["kg_heads"].device)
        triple_repr = triple_repr * mask

        aspect_vecs = self.kg(
            batch["kg_heads"],
            batch["kg_rels"],
            batch["kg_tails"],
            batch["kg_mask"],
            batch["kg_token_map"],
            batch["kg_aspect_ids"],
        )

        # average over ALL aspects (paper does not specify weighting)
        device = batch["kg_heads"].device

        kg_summary = torch.stack([
            a.mean(dim=0) if len(a) > 0 else torch.zeros(self.cfg.kg_emb_dim, device=device)
            for a in aspect_vecs
        ])
        kg_summary = self.kg.proj(kg_summary)

        # --- semantic with prefix ---
        sem = self.semantic(
            batch["input_ids"],
            batch["attention_mask"],
            kg_summary=kg_summary
        )
        syn = self.syntax(sem, batch["dep_rel_ids"], batch["dep_adj"])
        

        B, T, _ = sem.shape
        device = sem.device

        rel = torch.zeros(B, T, self.cfg.hidden_size, device=device)

        for b in range(B):
            spans = batch["aspect_spans"][b]

            if len(spans) == 0:
                continue

            for a_idx, a_vec in enumerate(aspect_vecs[b]):
                a_vec_proj = self.kg.proj(a_vec)

                for span in spans:
                    start, end = int(span[0]), int(span[1])

                    if end > start:
                        rel[b][start:end] = a_vec_proj

        # normalize
        rel = self.kg.norm(rel)

        z = self.fusion(sem, syn, rel)
        z_prime = self.arg(z, sem, syn, rel)
        return sem, syn, rel, z, z_prime

    def forward(self, batch):
        sem, syn, rel, z, z_prime = self.encode(batch)
        logits = self.classifier(z_prime)

        output = {"logits": logits}

        labels = batch["labels"]
        domain_ids = batch["domain_ids"]

        src_mask = domain_ids == 0
        tgt_mask = domain_ids == 1

        loss = 0

        # ✅ SOURCE LOSS (MAIN TRAINING SIGNAL)
        if src_mask.any():
            loss += F.cross_entropy(
                logits[src_mask].view(-1, self.cfg.num_labels),
                labels[src_mask].view(-1),
                ignore_index=self.cfg.ignore_index
            )

        # ✅ TARGET LOSS (few-shot ONLY)
        if tgt_mask.any() and batch["labels"][tgt_mask].ne(self.cfg.ignore_index).any():
            loss += F.cross_entropy(
                logits[tgt_mask].view(-1, self.cfg.num_labels),
                labels[tgt_mask].view(-1),
                ignore_index=self.cfg.ignore_index
            )

        # ✅ MMD ALIGNMENT
        if src_mask.any() and tgt_mask.any():
            z_src = z[src_mask]
            z_tgt = z[tgt_mask]

            z_src = z_src.mean(dim=1)
            z_tgt = z_tgt.mean(dim=1)

            align_loss = mmd_loss(z_src, z_tgt)

            loss = loss + self.cfg.mmd_lambda * align_loss
        
        output["loss"] = loss
        return output
