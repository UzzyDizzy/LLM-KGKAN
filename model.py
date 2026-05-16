#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable()
        self.backbone = prepare_model_for_kbit_training(
            self.backbone,
            use_gradient_checkpointing=True,
        )

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
        if self.backbone.config.hidden_size == cfg.hidden_size:
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
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
        prefix_mask = torch.ones(
            B,
            2 * self.prefix_len,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
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
    def __init__(self, hidden_size: int, num_relations: int, dropout: float, relation_mode: str = "full"):
        super().__init__()
        self.relation_mode = relation_mode
        if relation_mode == "full":
            self.rel_weights = nn.Parameter(torch.zeros(num_relations, hidden_size, hidden_size))
        elif relation_mode == "diagonal":
            self.rel_diag = nn.Parameter(torch.zeros(num_relations, hidden_size))
        else:
            raise ValueError(f"Unknown RGCN relation_mode: {relation_mode}")
        self.self_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.self_weight.weight)
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
                if self.relation_mode == "full":
                    out = out + torch.einsum("btd,df->btf", neigh, self.rel_weights[r])
                else:
                    out = out + neigh * self.rel_diag[r]
        out = F.gelu(out)
        out = self.dropout(out)
        return self.norm(h + out)


class SyntaxEncoder(nn.Module):
    def __init__(self, cfg: LLMKGKANConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RGCNLayer(
                    cfg.hidden_size,
                    cfg.num_dep_relations,
                    cfg.dropout,
                    relation_mode=getattr(cfg, "rgcn_relation_mode", "full"),
                )
                for _ in range(cfg.rgcn_layers)
            ]
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

        self.proj = nn.Linear(cfg.kg_emb_dim, cfg.hidden_size, bias=False)
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
        else:
            return h + r + t

    def forward(self, kg_heads, kg_rels, kg_tails, kg_mask, kg_token_map, kg_aspect_ids, kg_num_aspects=None):
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
            valid_aspect_ids = aspect_ids[kg_mask[b] > 0]
            if kg_num_aspects is not None:
                n_aspects = int(kg_num_aspects[b].item())
            elif valid_aspect_ids.numel() > 0:
                n_aspects = int(valid_aspect_ids.max().item()) + 1
            else:
                n_aspects = 0

            for a in range(n_aspects):
                idx = (aspect_ids == a) & (kg_mask[b] > 0)

                if idx.sum() == 0:
                    aspect_vecs.append(torch.zeros(self.cfg.kg_emb_dim, device=device))
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
        # Optimized vectorized KAN layer implementation to prevent 100GB OOM.
        # We use grid_size=8 as is standard for KANs.
        grid_size = 8
        self.w1 = nn.Parameter(torch.randn(input_dim, grid_size) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(input_dim, grid_size))
        self.w2 = nn.Parameter(torch.randn(input_dim, grid_size) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # x is (B, T, D)
        h = F.gelu(x.unsqueeze(-1) * self.w1 + self.b1) # (B, T, D, G)
        out = (h * self.w2).sum(dim=-1) + self.b2       # (B, T, D)
        return out

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
        nn.init.zeros_(self.g_tr.weight)
        nn.init.zeros_(self.g_tr.bias)
        nn.init.zeros_(self.g_pr.weight)
        nn.init.constant_(self.g_pr.bias, 2.0)

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
        self.register_buffer("class_weights", torch.ones(cfg.num_labels), persistent=False)

    def set_class_weights(self, weights: torch.Tensor) -> None:
        weights = weights.detach().float().clamp_min(1e-6)
        self.class_weights.copy_(weights.to(self.class_weights.device))

    def token_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, self.cfg.num_labels),
            labels.reshape(-1),
            ignore_index=self.cfg.ignore_index,
            weight=self.class_weights.to(logits.device, dtype=logits.dtype),
        )

    def encode(self, batch):
        aspect_vecs = self.kg(
            batch["kg_heads"],
            batch["kg_rels"],
            batch["kg_tails"],
            batch["kg_mask"],
            batch["kg_token_map"],
            batch["kg_aspect_ids"],
            batch.get("kg_num_aspects"),
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

            for a_idx, span in enumerate(spans):
                if a_idx < len(aspect_vecs[b]):
                    a_vec = aspect_vecs[b][a_idx]
                else:
                    a_vec = torch.zeros(self.cfg.kg_emb_dim, device=device)
                a_vec_proj = self.kg.proj(a_vec)

                start, end = int(span[0]), int(span[1])
                start = max(0, min(start, T))
                end = max(start, min(end, T))

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

        loss = logits.new_tensor(0.0)

        # ✅ SOURCE LOSS (MAIN TRAINING SIGNAL)
        if src_mask.any():
            loss += self.token_loss(logits[src_mask], labels[src_mask])

        # ✅ TARGET LOSS (few-shot ONLY)
        if tgt_mask.any() and batch["labels"][tgt_mask].ne(self.cfg.ignore_index).any():
            loss += self.token_loss(logits[tgt_mask], labels[tgt_mask])

        # ✅ MMD ALIGNMENT
        if src_mask.any() and tgt_mask.any():
            token_mask = batch["attention_mask"].to(z.device).bool()
            z_src = z[src_mask][token_mask[src_mask]]
            z_tgt = z[tgt_mask][token_mask[tgt_mask]]

            align_loss = mmd_loss(z_src, z_tgt)

            loss = loss + self.cfg.mmd_lambda * align_loss
        
        output["loss"] = loss
        return output
