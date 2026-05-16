# model.py — Paper-faithful architecture (Zou & Wang, Neurocomputing 2025)
import torch
import torch.nn as nn
import math
from config import (
    MAX_LEN, NUM_LABELS, DROPOUT, NUM_PROMPTS, FREEZE_ENCODER,
    ASPECT_ID_MAP, SENTIMENT_ID_MAP, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, HF_TOKEN,
)
from utils import build_representation_batch, extract_spans_from_preds, get_relative_positions
from peft import get_peft_model, LoraConfig
from transformers import LlamaModel


# ===================== Gradient Reversal Layer (Eq 20-21) =====================
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


# ===================== Main Model =====================
class LLMSynABSA(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoModel
        from transformers import LlamaModel

        # ===== Backbone: LLaMA encoder (Eq 1) =====
        self.encoder = LlamaModel.from_pretrained(
            model_name,
            token=HF_TOKEN if HF_TOKEN else None,
            torch_dtype=torch.bfloat16,
        )
        d = self.encoder.config.hidden_size  # 4096 for LLaMA-3-8B

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.encoder = get_peft_model(self.encoder, lora_config)

        # self.encoder.enable_input_require_grads()
        # self.encoder.gradient_checkpointing_enable(
        #     gradient_checkpointing_kwargs={"use_reentrant": False}
        # )

        self.dropout = nn.Dropout(DROPOUT)

        # ===== Aspect & Sentiment Classifiers (Eq 2, 3) =====
        self.aspect_fc = nn.Sequential(
            nn.Linear(d, len(ASPECT_ID_MAP)),
        )
        self.sentiment_fc = nn.Sequential(
            nn.Linear(d, len(SENTIMENT_ID_MAP)),
        )

        # ===== Syntax-Aware Transformer (Eq 7–14) =====
        self.num_heads = 32
        self.dk = d // self.num_heads

        self.rel_emb = nn.Embedding(2 * MAX_LEN + 1, self.dk)

        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

        # Per-head θ, φ scalars (Eq 8)
        self.theta = nn.Parameter(torch.randn(self.num_heads) * 0.02)
        self.phi = nn.Parameter(torch.randn(self.num_heads) * 0.02)

        # LayerNorms (Eq 12, 14)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        # FFN (Eq 13) — paper intermediate = 11008
        ffn_dim = 11008
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(ffn_dim, d),
        )

        # ===== Domain Topic Predictor (Eq 15–21) =====
        self.domain_attn = nn.Linear(d * 2, 1)
        self.domain_proj = nn.Linear(d * 2, d)
        self.domain_fc = nn.Linear(d, 2)  # source vs target

        # ===== Soft Prompt Learning (Eq 22) =====
        # Feature-conditioned prompt generation (conditioned on H, A, S, G)
        self.prompt_proj = nn.Linear(d * 4, NUM_PROMPTS * d)
        self.prompt_norm = nn.LayerNorm(d)

        # ===== Feature Aggregation (Eq 23–24) =====
        self.gates = nn.ModuleList([
            nn.Linear(d * 6, 1)
            for _ in range(6)
        ])

        # ===== Unified Classifier (Eq 25) =====
        self.classifier = nn.Linear(d, NUM_LABELS)

        self._init_weights()

    def _init_weights(self):
        """Initialize added components with small values (paper: initializer_range=0.02)."""
        for module in [self.aspect_fc, self.sentiment_fc, self.Wq, self.Wk,
                       self.Wv, self.Wo, self.domain_attn, self.domain_proj,
                       self.domain_fc, self.prompt_proj, self.classifier]:
            for m in module.modules() if isinstance(module, nn.Sequential) else [module]:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for g in self.gates:
            nn.init.constant_(g.bias, -2.0)
            nn.init.normal_(g.weight, std=0.02)
            #nn.init.zeros_(g.bias)

    def forward(self, ids, mask, texts, syntax_cache,
            aspect_labels=None, sentiment_labels=None,
            current_step=0, total_steps=1):
        B, L = ids.shape

        # ===== Eq 1: Contextual feature representation =====
        if FREEZE_ENCODER:
            with torch.no_grad():
                H = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            H = H.detach()  # ensure no grad flows back
        else:
            H = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state

        H = H.to(self.classifier.weight.dtype)

        # ===== Eq 2, 3: Aspect & Sentiment classifiers =====
        H_drop = self.dropout(H)
        aspect_logits = self.aspect_fc(H_drop)       # (B, L, 3)
        sentiment_logits = self.sentiment_fc(H_drop)  # (B, L, 4)

        # Extract spans: use gold labels during training, predictions during inference
        if aspect_labels is not None and sentiment_labels is not None:
            aspect_spans = extract_spans_from_preds(
                aspect_labels.cpu().tolist(), ASPECT_ID_MAP)
            sentiment_spans = extract_spans_from_preds(
                sentiment_labels.cpu().tolist(), SENTIMENT_ID_MAP)
        else:
            aspect_preds = torch.argmax(aspect_logits, dim=-1)
            sentiment_preds = torch.argmax(sentiment_logits, dim=-1)
            aspect_spans = extract_spans_from_preds(
                aspect_preds.cpu().tolist(), ASPECT_ID_MAP)
            sentiment_spans = extract_spans_from_preds(
                sentiment_preds.cpu().tolist(), SENTIMENT_ID_MAP)

        # Eq 4, 5: Mean-pooled aspect/sentiment representations
        A_vec = build_representation_batch(H, aspect_spans)   # (B, d)
        S_vec = build_representation_batch(H, sentiment_spans)  # (B, d)

        A = A_vec.unsqueeze(1).expand_as(H)  # (B, L, d)
        S = S_vec.unsqueeze(1).expand_as(H)  # (B, L, d)

        # ===== Syntax-Aware Transformer (Eq 7–14) =====
        F_syn = self._syntax_transformer(H, texts, syntax_cache,
                                          aspect_spans, sentiment_spans)

        # ===== Domain Topic Predictor (Eq 15–21) =====
        G, domain_logits = self._domain_predictor(
            H, A, current_step, total_steps
        )  # G: (B, d)

        # ===== Soft Prompt Learning (Eq 22) =====
        P = self._prompt_learning(H, A_vec, S_vec, G)  # (B, L, d)

        # ===== Feature Aggregation (Eq 23–24) =====
        G_exp = G.unsqueeze(1).expand_as(H)  # (B, L, d)
        concat = torch.cat([H, A, S, F_syn, G_exp, P], dim=-1)  # (B, L, 6d)

        gates = [torch.sigmoid(g(concat)) for g in self.gates]

        psi = (gates[0] * H +
               gates[1] * A +
               gates[2] * S +
               gates[3] * F_syn +
               gates[4] * G_exp +
               gates[5] * P)  # (B, L, d)

        # ===== Eq 25: Unified classifier =====
        out = self.classifier(self.dropout(psi))  # (B, L, 7)

        return out, aspect_logits, sentiment_logits, domain_logits

    def _syntax_transformer(self, H, texts, syntax_cache,
                            aspect_spans, sentiment_spans):
        """Syntax-aware Transformer (Eq 7–14)."""
        B, L, d = H.shape

        # Eq 7: Q, K, V projections
        Q = self.Wq(H).view(B, L, self.num_heads, self.dk).permute(0, 2, 1, 3)
        K = self.Wk(H).view(B, L, self.num_heads, self.dk).permute(0, 2, 1, 3)
        V = self.Wv(H).view(B, L, self.num_heads, self.dk).permute(0, 2, 1, 3)
        # Q, K, V: (B, heads, L, dk)

        # Relative position embeddings
        rel_pos = get_relative_positions(L)  # (L, L) indices
        E_p = self.rel_emb(rel_pos)          # (L, L, dk)

        # Eq 8: Attention scores with 4 terms
        # term1: Q · K^T
        term1 = torch.matmul(Q, K.transpose(-2, -1))  # (B, heads, L, L)

        # term2: Q · E_p^T
        E_p_exp = E_p.unsqueeze(0).expand(B, -1, -1, -1)  # (B, L, L, dk)
        term2 = torch.einsum("bhld,blkd->bhlk", Q, E_p_exp)  # (B, heads, L, L)

        # θ and ϕ per-head scalars
        theta = self.theta.view(1, self.num_heads, 1, 1)
        phi = self.phi.view(1, self.num_heads, 1, 1)

        # term3: θ * (1 · K_v^T)
        # preserve token-token structure
        term3 = theta * K.mean(dim=-1).unsqueeze(-2)
        # shape: (B, heads, 1, L)
        term3 = term3.expand(-1, -1, L, -1)

        # term4: ϕ * (1 · E_puv^T)
        term4 = phi * E_p.mean(dim=-1)
        term4 = term4.expand(B, -1, -1, -1)

        scores = (term1 + term2 + term3 + term4) / math.sqrt(self.dk)

        # Eq 6, 9: Add syntax-aware dependency matrix M
        M = torch.stack([
            syntax_cache.get(
                t,
                torch.zeros(MAX_LEN, MAX_LEN)
            )
            for t in texts
        ]).to(H.device)

        # Rule 1: Dynamic intra-span visibility for detected aspects/sentiments
        for b in range(B):
            for s, e in aspect_spans[b]:
                if s < MAX_LEN and e <= MAX_LEN:
                    M[b, s:e, s:e] = 0.0
            for s, e in sentiment_spans[b]:
                if s < MAX_LEN and e <= MAX_LEN:
                    M[b, s:e, s:e] = 0.0

        M = M[:, :L, :L]  # trim to actual sequence length
        
        scores = scores + M.unsqueeze(1).to(H.dtype)  # Eq 9: λ* = λ + M

        # Eq 10: Attention output
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        H_att = torch.matmul(attn, V)  # (B, heads, L, dk)

        # Eq 11: Merge heads
        H_att = H_att.permute(0, 2, 1, 3).contiguous().view(B, L, d)
        H_att = self.Wo(H_att)

        # Eq 12: Residual + LayerNorm
        H_att = self.norm1(H + H_att)

        # Eq 13–14: FFN + Residual + LayerNorm
        F_out = self.norm2(H_att + self.ffn(H_att))

        return F_out  # (B, L, d)

    def _domain_predictor(self, H, A, current_step, total_steps):
        """
        Domain topic predictor with adversarial training (Eq 15–21).
        GRL applied BEFORE the attention layer (paper Section 3.4).
        """
        B, L, d = H.shape

        # Concatenate contextual and aspect features (paper: h_c = [h_C; h_A])
        HC = torch.cat([H, A], dim=-1)  # (B, L, 2d)

        # Apply GRL BEFORE attention (paper: "gradient reversal layer before the attention layer")
        p = current_step / total_steps
        lambda_ = 0.1 * (2. / (1. + math.exp(-5 * p)) - 1)
        HC_rev = GRL.apply(HC, lambda_)
        #HC_rev = GRL.apply(HC, 1.0)

        # Eq 15: Attention scores
        delta = torch.tanh(self.domain_attn(HC_rev))  # (B, L, 1)

        # Eq 16: Attention weights
        gamma = torch.softmax(delta, dim=1)  # (B, L, 1)

        # Eq 17: Weighted sum → domain topic feature
        G_full = torch.sum(gamma * HC_rev, dim=1)  # (B, 2d)
        G = self.domain_proj(G_full)  # (B, d)

        # Eq 18: Domain classification
        domain_logits = self.domain_fc(G)  # (B, 2)

        return G, domain_logits

    def _prompt_learning(self, H, A_vec, S_vec, G):
        """
        Soft prompt learning conditioned on analyzed features (Eq 22).
        Generates prompt embeddings from contextual, aspect, sentiment,
        and domain topic features.
        """
        B, L, d = H.shape

        # Feature summary from analyzed representations
        H_mean = H.mean(dim=1)  # (B, d)
        feature_concat = torch.cat([H_mean, A_vec, S_vec, G], dim=-1)  # (B, 4d)

        # Generate prompt embeddings
        prompt_flat = self.prompt_proj(feature_concat)  # (B, m*d)
        prompt_emb = prompt_flat.view(B, NUM_PROMPTS, d)  # (B, m, d)
        prompt_emb = self.prompt_norm(prompt_emb)

        # Mean pool prompt embeddings and broadcast
        P = prompt_emb.mean(dim=1, keepdim=True)  # (B, 1, d)
        P = P.expand(-1, L, -1)  # (B, L, d)

        return P
