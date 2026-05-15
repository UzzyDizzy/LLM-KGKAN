# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config

cfg = Config()


# =====================================================
# Gradient Reversal Layer
# =====================================================

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grl(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# =====================================================
# Base network = feature extractor + task classifier
# =====================================================

class BaseNet(nn.Module):

    def __init__(self, vocab_size, pos_size, emb_matrix):
        super().__init__()

        # ---------------- word embedding ----------------
        self.word_emb = nn.Embedding(
            vocab_size,
            cfg.word_dim,
            padding_idx=0
        )

        self.word_emb.weight.data.copy_(
            torch.tensor(
                emb_matrix,
                dtype=torch.float32
            )
        )

        # ---------------- pos embedding ----------------
        self.pos_emb = nn.Embedding(
            pos_size,
            cfg.pos_dim,
            padding_idx=0
        )

        # ---------------- dropout on embeddings ----------------
        self.emb_drop = nn.Dropout(cfg.dropout)

        # ---------------- BiLSTM ----------------
        self.lstm = nn.LSTM(
            input_size = cfg.word_dim + cfg.pos_dim,
            hidden_size = cfg.hidden,
            num_layers = cfg.num_layers,
            batch_first = True,
            dropout = cfg.dropout,
            bidirectional = True
        )

        # ---------------- dropout on hidden ----------------
        self.h_drop = nn.Dropout(cfg.dropout)

        # ---------------- classifier ----------------
        self.fc = nn.Linear(
            cfg.hidden * 2,
            len(cfg.labels)
        )

    # -------------------------------------------------

    def feature_extractor(self, x, p):

        ew = self.word_emb(x)
        ep = self.pos_emb(p)

        e = torch.cat([ew, ep], dim=-1)

        e = self.emb_drop(e)

        h, _ = self.lstm(e)

        h = self.h_drop(h)

        return h

    # -------------------------------------------------

    def task_classifier(self, h):

        logits = self.fc(h)

        prob = F.softmax(logits, dim=-1)

        return logits, prob

    # -------------------------------------------------

    def forward(self, x, p):

        h = self.feature_extractor(x, p)

        logits, prob = self.task_classifier(h)

        return h, logits, prob
class DomainDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        d = cfg.hidden * 2

        # Eq (4)
        self.Wm = nn.Linear(d, d)
        self.vm = nn.Linear(d, 1)

        # Eq (7)
        self.fc = nn.Linear(d, 2)

    def forward(self, h, mask=None):
        """
        h: [B,T,D]
        """

        # Eq (4)
        m = torch.tanh(self.Wm(h))

        score = self.vm(m).squeeze(-1)   # [B,T]

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # Eq (5)
        alpha = F.softmax(score, dim=1)

        # Eq (6)
        r = torch.sum(
            h * alpha.unsqueeze(-1),
            dim=1
        )

        # Eq (7)
        logits = self.fc(r)

        prob = F.softmax(logits, dim=-1)

        return logits, prob
class AHF(nn.Module):

    def __init__(self,
                 vocab_size,
                 pos_size,
                 emb_matrix):
        super().__init__()

        # ---------------- student ----------------
        self.student = BaseNet(
            vocab_size,
            pos_size,
            emb_matrix
        )

        # ---------------- teacher ----------------
        self.teacher = BaseNet(
            vocab_size,
            pos_size,
            emb_matrix
        )

        # ---------------- discriminator ----------------
        self.domain = DomainDiscriminator()

        # initialize teacher = student
        self.teacher.load_state_dict(
            self.student.state_dict()
        )

        # teacher no gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

    # -------------------------------------------------

    @torch.no_grad()
    def ema_update(self):
        """
        Eq (9)
        θT = γ θT + (1-γ) θS
        """

        for tp, sp in zip(
            self.teacher.parameters(),
            self.student.parameters()
        ):
            tp.data = (
                cfg.gamma * tp.data
                + (1 - cfg.gamma) * sp.data
            )

    # -------------------------------------------------

    def domain_forward(self, h, mask):

        rev = grl(h, cfg.eta)

        logits, prob = self.domain(
            rev,
            mask
        )

        return logits, prob