#train.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from model import AHF
from metrics import span_f1_score

scaler = torch.cuda.amp.GradScaler(
    enabled=torch.cuda.is_available()
)

cfg = Config()

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def pad(seq, maxlen, val):
    seq = seq[:maxlen]
    return seq + [val] * (maxlen - len(seq))


def make_batches(data, bs):

    idx = list(range(len(data)))
    random.shuffle(idx)

    for i in range(0, len(idx), bs):
        yield [data[j] for j in idx[i:i+bs]]


def batch_to_tensor(batch, labeled=True):

    toks = [x["tokens"] for x in batch]

    x = [pad(z["x"], cfg.max_len, 0) for z in batch]
    p = [pad(z["p"], cfg.max_len, 0) for z in batch]

    lens = [min(len(z["x"]), cfg.max_len) for z in batch]

    mask = []
    for L in lens:
        mask.append(
            [1]*L + [0]*(cfg.max_len-L)
        )

    x = torch.tensor(
        x,
        dtype=torch.long,
        pin_memory=cfg.pin_memory
    ).to(cfg.device, non_blocking=True)

    p = torch.tensor(
        p,
        dtype=torch.long,
        pin_memory=cfg.pin_memory
    ).to(cfg.device, non_blocking=True)

    mask = torch.tensor(
        mask,
        dtype=torch.float32,
        pin_memory=cfg.pin_memory
    ).to(cfg.device, non_blocking=True)

    if labeled:
        y = [
            pad(z["y"], cfg.max_len, -100)
            for z in batch
        ]
        y = torch.tensor(
            y,
            dtype=torch.long,
            pin_memory=cfg.pin_memory
        ).to(cfg.device, non_blocking=True)
    else:
        y = None

    return toks, x, p, y, lens, mask
def compute_threshold(confs, rho):

    if len(confs) == 0:
        return rho

    confs = sorted(confs, reverse=True)

    k = max(
        1,
        int(len(confs) * cfg.beta / 100)
    )

    q_beta = confs[k-1]

    tau = max(rho, q_beta)

    return tau

@torch.no_grad()
def evaluate(model, dataset):

    model.eval()

    preds = []
    golds = []
    toks_all = []

    for batch in make_batches(
        dataset, cfg.batch_size
    ):

        toks,x,p,y,lens,mask = batch_to_tensor(
            batch, labeled=True
        )

        _,_,prob = model.teacher(x,p)

        pred = prob.argmax(-1).cpu().numpy()
        gold = y.cpu().numpy()

        for i in range(len(batch)):

            preds.append(
                pred[i][:lens[i]].tolist()
            )

            golds.append(
                gold[i][:lens[i]].tolist()
            )

            toks_all.append(toks[i])

    p,r,f1 = span_f1_score(
        preds,
        golds,
        toks_all
    )

    return p,r,f1

def train_one_pair(
    src_train,
    src_val,
    tgt_train,
    tgt_test,
    word2id,
    pos2id,
    emb_matrix,
    run_name="run"
):

    model = AHF(
        len(word2id),
        len(pos2id),
        emb_matrix
    ).to(cfg.device)

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=cfg.lr
    )

    ce_loss = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    best_val = -1
    best_path = f"{cfg.model_dir}/{run_name}.pt"

    # ====================================
    for epoch in range(cfg.epochs):

        model.train()

        src_batches = list(
            make_batches(
                src_train,
                cfg.batch_size
            )
        )

        tgt_batches = list(
            make_batches(
                tgt_train,
                cfg.batch_size
            )
        )

        steps = min(
            len(src_batches),
            len(tgt_batches)
        )

        losses = []

        for step in range(steps):

            sb = src_batches[step]
            tb = tgt_batches[step]

            # ---------------- source ----------------
            stoks, sx, sp, sy, slens, smask = batch_to_tensor(
                sb, labeled=True
            )

            # ---------------- target ----------------
            ttoks, tx, tp, _, tlens, tmask = batch_to_tensor(
                tb, labeled=False
            )

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):

                # ====================================
                # source student
                hs, logits_s, prob_s = model.student(sx, sp)

                # Eq (10)
                Lcr = ce_loss(
                    logits_s.view(-1, len(cfg.labels)),
                    sy.view(-1)
                )

                # ====================================
                # target student
                ht_s, logits_ts, prob_ts = model.student(tx, tp)

                # teacher pseudo labels
                with torch.no_grad():
                    ht_t, logits_tt, prob_tt = model.teacher(tx, tp)

                # ====================================
                # thresholds
                pred = prob_tt.argmax(-1)
                conf = prob_tt.max(-1).values

                conf_o = conf[(pred == 0) * tmask.bool()]
                conf_a = conf[(pred != 0) * tmask.bool()]

                tau1 = compute_threshold(
                    conf_o.detach().cpu().tolist(),
                    cfg.rho1
                )

                tau2 = compute_threshold(
                    conf_a.detach().cpu().tolist(),
                    cfg.rho2
                )

                # ====================================
                # domain discriminator
                h_all = torch.cat([hs, ht_s], dim=0)
                mask_all = torch.cat([smask, tmask], dim=0)

                dlogits, dprob = model.domain_forward(
                    h_all,
                    mask_all
                )

                z = torch.cat([
                    torch.ones(len(sb)),
                    torch.zeros(len(tb))
                ]).long().to(cfg.device)

                Ladv = ce_loss(dlogits, z)

                # ====================================
                # Eq (13) vectorized
                tgt_di = dprob[len(sb):, 1]     # P(source)

                tau_tensor = torch.where(
                    pred == 0,
                    torch.full_like(conf, tau1),
                    torch.full_like(conf, tau2)
                )

                M = (conf > tau_tensor).float()
                M = M * tmask

                diff = (prob_tt - prob_ts) ** 2
                token_loss = diff.sum(dim=-1)

                Lse = (
                    tgt_di.unsqueeze(1)
                    * M
                    * token_loss
                ).sum()

                # Eq (14)
                loss = Lcr + Lse + cfg.lambda_adv * Ladv

            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            model.ema_update()

            losses.append(
                loss.item()
            )

        # ---------------- validation ----------------
        do_eval = (
            (epoch + 1) % 2 == 0
            or (epoch + 1) >= cfg.epochs - 2
        )

        if do_eval:

            vp, vr, vf1 = evaluate(
                model,
                src_val
            )

            print(
                f"Epoch {epoch+1} | "
                f"loss={np.mean(losses):.4f} | "
                f"valF1={vf1:.4f}"
            )

            if vf1 > best_val:
                best_val = vf1
                torch.save(
                    model.state_dict(),
                    best_path
                )

        else:

            print(
                f"Epoch {epoch+1} | "
                f"loss={np.mean(losses):.4f}"
            )

    # ====================================
    # load best checkpoint
    model.load_state_dict(
        torch.load(best_path, map_location=cfg.device)
    )

    tp,tr,tf1 = evaluate(
        model,
        tgt_test
    )

    return tp,tr,tf1