"""
BERT Syntactic Pre-training — Step 5 of the KETGM pipeline.

Pre-trains BERT encoder with POS classification and DEP classification
auxiliary tasks (Section III-D, following Gong et al. 2020).

This step was MISSING in the original notebook — the paper requires
BERT to be pre-trained on syntactic tasks before full KETGM training.
"""
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm.auto import tqdm as tqdm_auto

from config import Config
from utils import ABSADataset, collate_fn
from models import BERTSyntactic


def run(train_samples, eval_samples, pos2id, dep2id, device, config=None):
    """
    Pre-train BERT with POS and DEP classification tasks.

    Returns
    -------
    dict with keys:
        bert_syn  — pre-trained BERTSyntactic model
        tokenizer — BertTokenizerFast
    """
    if config is None:
        config = Config

    tokenizer = BertTokenizerFast.from_pretrained(config.BERT_MODEL_NAME)
    train_ds = ABSADataset(train_samples, pos2id, dep2id, tokenizer)
    eval_ds  = ABSADataset(eval_samples,  pos2id, dep2id, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
    eval_dl  = DataLoader(eval_ds,  batch_size=config.BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn)

    bert_syn = BERTSyntactic(num_pos_tags=len(pos2id),
                             num_dep_tags=len(dep2id)).to(device)
    print(f'BERT params: {sum(p.numel() for p in bert_syn.parameters()):,}')
    print(f'Batches/epoch: {len(train_dl)}')

    optimizer = torch.optim.Adam(bert_syn.parameters(), lr=config.BERT_LR)

    # ── Pre-training loop (POS + DEP tasks) ───────────────────────────
    t0 = time.time()
    best_loss = float('inf')

    for epoch in range(config.BERT_PRETRAIN_EPOCHS):
        bert_syn.train()
        ep_loss = 0.0
        for batch in tqdm_auto(train_dl, desc=f'BERT-Syn Ep {epoch+1:02d}', leave=False):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            wids = batch['word_ids'].to(device)
            pos  = batch['pos_ids'].to(device)
            dep  = batch['dep_ids'].to(device)
            nw   = batch['num_words'].to(device)

            hidden, pos_logits, dep_logits = bert_syn(ids, mask)

            # Gather first-subword logits for per-word POS/DEP classification
            B, S, _ = pos_logits.shape
            max_w = nw.max().item()

            pos_pred = torch.zeros(B, max_w, pos_logits.size(-1), device=device)
            dep_pred = torch.zeros(B, max_w, dep_logits.size(-1), device=device)

            for b in range(B):
                seen = set()
                for s_idx in range(S):
                    wid = wids[b, s_idx].item()
                    if wid < 0 or wid >= max_w or wid in seen:
                        continue
                    pos_pred[b, wid] = pos_logits[b, s_idx]
                    dep_pred[b, wid] = dep_logits[b, s_idx]
                    seen.add(wid)

            # POS classification loss
            pos_loss = F.cross_entropy(
                pos_pred.reshape(-1, pos_pred.size(-1)),
                pos[:, :max_w].reshape(-1),
                ignore_index=-100
            )

            # DEP classification loss
            dep_loss = F.cross_entropy(
                dep_pred.reshape(-1, dep_pred.size(-1)),
                dep[:, :max_w].reshape(-1),
                ignore_index=-100
            )

            loss = pos_loss + dep_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_syn.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            ep_loss += loss.item()

        avg_loss = ep_loss / len(train_dl)
        print(f'  BERT-Syn Epoch {epoch+1}/{config.BERT_PRETRAIN_EPOCHS}  loss={avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(bert_syn.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR, 'bert_syntactic.pt'))
            print(f'    ★ New best')

    elapsed = time.time() - t0
    print(f'\n✓ BERT syntactic pre-training done  ({elapsed:.1f}s)')

    return dict(bert_syn=bert_syn, tokenizer=tokenizer)
