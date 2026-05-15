"""
KETGM Full Training — Step 6 of the KETGM pipeline.

Trains the full KETGM model with proper reconstruction loss (Eq 10):
    L = L_softmax + μ * L_recon

Paper-faithful: feat_map/feat_recon are trainable inside KETGM so
reconstruction loss contributes meaningful gradients (Fig 5).
Paper Section III-E / III-F.
"""
import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm.auto import tqdm as tqdm_auto

from config import Config
from utils import ABSADataset, collate_fn
from models import KETGM


def _prepare_node_embeds_tc(batch_tokens, num_words, kg_data,
                            all_tc, device, rgcn_dim):
    """
    Look up raw R-GCN node embeddings (tc) per word in the batch.

    Returns  (B, max_words, rgcn_dim) tensor of tc embeddings.
    """
    B = len(batch_tokens)
    max_w = int(num_words.max().item())
    tc_batch = torch.zeros(B, max_w, rgcn_dim, device=device)
    t2n = kg_data['token2node']
    for b in range(B):
        for w, tok in enumerate(batch_tokens[b]):
            if w >= max_w:
                break
            key = tok.lower().replace(' ', '_')
            if key in t2n:
                tc_batch[b, w] = all_tc[t2n[key]]
    return tc_batch


def run(train_samples, eval_samples, pos2id, dep2id,
        kg, all_tc, topic_tc, device,
        bert_state_dict=None, rgcn_feat_map_state=None,
        rgcn_feat_recon_state=None, config=None):
    """
    Train the full KETGM model.

    Parameters
    ----------
    all_tc          : (N, rgcn_dim)  raw R-GCN node embeddings (frozen)
    topic_tc        : (T, rgcn_dim)  raw R-GCN topic embeddings (frozen)
    bert_state_dict : state_dict from pre-trained BERTSyntactic (optional)
    rgcn_feat_map_state   : state_dict for feat_map from R-GCN (optional)
    rgcn_feat_recon_state : state_dict for feat_recon from R-GCN (optional)

    Returns
    -------
    dict with keys: model, best_f1
    """
    if config is None:
        config = Config

    from seqeval.metrics import f1_score as seq_f1

    tokenizer = BertTokenizerFast.from_pretrained(config.BERT_MODEL_NAME)
    train_ds = ABSADataset(train_samples, pos2id, dep2id, tokenizer)
    eval_ds  = ABSADataset(eval_samples,  pos2id, dep2id, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
    eval_dl  = DataLoader(eval_ds,  batch_size=config.BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn)

    # ── Build KETGM model ─────────────────────────────────────────────
    model = KETGM(
        num_pos_tags=len(pos2id),
        num_dep_tags=len(dep2id),
        rgcn_dim=config.RGCN_HIDDEN,
        bert_dim=config.BERT_HIDDEN,
        num_tags=len(config.TAG2IDX),
        mu=config.MU,
        dropout=config.CLASSIFIER_DROPOUT,
    ).to(device)

    # Initialize feat_map/feat_recon from pre-trained R-GCN
    if rgcn_feat_map_state is not None:
        model.feat_map.load_state_dict(rgcn_feat_map_state)
        print('  ✓ Loaded feat_map weights from pre-trained R-GCN')
    if rgcn_feat_recon_state is not None:
        model.feat_recon.load_state_dict(rgcn_feat_recon_state)
        print('  ✓ Loaded feat_recon weights from pre-trained R-GCN')

    # Initialize BERT from syntactic pre-training
    if bert_state_dict is not None:
        model.bert_enc.load_state_dict(bert_state_dict, strict=False)
        print('  ✓ Loaded BERT weights from syntactic pre-training')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.KETGM_LR,
    )

    print(f'KETGM params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Batches/epoch: {len(train_dl)}')

    # ── Training loop ─────────────────────────────────────────────────
    best_f1 = 0.0
    t0 = time.time()

    for epoch in range(config.NUM_EPOCHS):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        ep_loss = 0
        for batch in tqdm_auto(train_dl, desc=f'Ep {epoch+1:02d}', leave=False):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            wids = batch['word_ids'].to(device)
            bio  = batch['bio_ids'].to(device)
            nw   = batch['num_words'].to(device)
            toks = batch['tokens']

            # Look up raw tc node embeddings per word (frozen)
            with torch.no_grad():
                tc_batch = _prepare_node_embeds_tc(
                    toks, nw, kg, all_tc, device, config.RGCN_HIDDEN)

            # Forward pass — model internally computes feat_map,
            # topic_attn, recon_loss (all trainable)
            logits, loss = model(
                ids, mask, wids, nw,
                node_embeds_tc=tc_batch,
                topic_tc=topic_tc,
                bio_ids=bio,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            ep_loss += loss.item()

        # ── Eval ──────────────────────────────────────────────────────
        model.eval()
        all_preds, all_golds = [], []
        with torch.no_grad():
            for batch in eval_dl:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                wids = batch['word_ids'].to(device)
                bio  = batch['bio_ids']
                nw   = batch['num_words'].to(device)
                toks = batch['tokens']

                tc_batch = _prepare_node_embeds_tc(
                    toks, nw, kg, all_tc, device, config.RGCN_HIDDEN)

                logits, _ = model(
                    ids, mask, wids, nw,
                    node_embeds_tc=tc_batch,
                    topic_tc=topic_tc,
                )
                preds = logits.argmax(dim=-1).cpu()

                for b in range(preds.size(0)):
                    n = nw[b].item()
                    p_tags = [config.IDX2TAG[preds[b, w].item()] for w in range(n)]
                    g_tags = [config.IDX2TAG[bio[b, w].item()]
                              if bio[b, w].item() != -100 else 'O'
                              for w in range(n)]
                    all_preds.append(p_tags)
                    all_golds.append(g_tags)

        f1 = seq_f1(all_golds, all_preds, average='micro')
        print(f'  Epoch {epoch+1:2d}/{config.NUM_EPOCHS}  '
              f'loss={ep_loss/len(train_dl):.4f}  Micro-F1={f1:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR, 'ketgm_best.pt'))
            print(f'    ★ New best')

    print(f'\n✓ Training done  ({time.time()-t0:.0f}s)  Best Micro-F1 = {best_f1:.4f}')

    return dict(model=model, best_f1=best_f1)
