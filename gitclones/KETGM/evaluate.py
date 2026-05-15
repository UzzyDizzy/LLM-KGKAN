"""
Evaluation — Step 7 of the KETGM pipeline.

Loads the best KETGM checkpoint and computes Micro-F1 on the eval set.
Paper Section IV.
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import classification_report

from config import Config
from utils import ABSADataset, collate_fn
from models import KETGM
from train import _prepare_node_embeds_tc


def run(eval_samples, pos2id, dep2id, kg, all_tc, topic_tc,
        device, model=None, config=None):
    """
    Evaluate the KETGM model on the eval set.

    Parameters
    ----------
    model  : KETGM model (if None, loads from checkpoint)

    Returns
    -------
    dict with keys: f1, all_preds, all_golds, report
    """
    if config is None:
        config = Config

    tokenizer = BertTokenizerFast.from_pretrained(config.BERT_MODEL_NAME)
    eval_ds = ABSADataset(eval_samples, pos2id, dep2id, tokenizer)
    eval_dl = DataLoader(eval_ds, batch_size=config.BATCH_SIZE,
                         shuffle=False, collate_fn=collate_fn)

    # Load model from checkpoint if not provided
    if model is None:
        model = KETGM(
            num_pos_tags=len(pos2id),
            num_dep_tags=len(dep2id),
            rgcn_dim=config.RGCN_HIDDEN,
            bert_dim=config.BERT_HIDDEN,
            num_tags=len(config.TAG2IDX),
            mu=config.MU,
            dropout=config.CLASSIFIER_DROPOUT,
        ).to(device)
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, 'ketgm_best.pt')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'  ✓ Loaded checkpoint: {ckpt_path}')

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
    report = classification_report(all_golds, all_preds)

    print(f'\n═══ Evaluation Results ═══')
    print(f'Micro-F1: {f1:.4f}')
    print(report)

    return dict(f1=f1, all_preds=all_preds, all_golds=all_golds, report=report)
