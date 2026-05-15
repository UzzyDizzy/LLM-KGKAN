"""
Inference — Optional step for the KETGM pipeline.

Runs the trained KETGM model on samples and prints
per-token aspect + sentiment predictions.
"""
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from config import Config
from utils import ABSADataset, collate_fn
from train import _prepare_node_embeds_tc


def run(model, test_samples, pos2id, dep2id, kg, all_tc, topic_tc,
        device, num_display=5, config=None):
    """
    Run inference on test samples and display predictions.

    Parameters
    ----------
    num_display : int — number of sample predictions to print

    Returns
    -------
    dict with keys: all_preds, all_texts
    """
    if config is None:
        config = Config

    tokenizer = BertTokenizerFast.from_pretrained(config.BERT_MODEL_NAME)
    test_ds = ABSADataset(test_samples, pos2id, dep2id, tokenizer)
    test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                         shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_preds = []
    all_texts = []

    with torch.no_grad():
        for batch in test_dl:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            wids = batch['word_ids'].to(device)
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
                words = toks[b]
                tags = [config.IDX2TAG[preds[b, w].item()] for w in range(n)]
                all_preds.append(tags)
                all_texts.append(words)

    # Display sample predictions
    for i in range(min(num_display, len(all_texts))):
        print("\nSentence:")
        print(" ".join(all_texts[i]))
        print("Predictions:")
        for w, t in zip(all_texts[i], all_preds[i]):
            print(f"{w:15s} -> {t}")

    return dict(all_preds=all_preds, all_texts=all_texts)
