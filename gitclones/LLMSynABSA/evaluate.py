# evaluate.py — Evaluation with Macro-F1 (paper Section 4.1)
import torch
from config import MAX_LEN, NUM_LABELS, ID_MAP, device
from utils import extract_pairs


def evaluate(model, loader, tokenizer, syntax_cache):
    """
    Evaluate on target domain test set.
    Returns Macro-F1 over (aspect, sentiment) pairs.
    """
    model.eval()
    all_gold_pairs = []
    all_pred_pairs = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            texts = batch["text"]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out, _, _, _ = model(ids, mask, texts, syntax_cache)

            preds = torch.argmax(out, dim=-1)  # (B, L)

            for b in range(ids.size(0)):
                text = texts[b]
                enc = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LEN,
                )
                offsets = enc["offset_mapping"]

                pred_labels = [ID_MAP.get(p.item(), "O") for p in preds[b]]
                pred_pairs = extract_pairs(pred_labels, offsets, text)

                # Gold labels
                if "labels" in batch:
                    gold_labels_raw = batch["labels"][b].tolist()
                    gold_labels = [
                        ID_MAP.get(g, "O") if g != -100 else "O"
                        for g in gold_labels_raw
                    ]
                    gold_pairs = extract_pairs(gold_labels, offsets, text)
                else:
                    gold_pairs = set()

                all_gold_pairs.append(gold_pairs)
                all_pred_pairs.append(pred_pairs)

    # Macro-F1 per sentiment class
    sentiments = ["POS", "NEG", "NEU"]
    f1_scores = []

    for sent in sentiments:
        tp, fp, fn = 0, 0, 0
        for gold, pred in zip(all_gold_pairs, all_pred_pairs):
            gold_s = {(asp, s) for asp, s in gold if s == sent}
            pred_s = {(asp, s) for asp, s in pred if s == sent}
            tp += len(gold_s & pred_s)
            fp += len(pred_s - gold_s)
            fn += len(gold_s - pred_s)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)

    # Also compute overall (micro) stats
    total_tp, total_fp, total_fn = 0, 0, 0
    for gold, pred in zip(all_gold_pairs, all_pred_pairs):
        total_tp += len(gold & pred)
        total_fp += len(pred - gold)
        total_fn += len(gold - pred)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0)

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "precision": micro_p,
        "recall": micro_r,
        "per_class_f1": dict(zip(sentiments, f1_scores)),
    }