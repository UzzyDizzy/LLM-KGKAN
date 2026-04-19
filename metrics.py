# metrics.py

from sklearn.metrics import f1_score
import torch


def compute_macro_f1(logits, labels, ignore_index=-100):
    preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()

    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]

    return f1_score(labels, preds, average="macro")

def extract_triplets(aspect_preds, opinion_preds, sentiment_preds):
    triplets = []

    T = aspect_preds.shape[0]

    for i in range(T):
        if aspect_preds[i] == 1:
            for j in range(T):
                if opinion_preds[j] == 1:
                    sentiment = sentiment_preds[i][j]
                    triplets.append((i, j, sentiment))

    return set(triplets)

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import f1_score

def compute_bio_f1(logits, labels):
    preds = logits.argmax(-1).view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()

    mask = labels != -100
    return f1_score(labels[mask], preds[mask], average="macro")