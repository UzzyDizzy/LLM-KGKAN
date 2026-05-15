# metrics.py
from config import Config

cfg = Config()


# -----------------------------------------
# decode BIO spans with polarity
# -----------------------------------------

def spans_from_ids(tokens, ids):

    labs = [
        cfg.id2label[x]
        for x in ids
    ]

    res = []

    i = 0

    while i < len(labs):

        tag = labs[i]

        if tag.startswith("B-"):

            pol = tag.split("-")[1]

            j = i + 1

            while (
                j < len(labs)
                and labs[j] == f"I-{pol}"
            ):
                j += 1

            phrase = " ".join(tokens[i:j])

            res.append(
                (i, j-1, phrase, pol)
            )

            i = j

        else:
            i += 1

    return set(res)


# -----------------------------------------
# exact span F1
# -----------------------------------------

def span_f1_score(
    preds,
    golds,
    token_batches
):

    tp = 0
    fp = 0
    fn = 0

    for p,g,toks in zip(
        preds,
        golds,
        token_batches
    ):

        ps = spans_from_ids(
            toks, p
        )

        gs = spans_from_ids(
            toks, g
        )

        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)

    f1 = (
        2 * prec * rec
        / (prec + rec + 1e-9)
    )

    return prec, rec, f1