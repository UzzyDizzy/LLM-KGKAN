# utils.py
import numpy as np
import torch

def pad(seq, maxlen, pad_value=0):
    seq = seq[:maxlen]
    return seq + [pad_value]*(maxlen-len(seq))

def threshold_from_probs(confs, beta, rho):
    if len(confs)==0:
        return rho
    confs = sorted(confs, reverse=True)
    k = max(1, int(len(confs)*beta/100))
    q = confs[k-1]
    return max(rho, q)

def spans_from_labels(tokens, labels):
    res=[]
    i=0
    while i<len(labels):
        tag=labels[i]
        if tag.startswith("B-"):
            pol=tag.split("-")[1]
            j=i+1
            while j<len(labels) and labels[j]==f"I-{pol}":
                j+=1
            aspect=" ".join(tokens[i:j])
            res.append((i,j-1,aspect,pol))
            i=j
        else:
            i+=1
    return res

def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
