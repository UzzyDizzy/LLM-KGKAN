"""
scripts/evaluate.py — Unified evaluation for all models.
"""
import os, json, time
import numpy as np
import torch
from sklearn.metrics import f1_score
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    LABELS, LABEL2ID, ID2LABEL, NUM_LABELS, IGNORE_INDEX,
    get_result_path, MODEL_DISPLAY_NAMES,
)

def compute_macro_f1(preds, labels, ignore_index=IGNORE_INDEX):
    if preds.dim() == 3:
        preds = preds.argmax(dim=-1)
    pf = preds.reshape(-1).cpu().numpy()
    lf = labels.reshape(-1).cpu().numpy()
    mask = lf != ignore_index
    pf, lf = pf[mask], lf[mask]
    if len(lf) == 0:
        return 0.0
    return float(f1_score(lf, pf, average="macro", zero_division=0)) * 100

def save_result(model_name, setting, src, tgt, macro_f1, extra=None):
    path = get_result_path(model_name, setting, src, tgt)
    result = {"model": model_name, "setting": setting, "src": src, "tgt": tgt,
              "macro_f1": round(macro_f1, 2), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        result.update(extra)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, path)
    return path

def load_result(model_name, setting, src, tgt):
    path = get_result_path(model_name, setting, src, tgt)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None

def load_all_results(setting):
    from scripts.config import RESULTS_DIR
    d = os.path.join(RESULTS_DIR, setting)
    all_r = defaultdict(dict)
    if not os.path.exists(d):
        return all_r
    for fn in os.listdir(d):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                r = json.load(f)
            all_r[r["model"]][f"{r['src']}->{r['tgt']}"] = r["macro_f1"]
    return dict(all_r)

def build_table_df(models, pairs, setting):
    import pandas as pd
    rows = []
    for model in models:
        display = MODEL_DISPLAY_NAMES.get(model, model)
        row = {"Model": display}
        vals = []
        for src, tgt in pairs:
            r = load_result(model, setting, src, tgt)
            if r is not None:
                row[f"{src}→{tgt}"] = r["macro_f1"]
                vals.append(r["macro_f1"])
            else:
                row[f"{src}→{tgt}"] = None
        row["AVG"] = round(sum(vals)/len(vals), 2) if len(vals) == len(pairs) else None
        rows.append(row)
    return pd.DataFrame(rows)
