"""
scripts/convert_data.py — Convert unified CSVs to each repo's native format.
"""
import os, sys, json, csv, random
import pandas as pd
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.config import DOMAIN_FILES, DATA_DIR, GITCLONES_DIR, SEED

DOMAIN_LONG = {"L":"laptop","R":"restaurant","D":"device","S":"service",
               "A":"airline","SH":"shoes","W":"water_purifier","U":"university_course","H":"healthcare"}

def _load_csv(domain_key):
    df = pd.read_csv(DOMAIN_FILES[domain_key])
    if "split" not in df.columns:
        n = len(df); idx = list(range(n))
        random.seed(SEED); random.shuffle(idx)
        cut = int(n * 0.8)
        df["split"] = "test"
        df.loc[df.index[idx[:cut]], "split"] = "train"
    return df

def _group_sentences(df):
    """Group rows by sentence into list of dicts with tokens, aspects, BIO tags."""
    import spacy
    try: nlp = spacy.load("en_core_web_sm")
    except: nlp = None
    
    sents = []
    for text, grp in df.groupby("text", sort=False):
        text = str(text)
        if nlp is not None:
            doc = nlp(text)
            tokens = [t.text for t in doc]
            pos_tags = [t.tag_ for t in doc]
            dep_tags = [t.dep_ for t in doc]
        else:
            tokens = text.split()
            pos_tags = ["NN"] * len(tokens)
            dep_tags = ["dep"] * len(tokens)
        aspects = []
        for _, r in grp.iterrows():
            asp = str(r.get("aspect", ""))
            pol = str(r.get("polarity", "neutral")).lower()
            fr = int(r.get("from", 0)); to = int(r.get("to", 0))
            if asp and asp != "NULL" and asp != "nan":
                aspects.append({"aspect": asp, "polarity": pol, "from": fr, "to": to})
        
        # Build BIO tags
        bio = ["O"] * len(tokens)
        pol_map = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}
        char_pos = 0
        char2tok = {}
        for i, tok in enumerate(tokens):
            for c in range(char_pos, char_pos + len(tok)):
                char2tok[c] = i
            char_pos += len(tok) + 1
        
        for a in aspects:
            p = pol_map.get(a["polarity"], "NEU")
            start_tok = char2tok.get(a["from"])
            end_tok = char2tok.get(a["to"] - 1) if a["to"] > 0 else None
            if start_tok is not None:
                bio[start_tok] = f"B-{p}"
                if end_tok is not None:
                    for j in range(start_tok + 1, end_tok + 1):
                        if j < len(bio): bio[j] = f"I-{p}"
        
        split = grp.iloc[0].get("split", "train")
        sents.append({"text": text, "tokens": tokens, "bio": bio, "bio_tags": bio, "pos_tags": pos_tags, "dep_tags": dep_tags, "aspects": aspects, "split": split})
    return sents

# ═══════════════════════════════════════════
# AHF — needs CSV with split column
# ═══════════════════════════════════════════
def convert_for_ahf(domain_key):
    out_dir = os.path.join(GITCLONES_DIR, "AHF", "data")
    os.makedirs(out_dir, exist_ok=True)
    df = _load_csv(domain_key)
    out = os.path.join(out_dir, f"{DOMAIN_LONG[domain_key]}.csv")
    df.to_csv(out, index=False)
    return out

# ═══════════════════════════════════════════
# TransProto — needs .txt: "token1 token2\n" + "B-POS O I-POS\n"
# alternating lines: sentence, then tags
# ═══════════════════════════════════════════
def convert_for_transproto(src_key, tgt_key):
    tp_data = os.path.join(GITCLONES_DIR, "TransProto", "data", "cross_domain")
    os.makedirs(tp_data, exist_ok=True)
    
    for dk, role in [(src_key, "source"), (tgt_key, "target")]:
        sents = _group_sentences(_load_csv(dk))
        for split in ["train", "test"]:
            subset = [s for s in sents if s["split"] == split]
            fname = f"{DOMAIN_LONG[dk]}_{split}.txt"
            with open(os.path.join(tp_data, fname), "w", encoding="utf-8") as f:
                for s in subset:
                    f.write(" ".join(s["tokens"]) + "\n")
                    f.write(" ".join(s["bio"]) + "\n")
    
    # Also write unlabeled target
    tgt_sents = _group_sentences(_load_csv(tgt_key))
    tgt_train = [s for s in tgt_sents if s["split"] == "train"]
    with open(os.path.join(tp_data, f"{DOMAIN_LONG[tgt_key]}_unlabel.txt"), "w", encoding="utf-8") as f:
        for s in tgt_train:
            f.write(" ".join(s["tokens"]) + "\n")
            f.write(" ".join(["O"] * len(s["tokens"])) + "\n")
    return tp_data

# ═══════════════════════════════════════════
# BGCA — needs .txt: "sentence####aspect1=POS, aspect2=NEG"
# ═══════════════════════════════════════════
def convert_for_bgca(src_key, tgt_key):
    bgca_data = os.path.join(GITCLONES_DIR, "BGCA", "data", "cross_domain")
    os.makedirs(bgca_data, exist_ok=True)
    
    pol_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
    
    for dk in [src_key, tgt_key]:
        sents = _group_sentences(_load_csv(dk))
        for split in ["train", "test"]:
            sub = [s for s in sents if s["split"] == split]
            lines = []
            for s in sub:
                word_part = " ".join(s["tokens"])
                label_part = []
                for tok, bio in zip(s["tokens"], s["bio"]):
                    if bio == "O":
                        tag = "O"
                    else:
                        sentiment = bio.split("-")[-1]
                        tag = f"T-{sentiment}"
                    label_part.append(f"{tok}={tag}")
                
                label_str = " ".join(label_part)
                lines.append(f"{word_part}####{label_str}")
            
            fname = f"{DOMAIN_LONG[dk]}_{split}.txt"
            pair_dir = os.path.join(bgca_data, f"{DOMAIN_LONG[src_key]}-{DOMAIN_LONG[tgt_key]}")
            os.makedirs(pair_dir, exist_ok=True)
            with open(os.path.join(pair_dir, fname), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            if split == "test":
                with open(os.path.join(pair_dir, f"{DOMAIN_LONG[dk]}_dev.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
    
    # Target unlabeled
    tgt_df = _load_csv(tgt_key)
    tgt_train = tgt_df[tgt_df["split"] == "train"]
    pair_dir = os.path.join(bgca_data, f"{DOMAIN_LONG[src_key]}-{DOMAIN_LONG[tgt_key]}")
    with open(os.path.join(pair_dir, "target-unlabel.txt"), "w", encoding="utf-8") as f:
        for text in tgt_train["text"].unique():
            f.write(f"{text}####None\n")
    return pair_dir

# ═══════════════════════════════════════════
# DALM — needs: "token1 token2 ... #label#:positive"
# With BIO label per token
# ═══════════════════════════════════════════
def convert_for_dalm(src_key, tgt_key):
    dalm_data = os.path.join(GITCLONES_DIR, "DALM", "GPT2_based", "cross_domain_LM", "process_data")
    pair_dir = os.path.join(dalm_data, f"{DOMAIN_LONG[src_key]}-{DOMAIN_LONG[tgt_key]}")
    os.makedirs(pair_dir, exist_ok=True)
    
    src_sents = _group_sentences(_load_csv(src_key))
    tgt_sents = _group_sentences(_load_csv(tgt_key))
    
    def write_dalm(sents, path, domain_type=None):
        with open(path, "w", encoding="utf-8") as f:
            for s in sents:
                toks = s["tokens"]
                labs = s["bio"]
                if domain_type:
                    toks = [domain_type] + toks
                    labs = ["O"] + labs
                sentence = " ".join(toks)
                label = " ".join(labs)
                f.write(f"{sentence}####{label}\n")
    
    src_train = [s for s in src_sents if s["split"] == "train"]
    tgt_train = [s for s in tgt_sents if s["split"] == "train"]
    
    # Combined with domain prefixes
    with open(os.path.join(pair_dir, "final_train.txt"), "w", encoding="utf-8") as f:
        for s in src_train:
            toks = ["[source]"] + s["tokens"]
            labs = ["O"] + s["bio"]
            f.write(f"{' '.join(toks)}####{' '.join(labs)}\n")
        for s in tgt_train:
            toks = ["[target]"] + s["tokens"]
            labs = ["O"] + s["bio"]
            f.write(f"{' '.join(toks)}####{' '.join(labs)}\n")
            
    # Test data (target domain)
    tgt_test = [s for s in tgt_sents if s["split"] == "test"]
    with open(os.path.join(pair_dir, "test.txt"), "w", encoding="utf-8") as f:
        for s in tgt_test:
            toks = ["[target]"] + s["tokens"]
            labs = ["O"] + s["bio"]
            f.write(f"{' '.join(toks)}####{' '.join(labs)}\n")
    return pair_dir

# ═══════════════════════════════════════════
# KGAN — write preprocessed data in its expected format
# ═══════════════════════════════════════════
def convert_for_kgan(domain_key):
    """Write data in KGAN's expected token/sentiment marker format."""
    kgan_data = os.path.join(GITCLONES_DIR, "KGAN", "dataset")
    os.makedirs(kgan_data, exist_ok=True)
    
    df = _load_csv(domain_key)
    suffix_map = {"positive": "/p", "negative": "/n", "neutral": "/0"}

    def mark_aspect(text, aspect, polarity):
        suffix = suffix_map.get(str(polarity).lower(), "/0")
        tokens = str(text).split()
        aspect_tokens = str(aspect).split()
        if not tokens or not aspect_tokens:
            return None
        low_tokens = [t.lower().strip(".,!?;:'\"`()[]{}") for t in tokens]
        low_aspect = [t.lower().strip(".,!?;:'\"`()[]{}") for t in aspect_tokens]
        start = None
        for i in range(len(tokens) - len(aspect_tokens) + 1):
            if low_tokens[i:i + len(aspect_tokens)] == low_aspect:
                start = i
                break
        if start is None:
            return None
        marked = list(tokens)
        for j in range(start, start + len(aspect_tokens)):
            marked[j] = marked[j] + suffix
        return " ".join(marked)
    
    for split in ["train", "test"]:
        sub_df = df[df["split"] == split] if "split" in df.columns else df
        sub_lines = []
        for _, r in sub_df.iterrows():
            text = str(r["text"])
            aspect = str(r.get("aspect", ""))
            if aspect and aspect != "NULL" and aspect != "nan":
                line = mark_aspect(text, aspect, r.get("polarity", "neutral"))
                if line:
                    sub_lines.append(line)
        
        out_dir = os.path.join(kgan_data, DOMAIN_LONG[domain_key])
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sub_lines))
    return kgan_data

# ═══════════════════════════════════════════
# SenticGCN — needs .raw: 3 lines per sample
# line1: sentence, line2: aspect term, line3: polarity (0/1/2 = neg/neu/pos)
# ═══════════════════════════════════════════
def convert_for_senticgcn(domain_key):
    sgcn_data = os.path.join(GITCLONES_DIR, "Sentic-GCN", "datasets", "custom")
    os.makedirs(sgcn_data, exist_ok=True)
    
    df = _load_csv(domain_key)
    # SenticGCN polarity: -1=neg, 0=neu, 1=pos
    pol_map = {"positive": 1, "negative": -1, "neutral": 0}
    
    for split in ["train", "test"]:
        sub = df[df["split"] == split] if "split" in df.columns else df
        fname = f"{DOMAIN_LONG[domain_key]}_{split}.raw"
        with open(os.path.join(sgcn_data, fname), "w", encoding="utf-8") as f:
            for _, r in sub.iterrows():
                text = str(r["text"]).strip()
                aspect = str(r.get("aspect", "")).strip()
                pol = pol_map.get(str(r.get("polarity", "")).lower(), 0)
                if aspect and aspect != "NULL" and aspect != "nan":
                    # Mark aspect position with $T$
                    marked = text.replace(aspect, "$T$", 1)
                    f.write(marked + "\n")
                    f.write(aspect + "\n")
                    f.write(str(pol) + "\n")
    return sgcn_data

# ═══════════════════════════════════════════
# KETGM — needs its own data structure
# ═══════════════════════════════════════════
def convert_for_ketgm(src_key, tgt_key):
    ketgm_data = os.path.join(GITCLONES_DIR, "KETGM", "data")
    os.makedirs(ketgm_data, exist_ok=True)
    
    for dk in [src_key, tgt_key]:
        sents = _group_sentences(_load_csv(dk))
        out = []
        for s in sents:
            out.append({
                "tokens": s["tokens"],
                "bio_tags": s["bio_tags"],
                "pos_tags": s["pos_tags"],
                "dep_tags": s["dep_tags"],
                "aspects": s["aspects"],
                "split": s["split"]
            })
        with open(os.path.join(ketgm_data, f"{DOMAIN_LONG[dk]}.json"), "w") as f:
            json.dump(out, f, indent=1)
    return ketgm_data

# ═══════════════════════════════════════════
# Master converter
# ═══════════════════════════════════════════
_converted = set()

def ensure_converted(model_name, src, tgt):
    """Ensure data is converted for the given model+pair. Idempotent."""
    key = f"{model_name}_{src}_{tgt}"
    if key in _converted:
        return
    
    if model_name == "ahf":
        convert_for_ahf(src); convert_for_ahf(tgt)
    elif model_name == "transproto":
        convert_for_transproto(src, tgt)
    elif model_name == "bgca":
        convert_for_bgca(src, tgt)
    elif model_name == "dalm":
        convert_for_dalm(src, tgt)
    elif model_name == "kgan":
        convert_for_kgan(src); convert_for_kgan(tgt)
    elif model_name == "senticgcn":
        convert_for_senticgcn(src); convert_for_senticgcn(tgt)
    elif model_name == "ketgm":
        convert_for_ketgm(src, tgt)
    
    _converted.add(key)
    print(f"[DATA] Converted for {model_name} {src}->{tgt}")

def convert_all(pairs):
    """Pre-convert data for all models and pairs."""
    for model in ["ahf","transproto","bgca","dalm","kgan","senticgcn","ketgm"]:
        for s, t in pairs:
            try:
                ensure_converted(model, s, t)
            except Exception as e:
                print(f"[WARN] convert {model} {s}->{t}: {e}")

if __name__ == "__main__":
    from scripts.config import STANDARD_PAIRS
    convert_all(STANDARD_PAIRS)
    print("Done")
