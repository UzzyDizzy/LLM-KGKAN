"""
train_all.py — Native training orchestrator with data conversion.
"""
import os, sys, json, subprocess, traceback, random, gc, argparse, time, re, contextlib
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.config import *
from scripts.evaluate import compute_macro_f1, save_result, load_result
from scripts.convert_data import ensure_converted

os.environ["TOKENIZERS_PARALLELISM"]="false"

DL = {"L":"laptop","R":"restaurant","D":"device","S":"service",
      "A":"airline","SH":"shoes","W":"water_purifier","U":"university_course","H":"healthcare"}

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _done(n,st,s,t): return load_result(n,st,s,t) is not None

def _parse_f1(stdout):
    for line in reversed(stdout.split('\n')):
        m = re.search(r'[Ff]1[:\s=]+([0-9.]+)', line)
        if m:
            v = float(m.group(1))
            return v * 100 if v < 1 else v
        
        # BGCA format: "epoch-15[best]    61.82/64.00/62.89/"
        m_bgca = re.search(r'\[best\]\s+[0-9.]+\/[0-9.]+\/([0-9.]+)\/', line)
        if m_bgca:
            v = float(m_bgca.group(1))
            return v * 100 if v < 1 else v
            
        m_dict = re.search(r"'f1': ([0-9.]+)", line)
        if m_dict:
            v = float(m_dict.group(1))
            return v * 100 if v < 1 else v
            
    return None

import subprocess
import threading
import queue
import time
import sys

def run_and_stream(cmd, cwd, timeout=None):
    print(f"\n>>> Executing: {' '.join(cmd)}\n")

    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    q = queue.Queue()

    def enqueue_output(pipe, q):
        try:
            for line in iter(pipe.readline, ''):
                q.put(line)
        finally:
            pipe.close()

    t = threading.Thread(
        target=enqueue_output,
        args=(p.stdout, q),
        daemon=True
    )
    t.start()

    output = []
    start = time.time()

    while True:

        if timeout and time.time()-start > timeout:
            p.kill()
            print(f"\n[TIMEOUT after {timeout}s]")
            break

        try:
            line = q.get(timeout=0.2)

            sys.stdout.write(line)
            sys.stdout.flush()

            output.append(line)

        except queue.Empty:
            if p.poll() is not None:
                break

    p.wait()

    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")

    return ''.join(output)


def _ensure_pytorch_pretrained_bert(tag):
    try:
        import pytorch_pretrained_bert
    except ImportError:
        print(f"[{tag}] Installing missing legacy dependency: pytorch-pretrained-bert")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-pretrained-bert"])


@contextlib.contextmanager
def _temp_env(path):
    old_cwd = os.getcwd()
    sys.path.insert(0, path)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if path in sys.path:
            sys.path.remove(path)
        for m in ["config", "model", "train", "data", "utils", "metrics", "models", "main"]:
            if m in sys.modules:
                del sys.modules[m]

# ═══════════════════════════════════════════
# AHF — BiLSTM (gitclones/AHF/)
# ═══════════════════════════════════════════
def train_ahf(src, tgt, setting="standard", k_shot=None):
    if _done("ahf",setting,src,tgt): return
    ensure_converted("ahf", src, tgt)
    ahf = os.path.join(GITCLONES_DIR, "AHF")
    with _temp_env(ahf):
        set_seed()
        try:
            from data import load_transfer_pair
            from train import train_one_pair
            st, sv, tt, te, w2id, p2id, emb = load_transfer_pair(DL[src], DL[tgt])
            if k_shot and k_shot > 0: tt = tt[:k_shot*3]
            tp, tr, tf1 = train_one_pair(st, sv, tt, te, w2id, p2id, emb, f"{src}_{tgt}")
            save_result("ahf", setting, src, tgt, tf1 * 100)
            print(f"[OK] ahf {src}->{tgt}: F1={tf1*100:.2f}")
        except Exception as e:
            print(f"[ERR] ahf {src}->{tgt}: {e}"); traceback.print_exc()
    gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════════
# TransProto — subprocess
# ═══════════════════════════════════════════
def train_transproto(src, tgt, setting="standard", k_shot=None):
    if _done("transproto",setting,src,tgt): return
    ensure_converted("transproto", src, tgt)
    tp = os.path.join(GITCLONES_DIR, "TransProto")
    data_dir = os.path.join(tp, "data", "cross_domain")
    out = os.path.join(SAVED_MODELS_DIR, "transproto", f"{src}_{tgt}")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(tp, "train_bert_bridge.py"),
           "--source", DL[src], "--target", DL[tgt],
           "--bert_type", "bert-base-uncased",
           "--seed", str(SEED), "--num_epoch", "30", "--batch_size", "32"]
    try:
        # Check for legacy dependency used by TransProto
        _ensure_pytorch_pretrained_bert("TRANSPROTO")
        
        out = run_and_stream(cmd, cwd=tp, timeout=7200)
        f1 = _parse_f1(out)
        if f1:
            save_result("transproto", setting, src, tgt, f1)
            print(f"[OK] transproto {src}->{tgt}: F1={f1:.2f}")
        else:
            print(f"[ERR] transproto {src}->{tgt}: No F1 found in logs")
    except Exception as e:
        print(f"[ERR] transproto {src}->{tgt}: {e}")

# ═══════════════════════════════════════════
# BGCA — T5 subprocess
# ═══════════════════════════════════════════
def train_bgca(src, tgt, setting="standard", k_shot=None):
    if _done("bgca",setting,src,tgt): return
    ensure_converted("bgca", src, tgt)
    bgca = os.path.join(GITCLONES_DIR, "BGCA", "code")
    pair_dir = os.path.join(GITCLONES_DIR, "BGCA", "data", "cross_domain", f"{DL[src]}-{DL[tgt]}")
    cmd = [sys.executable, os.path.join(bgca, "main.py"),
           "--task", "uabsa", "--dataset", pair_dir,
           "--paradigm", "extraction-universal", "--model_name_or_path", "t5-base",
           "--do_train", "--do_eval", "--train_by_pair",
           "--num_train_epochs", "20", "--train_batch_size", "16",
           "--learning_rate", "3e-4", "--seed", str(SEED), "--n_runs", "1"]
    try:
        try:
            import editdistance
        except ImportError:
            print("[BGCA] Installing missing dependency: editdistance")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "editdistance"])
            
        out = run_and_stream(cmd, cwd=bgca, timeout=7200)
        f1 = _parse_f1(out)
        if f1:
            save_result("bgca", setting, src, tgt, f1)
            print(f"[OK] bgca {src}->{tgt}: F1={f1:.2f}")
        else:
            print(f"[ERR] bgca {src}->{tgt}: No F1 found in logs")
    except Exception as e:
        print(f"[ERR] bgca {src}->{tgt}: {e}")

# ═══════════════════════════════════════════
# KETGM — BERT + R-GCN
# ═══════════════════════════════════════════
def train_ketgm(src, tgt, setting="standard", k_shot=None):
    if _done("ketgm",setting,src,tgt): return
    ensure_converted("ketgm", src, tgt)
    kd = os.path.join(GITCLONES_DIR, "KETGM")
    with _temp_env(kd):
        set_seed()
        try:
            from config import Config as KC
            from utils import ABSADataset as KDS, collate_fn as kcf
            from models import KETGM as KM
            from train import run as krun
            from torch.utils.data import DataLoader
            from transformers import BertTokenizerFast

            cfg = KC(); cfg.source_domain = DL[src]; cfg.target_domain = DL[tgt]
            data_path = os.path.join(kd, "data")
            device = torch.device(DEVICE)
            tok = BertTokenizerFast.from_pretrained(cfg.bert_model if hasattr(cfg,'bert_model') else "bert-base-uncased")
            
            # Load converted JSON data
            import json as jj
            src_data = jj.load(open(os.path.join(data_path, f"{DL[src]}.json")))
            tgt_data = jj.load(open(os.path.join(data_path, f"{DL[tgt]}.json")))
            train_s = [s for s in src_data if s["split"]=="train"]
            test_t = [s for s in tgt_data if s["split"]=="test"]
            
            from utils import build_tag_vocabs
            pos2id, dep2id = build_tag_vocabs(train_s + test_t)
            if not pos2id: pos2id = {"NN":0}
            if not dep2id: dep2id = {"dep":0}
            
            from topic_extraction import extract_topics
            from knowledge_graph import run as kg_run
            from pretrain_rgcn import run as rgcn_run
            import spacy
            
            cfg.CONCEPTNET_CSV_GZ = os.path.join(ROOT, "knowledge_graphs", "conceptnet-assertions-5.7.0.csv")
            cfg.CONCEPTNET_EN_PKL = os.path.join(ROOT, "knowledge_graphs", "conceptnet_en_ketgm.pkl")
            
            print(f"[KETGM] Extracting topics for {src}->{tgt}...")
            texts = [s.get("text", " ".join(s["tokens"])) for s in train_s + test_t]
            topic_words = extract_topics(texts, num_topics=cfg.NUM_TOPICS, words_per_topic=cfg.WORDS_PER_TOPIC)
            
            print(f"[KETGM] Building knowledge graph...")
            try: nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            except: 
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            
            kg_out = kg_run(train_s, test_t, topic_words, nlp, config=cfg)
            kg_data = kg_out["kg"]
            
            print(f"[KETGM] Pre-training R-GCN...")
            rgcn_out = rgcn_run(kg_data, device, config=cfg)
            
            result = krun(train_s, test_t, pos2id, dep2id, kg_data, 
                          rgcn_out["all_tc"], rgcn_out["topic_tc"], device, config=cfg)
            f1 = result if isinstance(result,(int,float)) else result.get("best_f1", result.get("f1",0))
            if f1 < 1: f1 *= 100
            save_result("ketgm", setting, src, tgt, f1)
            print(f"[OK] ketgm {src}->{tgt}: F1={f1:.2f}")
        except Exception as e:
            print(f"[ERR] ketgm {src}->{tgt}: {e}"); traceback.print_exc()
    gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════════
# DALM — GPT-2
# ═══════════════════════════════════════════
def train_dalm(src, tgt, setting="standard", k_shot=None):
    cached = load_result("dalm", setting, src, tgt)
    if cached is not None:
        cached_f1 = float(cached.get("macro_f1", 0.0))
        if cached_f1 > 0.0:
            print(f"[SKIP] dalm {src}->{tgt}: F1={cached_f1:.2f}")
            return
        print(f"[RERUN] dalm {src}->{tgt}: cached F1={cached_f1:.2f}; rerunning after DALM fixes")
    ensure_converted("dalm", src, tgt)
    dalm_root = os.path.join(GITCLONES_DIR, "DALM")
    dalm_lm = os.path.join(dalm_root, "GPT2_based", "cross_domain_LM")
    pair_dir = os.path.join(dalm_lm, "process_data", f"{DL[src]}-{DL[tgt]}")
    
    # DALM expects "rest" instead of "restaurant" for its native dataset files
    dalm_src = "rest" if DL[src] == "restaurant" else DL[src]
    dalm_tgt = "rest" if DL[tgt] == "restaurant" else DL[tgt]
    
    # 1. Train cross-domain LM
    cmd1 = [sys.executable, os.path.join(dalm_lm, "train.py"),
           "--input_dir", os.path.join(pair_dir, "final_train.txt"),
           "--model_dir", os.path.join(dalm_root, "GPT2_based", "models"),
           "--source_domain", dalm_src, "--target_domain", dalm_tgt,
           "--seed", str(SEED), "--max_seq_length", "100",
           "--train_batch_size", "32", "--learning_rate", "3e-4",
           "--num_train_epochs", "20"]
    
    # 2. Generate pseudo labels
    cmd2 = [sys.executable, os.path.join(dalm_lm, "generate.py"),
            "--target", dalm_tgt, "--source", dalm_src, "--generate_number", "10000"]
    # 2.5 Train source-only model to filter pseudo labels
    cmd_pseudo = [sys.executable, os.path.join(dalm_root, "absa", "pseudo_labeling.py"), "--task", "absa",
                  "--domain_pair", f"{dalm_src}-{dalm_tgt}", "--model_name_or_path", "bert-base-uncased",
                  "--output_dir", os.path.join(dalm_root, "pseudo_outputs"),
                  "--do_train", "--do_pseudo_labeling"]
    
    # 3. Filter pseudo labels
    cmd3 = [sys.executable, os.path.join(dalm_root, "absa", "filter.py"), "--task", "absa",
            "--domain_pair", f"{dalm_src}-{dalm_tgt}", "--model_name_or_path", "bert-base-uncased",
            "--do_filter", "--output_dir", os.path.join(dalm_root, "GPT2_based", "generated_data")]
            
    # 4. Train ABSA model
    cmd4 = [sys.executable, os.path.join(dalm_root, "absa", "main.py"), "--task", "absa",
            "--domain_pair", f"{dalm_src}-{dalm_tgt}", "--model_name_or_path", "bert-base-uncased",
            "--data_path", os.path.join(dalm_root, "GPT2_based", "generated_data", f"{dalm_src}-{dalm_tgt}", "filter.txt"),
            "--output_dir", os.path.join(dalm_root, "GPT2_based", "main_outputs"),
            "--do_train", "--do_eval"]

    try:
        run_and_stream(cmd1, cwd=dalm_root, timeout=7200)
        run_and_stream(cmd2, cwd=dalm_root, timeout=7200)
        run_and_stream(cmd_pseudo, cwd=dalm_root, timeout=7200)
        run_and_stream(cmd3, cwd=dalm_root, timeout=7200)
        filter_path = os.path.join(dalm_root, "GPT2_based", "generated_data", f"{dalm_src}-{dalm_tgt}", "filter.txt")
        if not os.path.exists(filter_path) or os.path.getsize(filter_path) == 0:
            raise RuntimeError(f"DALM filter produced no training samples: {filter_path}")
        out = run_and_stream(cmd4, cwd=dalm_root, timeout=7200)
        
        f1 = _parse_f1(out)
        if f1 is not None:
            save_result("dalm", setting, src, tgt, f1)
            print(f"[OK] dalm {src}->{tgt}: F1={f1:.2f}")
        else:
            print(f"[ERR] dalm {src}->{tgt}: No F1 found in logs")
    except Exception as e:
        print(f"[ERR] dalm {src}->{tgt}: {e}")

# ═══════════════════════════════════════════
# LLMSynABSA — LLaMA-3
# ═══════════════════════════════════════════
def train_llmsynabsa(src, tgt, setting="standard", k_shot=None):
    if _done("llmsynabsa",setting,src,tgt): return
    lsa = os.path.join(GITCLONES_DIR, "LLMSynABSA")
    with _temp_env(lsa):
        set_seed()
        try:
            from model import LLMSynABSA as LSA
            from transformers import AutoTokenizer
            device = torch.device(DEVICE)
            tok_args = {"token": HF_TOKEN} if HF_TOKEN else {}
            tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", **tok_args)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            model = LSA("meta-llama/Meta-Llama-3-8B-Instruct").to(device)
            for n, p in model.named_parameters():
                if p.device != device: p.data = p.data.to(device)
            # Use our unified data loading
            from scripts.data_utils import build_dataloaders
            from functools import partial
            from scripts.data_utils import collate_for_bert
            collate = partial(collate_for_bert, tokenizer=tok, max_len=96)
            loaders = build_dataloaders(src, tgt, tok, 96, 8, k_shot=k_shot, collate_fn=collate)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
            best = 0
            for ep in range(10):
                model.train()
                for i, batch in enumerate(loaders["src_loader"]):
                    opt.zero_grad()
                    ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    texts = batch["texts"]
                    out = model(ids, mask, texts, {}); logits = out if isinstance(out, torch.Tensor) else out[0]
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, NUM_LABELS), labels.view(-1), ignore_index=IGNORE_INDEX)
                    loss.backward(); opt.step()
                    if i % 10 == 0: print(f"    Batch {i} Loss: {loss.item():.4f}")
                model.eval(); ap, al = [], []
                with torch.no_grad():
                    for b in loaders["tgt_val_loader"]:
                        ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
                        texts = b["texts"]
                        out = model(ids, mask, texts, {}); lg = out if isinstance(out, torch.Tensor) else out[0]
                        ap.append(lg.cpu()); al.append(b["labels"])
                if ap:
                    f1 = compute_macro_f1(torch.cat(ap), torch.cat(al))
                    if f1 > best: best = f1
                    print(f"  Ep {ep+1}: F1={f1:.2f}")
            save_result("llmsynabsa", setting, src, tgt, best)
            print(f"[OK] llmsynabsa {src}->{tgt}: F1={best:.2f}")
        except Exception as e:
            print(f"[ERR] llmsynabsa {src}->{tgt}: {e}"); traceback.print_exc()
    gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════════
# KGAN — subprocess
# ═══════════════════════════════════════════
def train_kgan(src, tgt, setting="standard", k_shot=None):
    if _done("kgan",setting,src,tgt): return
    ensure_converted("kgan", src, tgt)
    kgan = os.path.join(GITCLONES_DIR, "KGAN")
    # KGAN needs cross-domain: train on src, test on tgt
    cmd = [sys.executable, os.path.join(kgan, "main_total.py"),
           "-model", "KGNN", "-ds_name", DL[src],
           "-is_bert", "2", "-bs", "32", "-n_epoch", "20",
           "-learning_rate", "0.00003"]
    try:
        _ensure_pytorch_pretrained_bert("KGAN")

        try:
            import spacy
            spacy.load("en_core_web_lg")
        except Exception:
            print("[KGAN] Installing missing spaCy model: en_core_web_lg")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
            
        out = run_and_stream(cmd, cwd=kgan, timeout=7200)
        f1 = _parse_f1(out)
        if f1:
            save_result("kgan", setting, src, tgt, f1)
            print(f"[OK] kgan {src}->{tgt}: F1={f1:.2f}")
        else:
            print(f"[ERR] kgan {src}->{tgt}: No F1 found in logs")
    except Exception as e:
        print(f"[ERR] kgan {src}->{tgt}: {e}")

# ═══════════════════════════════════════════
# SenticGCN — subprocess
# ═══════════════════════════════════════════
def train_senticgcn(src, tgt, setting="standard", k_shot=None):
    if _done("senticgcn",setting,src,tgt): return
    ensure_converted("senticgcn", src, tgt)
    sgcn = os.path.join(GITCLONES_DIR, "Sentic-GCN")
    ds_dir = os.path.join(sgcn, "datasets", "custom")
    # Train on src, test on tgt
    cmd = [sys.executable, os.path.join(sgcn, "train_bert.py"),
           "--model_name", "senticgcn_bert",
           "--dataset", f"custom/{DL[src]}",
           "--seed", str(SEED), "--num_epoch", "30", "--batch_size", "16",
           "--lr", "2e-5", "--pretrained_bert_name", "bert-base-uncased"]
    try:
        _ensure_pytorch_pretrained_bert("SenticGCN")

        try:
            import ipdb
        except ImportError:
            print("[SenticGCN] Installing missing dependency: ipdb")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ipdb"])
            
        out = run_and_stream(cmd, cwd=sgcn, timeout=3600)
        f1 = _parse_f1(out)
        if f1:
            save_result("senticgcn", setting, src, tgt, f1)
            print(f"[OK] senticgcn {src}->{tgt}: F1={f1:.2f}")
        else:
            print(f"[ERR] senticgcn {src}->{tgt}: No F1 found in logs")
    except Exception as e:
        print(f"[ERR] senticgcn {src}->{tgt}: {e}")

# ═══════════════════════════════════════════
# BERT-UDA — inline BERT baseline
# ═══════════════════════════════════════════
def train_bert_uda(src, tgt, setting="standard", k_shot=None):
    if _done("bert_uda",setting,src,tgt): return
    set_seed()
    import torch.nn as nn, torch.nn.functional as F
    from transformers import BertModel, BertTokenizerFast
    from scripts.data_utils import build_dataloaders, collate_for_bert
    from functools import partial
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    collate = partial(collate_for_bert, tokenizer=tok, max_len=128)
    loaders = build_dataloaders(src, tgt, tok, 128, 32, k_shot=k_shot, collate_fn=collate)
    dev = torch.device(DEVICE)

    class Labeler(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.head = nn.Linear(768, NUM_LABELS); self.drop = nn.Dropout(0.1)
        def forward(self, ids, mask, labels=None):
            h = self.drop(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state)
            logits = self.head(h)
            loss = F.cross_entropy(logits.view(-1, NUM_LABELS), labels.view(-1),
                                   ignore_index=IGNORE_INDEX) if labels is not None else None
            return loss, logits

    model = Labeler().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best = 0
    for ep in range(15):
        model.train()
        for batch in loaders["src_loader"]:
            opt.zero_grad()
            loss, _ = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev),
                           batch["labels"].to(dev))
            loss.backward(); opt.step()
        model.eval(); ap, al = [], []
        with torch.no_grad():
            for b in loaders["tgt_val_loader"]:
                _, lg = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
                ap.append(lg.cpu()); al.append(b["labels"])
        if ap:
            f1 = compute_macro_f1(torch.cat(ap), torch.cat(al))
            if f1 > best: best = f1
    save_result("bert_uda", setting, src, tgt, best)
    print(f"[OK] bert_uda {src}->{tgt}: F1={best:.2f}")
    del model; gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════════
# LLM-KGKAN — main model
# ═══════════════════════════════════════════
def train_llm_kgkan(src, tgt, setting="standard", k_shot=None, kg=None, variant="full"):
    display = f"llm_kgkan{'_'+variant if variant!='full' else ''}"
    if _done(display, setting, src, tgt): return
    set_seed()
    try:
        sys.path.insert(0, ROOT)
        from config import LLMKGKANConfig
        from model import LLMKGKAN
        from data import ABSADataset as DS, collate_fn as cf
        from torch.utils.data import DataLoader
        import torch.nn as nn
        from itertools import cycle
        from sampling import few_shot_sample
        dev = torch.device(DEVICE)
        cfg = LLMKGKANConfig()
        if kg:
            cfg.num_entities = len(kg.ent2id)
            cfg.num_kg_relations = len(kg.rel2id)
        model = LLMKGKAN(cfg)
        model.syntax.to(dev)
        model.kg.to(dev)
        model.fusion.to(dev)
        model.arg.to(dev)
        model.dropout.to(dev)
        model.classifier.to(dev)
        model.semantic.proj.to(dev)
        model.semantic.kg_prefix_mlp.to(dev)
        model.semantic.task_prefix.data = model.semantic.task_prefix.data.to(dev)
        if variant == "wo_kg":
            for p in model.kg.parameters(): p.requires_grad_(False); p.zero_()
        elif variant == "wo_syn":
            for p in model.syntax.parameters(): p.requires_grad_(False); p.zero_()
        elif variant == "wo_arg":
            class IdentityARG(nn.Module):
                def forward(self, z, sem, syn, rel): return z
            model.arg = IdentityARG().to(dev)
        elif variant == "wo_kan":
            d = cfg.hidden_size
            class ConcatMLPFusion(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.mlp = nn.Sequential(
                        nn.Linear(d * 3, d),
                        nn.GELU(),
                        nn.Dropout(cfg.dropout)
                    )
                    self.norm = nn.LayerNorm(d)
                def forward(self, sem, syn, rel):
                    z = self.mlp(torch.cat([sem, syn, rel], dim=-1))
                    return self.norm(z)
            model.fusion = ConcatMLPFusion(d).to(dev)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        target_has_labels = k_shot is not None and k_shot > 0
        sds = DS(DOMAIN_FILES[src], cfg, kg, domain_id=0, use_labels=True)
        tds_train = DS(DOMAIN_FILES[tgt], cfg, kg, domain_id=1, use_labels=target_has_labels)
        if target_has_labels:
            tds_train = few_shot_sample(tds_train, k_shot)
        tds_eval = DS(DOMAIN_FILES[tgt], cfg, kg, domain_id=1, use_labels=True)

        train_batch_size = max(1, cfg.batch_size // 2)
        sl = DataLoader(sds, batch_size=train_batch_size, shuffle=True, collate_fn=cf)
        tl_train = DataLoader(tds_train, batch_size=train_batch_size, shuffle=True, collate_fn=cf)
        tl = DataLoader(tds_eval, batch_size=cfg.batch_size, collate_fn=cf)

        def pad_kg_width(batch, width):
            batch = dict(batch)
            cur = batch["kg_heads"].size(1)
            if cur == width:
                return batch
            pad_k = width - cur
            for key in ["kg_heads", "kg_rels", "kg_tails", "kg_mask", "kg_aspect_ids"]:
                x = batch[key]
                pad = torch.zeros(x.size(0), pad_k, dtype=x.dtype)
                batch[key] = torch.cat([x, pad], dim=1)
            m = batch["kg_token_map"]
            pad = torch.zeros(m.size(0), m.size(1), pad_k, dtype=m.dtype)
            batch["kg_token_map"] = torch.cat([m, pad], dim=2)
            return batch

        def merge_batches(src_batch, tgt_batch):
            width = max(src_batch["kg_heads"].size(1), tgt_batch["kg_heads"].size(1))
            src_batch = pad_kg_width(src_batch, width)
            tgt_batch = pad_kg_width(tgt_batch, width)
            merged = {}
            for key, value in src_batch.items():
                if isinstance(value, torch.Tensor):
                    merged[key] = torch.cat([value, tgt_batch[key]], dim=0)
            merged["aspect_spans"] = src_batch["aspect_spans"] + tgt_batch["aspect_spans"]
            return merged

        amp_enabled = dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled); best = 0
        for ep in range(cfg.epochs):
            model.train()
            tgt_iter = cycle(tl_train)
            for i, src_batch in enumerate(sl):
                tgt_batch = next(tgt_iter)
                batch = merge_batches(src_batch, tgt_batch)
                batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    out = model(batch); loss = out["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                if i % 10 == 0: print(f"    Batch {i} Loss: {loss.item():.4f}")
            model.eval(); ap, al = [], []
            with torch.no_grad():
                for b in tl:
                    b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                    with torch.amp.autocast("cuda", enabled=amp_enabled): o = model(b)
                    ap.append(o["logits"].cpu()); al.append(b["labels"].cpu())
            f1 = compute_macro_f1(torch.cat(ap), torch.cat(al))
            if f1 > best: best = f1
            print(f"  Ep {ep+1}: F1={f1:.2f} best={best:.2f}")
        save_result(display, setting, src, tgt, best)
        del model; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"[ERR] {display} {src}->{tgt}: {e}"); traceback.print_exc()

# ═══════════════════════════════════════════
# DISPATCHER
# ═══════════════════════════════════════════
TRAINERS = {
    "bert_uda": train_bert_uda, "ahf": train_ahf,
    "transproto": train_transproto, "bgca": train_bgca,
    "ketgm": train_ketgm, "dalm": train_dalm,
    "llmsynabsa": train_llmsynabsa, "kgan": train_kgan,
    "senticgcn": train_senticgcn, "llm_kgkan": train_llm_kgkan,
}

def train_model(name, src, tgt, setting="standard", k_shot=None, **kw):
    t = TRAINERS.get(name)
    if not t: print(f"[WARN] No trainer for {name}"); return
    t(src, tgt, setting=setting, k_shot=k_shot, **kw)

def train_all_models_on_pairs(models, pairs, setting="standard", k_shot=None, **kw):
    for mn in models:
        for s, t in pairs:
            print(f"\n{'='*50}\n{mn}: {s} -> {t} ({setting})\n{'='*50}")
            train_model(mn, s, t, setting, k_shot, **kw)
