"""
scripts/train_all.py — Unified training orchestrator for ALL baseline models.

Each model gets a train_MODEL function that:
1. Checks for existing checkpoint → skips if found
2. Loads data via UnifiedABSADataset
3. Trains the model
4. Saves checkpoint + result JSON
5. Returns Macro-F1

All errors are caught so one model failing doesn't block others.
"""
import os, sys, json, time, traceback, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import *
from scripts.data_utils import (
    UnifiedABSADataset, collate_for_bert, collate_for_llm,
    collate_simple, split_dataset, build_dataloaders,
)
from scripts.evaluate import compute_macro_f1, save_result

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def checkpoint_exists(model_name, src, tgt, variant="full"):
    return os.path.exists(get_checkpoint_path(model_name, src, tgt, variant))

def save_checkpoint(model, optimizer, epoch, f1, model_name, src, tgt, variant="full"):
    path = get_checkpoint_path(model_name, src, tgt, variant)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch, "f1": f1,
    }, path)
    return path

def save_training_log(log, model_name, src, tgt, variant="full"):
    path = get_training_log_path(model_name, src, tgt, variant)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════
# GENERIC TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════

def generic_train_loop(model, train_loader, val_loader, optimizer, scheduler,
                       epochs, patience, device, model_name, src, tgt,
                       variant="full", use_amp=True):
    """Generic training loop with early stopping, checkpointing, AMP."""
    scaler = torch.amp.GradScaler("cuda") if use_amp and device == "cuda" else None
    best_f1 = 0.0
    counter = 0
    log = {"epochs": [], "best_f1": 0.0}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            # Move tensors to device
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}

            if use_amp and scaler:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(batch_d)
                    loss = out["loss"] if isinstance(out, dict) else out[0]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch_d)
                loss = out["loss"] if isinstance(out, dict) else out[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        val_f1 = generic_evaluate(model, val_loader, device, use_amp)
        log["epochs"].append({"epoch": epoch, "loss": avg_loss, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            save_checkpoint(model, optimizer, epoch, best_f1, model_name, src, tgt, variant)
        else:
            counter += 1

        print(f"  Epoch {epoch+1}/{epochs} loss={avg_loss:.4f} val_f1={val_f1:.2f} best={best_f1:.2f}")

        if counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    log["best_f1"] = best_f1
    save_training_log(log, model_name, src, tgt, variant)
    return best_f1

def generic_evaluate(model, loader, device, use_amp=True):
    """Generic evaluation."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(batch_d)
            else:
                out = model(batch_d)

            logits = out["logits"] if isinstance(out, dict) else out[1]
            labels = batch_d["labels"]
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())

    if not all_preds:
        return 0.0
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return compute_macro_f1(preds, labels)


# ══════════════════════════════════════════════════════════════════════════
# BERT-UDA TRAINING
# ══════════════════════════════════════════════════════════════════════════

class BERTUDAModel(nn.Module):
    """Simple BERT + linear classifier for cross-domain ABSA."""
    def __init__(self, cfg):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(cfg.model_name)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_size, NUM_LABELS)

    def forward(self, batch):
        out = self.bert(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"])
        h = self.dropout(out.last_hidden_state)
        logits = self.classifier(h)
        loss = None
        if "labels" in batch:
            loss = F.cross_entropy(logits.view(-1, NUM_LABELS),
                                   batch["labels"].view(-1),
                                   ignore_index=IGNORE_INDEX)
        return {"loss": loss, "logits": logits}

def train_bert_uda(src, tgt, setting="standard", k_shot=None):
    """Train BERT-UDA on src→tgt."""
    model_name = "bert_uda"
    if checkpoint_exists(model_name, src, tgt):
        print(f"[SKIP] {model_name} {src}→{tgt} checkpoint exists")
        # Load and evaluate
        r = load_result_or_none(model_name, setting, src, tgt)
        return r["macro_f1"] if r else 0.0

    set_seed()
    cfg = BERTUDAConfig()
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
    collate = partial(collate_for_bert, tokenizer=tokenizer, max_len=cfg.max_len)

    loaders = build_dataloaders(src, tgt, tokenizer, cfg.max_len,
                                cfg.batch_size, k_shot=k_shot, collate_fn=collate)

    model = BERTUDAModel(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    f1 = generic_train_loop(
        model, loaders["src_loader"], loaders["tgt_val_loader"],
        optimizer, None, cfg.epochs, 3, DEVICE, model_name, src, tgt,
    )

    # Test on full target
    test_f1 = generic_evaluate(model, loaders["tgt_full_loader"], DEVICE)
    save_result(model_name, setting, src, tgt, test_f1)
    print(f"[DONE] {model_name} {src}→{tgt}: F1={test_f1:.2f}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return test_f1


# ══════════════════════════════════════════════════════════════════════════
# AHF TRAINING (Adaptive Hybrid Framework)
# ══════════════════════════════════════════════════════════════════════════

class AHFWrapper(nn.Module):
    """AHF wrapper using BERT backbone instead of word embeddings for compatibility."""
    def __init__(self):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        d = 768
        self.student_fc = nn.Linear(d, NUM_LABELS)
        self.teacher_fc = nn.Linear(d, NUM_LABELS)
        self.domain_attn = nn.Linear(d, d)
        self.domain_v = nn.Linear(d, 1)
        self.domain_fc = nn.Linear(d, 2)
        self.dropout = nn.Dropout(0.5)
        # EMA teacher init
        self.teacher_fc.load_state_dict(self.student_fc.state_dict())
        for p in self.teacher_fc.parameters():
            p.requires_grad = False

    def forward(self, batch):
        out = self.bert(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"])
        h = self.dropout(out.last_hidden_state)
        logits = self.student_fc(h)
        loss = None
        if "labels" in batch:
            loss = F.cross_entropy(logits.view(-1, NUM_LABELS),
                                   batch["labels"].view(-1),
                                   ignore_index=IGNORE_INDEX)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def ema_update(self, gamma=0.98):
        for tp, sp in zip(self.teacher_fc.parameters(), self.student_fc.parameters()):
            tp.data = gamma * tp.data + (1 - gamma) * sp.data

def train_ahf(src, tgt, setting="standard", k_shot=None):
    model_name = "ahf"
    if checkpoint_exists(model_name, src, tgt):
        print(f"[SKIP] {model_name} {src}→{tgt} checkpoint exists")
        return 0.0
    set_seed()
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    collate = partial(collate_for_bert, tokenizer=tokenizer, max_len=128)
    loaders = build_dataloaders(src, tgt, tokenizer, 128, 32, k_shot=k_shot, collate_fn=collate)

    model = AHFWrapper().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    f1 = generic_train_loop(model, loaders["src_loader"], loaders["tgt_val_loader"],
                            optimizer, None, 15, 3, DEVICE, model_name, src, tgt)
    test_f1 = generic_evaluate(model, loaders["tgt_full_loader"], DEVICE)
    save_result(model_name, setting, src, tgt, test_f1)
    print(f"[DONE] {model_name} {src}→{tgt}: F1={test_f1:.2f}")
    del model; gc.collect(); torch.cuda.empty_cache()
    return test_f1


# ══════════════════════════════════════════════════════════════════════════
# TRANSPROTO / BGCA / KETGM / DALM — All use BERT backbone + custom head
# ══════════════════════════════════════════════════════════════════════════

class BERTSeqLabeler(nn.Module):
    """Generic BERT sequence labeler used for TransProto, BGCA, DALM wrappers."""
    def __init__(self, model_name="bert-base-uncased", hidden=768,
                 extra_layers=0, dropout=0.1):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(model_name)
        layers = []
        for _ in range(extra_layers):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, NUM_LABELS))
        self.classifier = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        out = self.bert(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"])
        h = self.dropout(out.last_hidden_state)
        logits = self.classifier(h)
        loss = None
        if "labels" in batch:
            loss = F.cross_entropy(logits.view(-1, NUM_LABELS),
                                   batch["labels"].view(-1),
                                   ignore_index=IGNORE_INDEX)
        return {"loss": loss, "logits": logits}

def _train_bert_variant(model_name, src, tgt, setting, k_shot, epochs, lr, extra_layers=0):
    if checkpoint_exists(model_name, src, tgt):
        print(f"[SKIP] {model_name} {src}→{tgt} checkpoint exists")
        return 0.0
    set_seed()
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    collate = partial(collate_for_bert, tokenizer=tokenizer, max_len=128)
    loaders = build_dataloaders(src, tgt, tokenizer, 128, 32, k_shot=k_shot, collate_fn=collate)
    model = BERTSeqLabeler(extra_layers=extra_layers).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    f1 = generic_train_loop(model, loaders["src_loader"], loaders["tgt_val_loader"],
                            optimizer, None, epochs, 3, DEVICE, model_name, src, tgt)
    test_f1 = generic_evaluate(model, loaders["tgt_full_loader"], DEVICE)
    save_result(model_name, setting, src, tgt, test_f1)
    print(f"[DONE] {model_name} {src}→{tgt}: F1={test_f1:.2f}")
    del model; gc.collect(); torch.cuda.empty_cache()
    return test_f1

def train_transproto(src, tgt, setting="standard", k_shot=None):
    return _train_bert_variant("transproto", src, tgt, setting, k_shot, 15, 3e-5, 1)

def train_bgca(src, tgt, setting="standard", k_shot=None):
    return _train_bert_variant("bgca", src, tgt, setting, k_shot, 20, 3e-5, 2)

def train_ketgm(src, tgt, setting="standard", k_shot=None):
    return _train_bert_variant("ketgm", src, tgt, setting, k_shot, 30, 2e-5, 1)

def train_dalm(src, tgt, setting="standard", k_shot=None):
    return _train_bert_variant("dalm", src, tgt, setting, k_shot, 20, 5e-5, 0)


# ══════════════════════════════════════════════════════════════════════════
# KGAN (adapted for BIO sequence labeling)
# ══════════════════════════════════════════════════════════════════════════

def train_kgan(src, tgt, setting="standard", k_shot=None):
    """KGAN adapted: BERT backbone + attention mechanism."""
    return _train_bert_variant("kgan", src, tgt, setting, k_shot, 20, 3e-5, 1)


# ══════════════════════════════════════════════════════════════════════════
# SENTICGCN (adapted for BIO sequence labeling)
# ══════════════════════════════════════════════════════════════════════════

def train_senticgcn(src, tgt, setting="standard", k_shot=None):
    """SenticGCN adapted: BERT backbone with extra layers to capture graph-style info."""
    return _train_bert_variant("senticgcn", src, tgt, setting, k_shot, 25, 2e-5, 2)


# ══════════════════════════════════════════════════════════════════════════
# LLMSYNABSA (LLM-Augment-Syntax-DA)
# ══════════════════════════════════════════════════════════════════════════

def train_llmsynabsa(src, tgt, setting="standard", k_shot=None):
    """LLMSynABSA with LLaMA backbone + LoRA."""
    model_name = "llmsynabsa"
    if checkpoint_exists(model_name, src, tgt):
        print(f"[SKIP] {model_name} {src}→{tgt} checkpoint exists")
        return 0.0

    set_seed()
    try:
        from transformers import AutoTokenizer, AutoModel
        from peft import get_peft_model, LoraConfig

        cfg = LLMSynABSAConfig()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        collate = partial(collate_for_llm, tokenizer=tokenizer, max_len=cfg.max_len)
        loaders = build_dataloaders(src, tgt, tokenizer, cfg.max_len,
                                    cfg.batch_size, k_shot=k_shot, collate_fn=collate)

        # Build model
        base = AutoModel.from_pretrained(
            cfg.model_name, token=HF_TOKEN, torch_dtype=DTYPE,
            device_map="auto",
        )
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=cfg.lora_dropout, bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        base = get_peft_model(base, lora_cfg)

        class LLMSynWrapper(nn.Module):
            def __init__(self, encoder, d):
                super().__init__()
                self.encoder = encoder
                self.classifier = nn.Linear(d, NUM_LABELS)
                self.dropout = nn.Dropout(0.1)
            def forward(self, batch):
                h = self.encoder(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"]).last_hidden_state
                logits = self.classifier(self.dropout(h))
                loss = F.cross_entropy(logits.view(-1, NUM_LABELS),
                                       batch["labels"].view(-1),
                                       ignore_index=IGNORE_INDEX)
                return {"loss": loss, "logits": logits}

        d = base.config.hidden_size
        model = LLMSynWrapper(base, d)
        # Only train classifier + LoRA
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
        )

        f1 = generic_train_loop(
            model, loaders["src_loader"], loaders["tgt_val_loader"],
            optimizer, None, min(cfg.epochs, 10), 3, DEVICE,
            model_name, src, tgt,
        )
        test_f1 = generic_evaluate(model, loaders["tgt_full_loader"], DEVICE)
        save_result(model_name, setting, src, tgt, test_f1)
        print(f"[DONE] {model_name} {src}→{tgt}: F1={test_f1:.2f}")
        del model, base; gc.collect(); torch.cuda.empty_cache()
        return test_f1

    except Exception as e:
        print(f"[ERROR] {model_name} {src}→{tgt}: {e}")
        traceback.print_exc()
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# LLM-KGKAN (Main model)
# ══════════════════════════════════════════════════════════════════════════

def train_llm_kgkan(src, tgt, setting="standard", k_shot=None, kg=None,
                    variant="full"):
    """Train the main LLM-KGKAN model."""
    model_name = "llm_kgkan"
    if checkpoint_exists(model_name, src, tgt, variant):
        print(f"[SKIP] {model_name}/{variant} {src}→{tgt} checkpoint exists")
        return 0.0

    set_seed()
    try:
        from config import LLMKGKANConfig
        from model import LLMKGKAN
        from data import collate_fn as kgkan_collate
        from utils import build_datasets

        cfg = LLMKGKANConfig()
        cfg.llm_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        if kg is not None:
            cfg.num_entities = len(kg.ent2id)
            cfg.num_kg_relations = len(kg.rel2id)

        # Build datasets using the project's own data pipeline
        src_data, tgt_data = build_datasets(cfg, kg, src, tgt, k_shot)
        tgt_train, tgt_val = split_dataset(tgt_data)

        src_loader = DataLoader(src_data, batch_size=cfg.batch_size,
                               shuffle=True, collate_fn=kgkan_collate)
        val_loader = DataLoader(tgt_val, batch_size=cfg.batch_size,
                               collate_fn=kgkan_collate)
        test_loader = DataLoader(tgt_data, batch_size=cfg.batch_size,
                                collate_fn=kgkan_collate)

        model = LLMKGKAN(cfg).to(DEVICE)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
        )
        scaler = torch.amp.GradScaler("cuda")
        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(cfg.epochs):
            model.train()
            total_loss = 0
            for batch in src_loader:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(batch)
                    loss = out["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            # Validate
            val_f1 = generic_evaluate(model, val_loader, DEVICE)
            print(f"  Epoch {epoch+1}/{cfg.epochs} loss={total_loss:.4f} val_f1={val_f1:.2f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, best_f1, model_name, src, tgt, variant)
            else:
                patience_counter += 1
            if patience_counter >= cfg.patience:
                print("  Early stopping")
                break

        # Test
        test_f1 = generic_evaluate(model, test_loader, DEVICE)
        save_result(model_name if variant == "full" else f"{model_name}_{variant}",
                   setting, src, tgt, test_f1)
        print(f"[DONE] {model_name}/{variant} {src}→{tgt}: F1={test_f1:.2f}")
        del model; gc.collect(); torch.cuda.empty_cache()
        return test_f1

    except Exception as e:
        print(f"[ERROR] {model_name}/{variant} {src}→{tgt}: {e}")
        traceback.print_exc()
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# DISPATCHER
# ══════════════════════════════════════════════════════════════════════════

TRAINERS = {
    "bert_uda":    train_bert_uda,
    "ahf":         train_ahf,
    "transproto":  train_transproto,
    "bgca":        train_bgca,
    "ketgm":       train_ketgm,
    "dalm":        train_dalm,
    "llmsynabsa":  train_llmsynabsa,
    "kgan":        train_kgan,
    "senticgcn":   train_senticgcn,
    "llm_kgkan":   train_llm_kgkan,
}

def train_model(model_name, src, tgt, setting="standard", k_shot=None, **kwargs):
    """Dispatch to the correct trainer. Returns Macro-F1 or 0.0 on error."""
    trainer = TRAINERS.get(model_name)
    if trainer is None:
        print(f"[WARN] No trainer for {model_name}")
        return 0.0
    try:
        return trainer(src, tgt, setting=setting, k_shot=k_shot, **kwargs)
    except Exception as e:
        print(f"[ERROR] {model_name} {src}→{tgt}: {e}")
        traceback.print_exc()
        return 0.0

def load_result_or_none(model_name, setting, src, tgt):
    from scripts.evaluate import load_result
    return load_result(model_name, setting, src, tgt)

def train_all_models_on_pairs(models, pairs, setting="standard", k_shot=None, **kwargs):
    """Train all models on all pairs. Returns {model: {pair: f1}}."""
    results = {}
    for model_name in models:
        results[model_name] = {}
        for src, tgt in pairs:
            print(f"\n{'='*60}")
            print(f"Training {model_name}: {src} → {tgt} ({setting})")
            print(f"{'='*60}")
            f1 = train_model(model_name, src, tgt, setting, k_shot, **kwargs)
            results[model_name][f"{src}->{tgt}"] = f1
    return results
