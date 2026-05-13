"""
scripts/config.py — Centralized configuration for ALL models and experiments.

This file contains every tunable parameter for the full paper reproduction.
GPU profiles, PEFT settings, transfer pairs, API budgets, etc.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR          = os.path.join(PROJECT_ROOT, "data")
KG_DIR            = os.path.join(PROJECT_ROOT, "knowledge_graphs")
GITCLONES_DIR     = os.path.join(PROJECT_ROOT, "gitclones")
SAVED_MODELS_DIR  = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR       = os.path.join(PROJECT_ROOT, "results")
SCRIPTS_DIR       = os.path.join(PROJECT_ROOT, "scripts")

# Create dirs
for d in [SAVED_MODELS_DIR, RESULTS_DIR,
          os.path.join(RESULTS_DIR, "api_cache"),
          os.path.join(RESULTS_DIR, "standard"),
          os.path.join(RESULTS_DIR, "fewshot"),
          os.path.join(RESULTS_DIR, "zeroshot"),
          os.path.join(RESULTS_DIR, "ablation"),
          os.path.join(RESULTS_DIR, "training_logs")]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# ENV / API KEYS
# ══════════════════════════════════════════════════════════════════════════

ENV_FILE = os.path.join(PROJECT_ROOT, ".env.local")

def load_env():
    """Load .env.local into os.environ."""
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

load_env()

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")   # placeholder
GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY", "")      # placeholder

# ══════════════════════════════════════════════════════════════════════════
# GPU PROFILES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class GPUProfile:
    name: str
    vram_gb: int
    tflops: int
    batch_size_llm: int        # for 7-8B LLM models
    batch_size_bert: int       # for BERT-based models
    batch_size_small: int      # for LSTM/GCN models
    gradient_accumulation: int
    max_seq_len: int

# ── Active profile ──
RTX_P6000 = GPUProfile(
    name="RTX_P6000",
    vram_gb=96,
    tflops=500,
    batch_size_llm=16,
    batch_size_bert=32,
    batch_size_small=64,
    gradient_accumulation=1,
    max_seq_len=128,
)

# # ── H200 profile (commented out) ──
# H200 = GPUProfile(
#     name="H200",
#     vram_gb=141,
#     tflops=1979,
#     batch_size_llm=32,
#     batch_size_bert=64,
#     batch_size_small=128,
#     gradient_accumulation=1,
#     max_seq_len=128,
# )

GPU = RTX_P6000  # <-- change to H200 if using H200

# ══════════════════════════════════════════════════════════════════════════
# GLOBAL TRAINING SETTINGS
# ══════════════════════════════════════════════════════════════════════════

SEED = 42
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
PIN_MEMORY = True

# ══════════════════════════════════════════════════════════════════════════
# UNIFIED PEFT / QLORA SETTINGS (used for ALL models that support it)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PEFTConfig:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    # QLoRA
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

PEFT = PEFTConfig()

# ══════════════════════════════════════════════════════════════════════════
# LABEL SCHEMA (BIO for ABSA)
# ══════════════════════════════════════════════════════════════════════════

LABELS = ["O", "B-POS", "I-POS", "B-NEG", "I-NEG", "B-NEU", "I-NEU"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)
IGNORE_INDEX = -100

# Sentiment mapping from CSV
SENTIMENT_MAP = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}

# ══════════════════════════════════════════════════════════════════════════
# DOMAINS & DATA FILES
# ══════════════════════════════════════════════════════════════════════════

DOMAIN_FILES = {
    "L":  os.path.join(DATA_DIR, "laptop.csv"),
    "R":  os.path.join(DATA_DIR, "restaurant.csv"),
    "D":  os.path.join(DATA_DIR, "device.csv"),
    "S":  os.path.join(DATA_DIR, "service.csv"),
    "A":  os.path.join(DATA_DIR, "airline.csv"),
    "SH": os.path.join(DATA_DIR, "shoes.csv"),
    "W":  os.path.join(DATA_DIR, "water_purifier.csv"),
    "U":  os.path.join(DATA_DIR, "university_course.csv"),
    "H":  os.path.join(DATA_DIR, "healthcare.csv"),
}

SOURCE_DOMAINS = ["L", "R", "D", "S"]
LOW_RESOURCE_TARGETS = ["A", "SH", "W"]    # few-shot
ZERO_SHOT_TARGETS = ["U", "H"]             # zero-shot (no train labels)

# ── Standard cross-domain transfer pairs (Table 1) ──
# All src→tgt except L↔D
STANDARD_PAIRS = [
    ("L", "R"), ("L", "S"),
    ("R", "L"), ("R", "D"), ("R", "S"),
    ("D", "R"), ("D", "S"),
    ("S", "L"), ("S", "R"), ("S", "D"),
]

# ── Few-shot pairs (Table 6): {L,R,D,S} → {A,SH,W} ──
FEWSHOT_PAIRS = [
    (src, tgt) for src in SOURCE_DOMAINS for tgt in LOW_RESOURCE_TARGETS
]

# ── Zero-shot pairs (Table 7): {L,R,D,S} → {U,H} ──
ZEROSHOT_PAIRS = [
    (src, tgt) for src in SOURCE_DOMAINS for tgt in ZERO_SHOT_TARGETS
]

# ── Few-shot k values (Table 9) ──
FEWSHOT_K_VALUES = [2, 4, 8, 16]
DEFAULT_FEWSHOT_K = 16   # used for Table 3, 6

# ══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH PATHS
# ══════════════════════════════════════════════════════════════════════════

CONCEPTNET_GZ  = os.path.join(KG_DIR, "conceptnet-assertions-5.7.0.csv.gz")
CONCEPTNET_CSV = os.path.join(KG_DIR, "conceptnet-assertions-5.7.0.csv")
CONCEPTNET_PKL = os.path.join(KG_DIR, "conceptnet_en.pkl")

SENTICNET_TXT  = os.path.join(KG_DIR, "senticnet", "senticnet.txt")
SENTICNET_PKL  = os.path.join(KG_DIR, "senticnet_processed.pkl")

WORDNET_DIR    = os.path.join(KG_DIR, "wn3.1.dict", "dict")
WORDNET_PKL    = os.path.join(KG_DIR, "wordnet_processed.pkl")

HYBRID_KG_PKL  = os.path.join(KG_DIR, "hybrid_kg.pkl")

KG_MAX_EDGES   = 3_000_000   # max ConceptNet edges to load
KG_MAX_TRIPLES = 16          # per aspect subgraph
KG_HOPS        = 2

# ══════════════════════════════════════════════════════════════════════════
# MODEL-SPECIFIC CONFIGS
# ══════════════════════════════════════════════════════════════════════════

# ── LLM-KGKAN (paper Table 4) ──
@dataclass
class LLMKGKANModelConfig:
    llm_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    hidden_size: int = 4096
    batch_size: int = GPU.batch_size_llm
    epochs: int = 10
    lr: float = 2e-4
    weight_decay: float = 1e-2
    max_len: int = GPU.max_seq_len
    num_labels: int = NUM_LABELS
    num_dep_relations: int = 40
    num_entities: int = 10000       # updated after KG load
    num_kg_relations: int = 500     # updated after KG load
    kg_emb_dim: int = 128
    rgcn_layers: int = 2
    dropout: float = 0.1
    lora_r: int = PEFT.lora_r
    lora_alpha: int = PEFT.lora_alpha
    lora_dropout: float = PEFT.lora_dropout
    mmd_lambda: float = 0.2
    ignore_index: int = IGNORE_INDEX
    freeze_backbone: bool = True
    use_distmult: bool = False
    prefix_len: int = 10
    seed: int = SEED
    patience: int = 3              # early stopping
    LABEL2ID: dict = field(default_factory=lambda: LABEL2ID)

# ── BERT-UDA ──
@dataclass
class BERTUDAConfig:
    model_name: str = "bert-base-uncased"
    hidden_size: int = 768
    batch_size: int = GPU.batch_size_bert
    epochs: int = 15
    lr: float = 2e-5
    max_len: int = GPU.max_seq_len
    dropout: float = 0.1
    seed: int = SEED

# ── AHF ──
@dataclass
class AHFConfig:
    word_dim: int = 100
    pos_dim: int = 15
    hidden: int = 100
    num_layers: int = 2
    dropout: float = 0.5
    batch_size: int = GPU.batch_size_small
    lr: float = 0.001
    epochs: int = 25
    eta: float = 1.0
    lambda_adv: float = 0.1
    gamma: float = 0.98
    max_len: int = GPU.max_seq_len
    seed: int = SEED

# ── TransProto ──
@dataclass
class TransProtoConfig:
    model_name: str = "bert-base-uncased"
    batch_size: int = 16
    lr: float = 3e-5
    epochs: int = 15
    max_len: int = 120
    class_num: int = 3
    hidden_dim: int = 400
    seed: int = SEED

# ── BGCA ──
@dataclass
class BGCAConfig:
    model_name: str = "facebook/bart-base"
    batch_size: int = GPU.batch_size_bert
    lr: float = 3e-5
    epochs: int = 20
    max_len: int = GPU.max_seq_len
    seed: int = SEED

# ── KETGM ──
@dataclass
class KETGMConfig:
    bert_model: str = "bert-base-uncased"
    bert_hidden: int = 768
    rgcn_hidden: int = 200
    rgcn_num_layers: int = 2
    rgcn_num_bases: int = 30
    num_topics: int = 6
    words_per_topic: int = 10
    batch_size: int = GPU.batch_size_bert
    epochs: int = 30
    lr: float = 2e-5
    mu: float = 1.0
    classifier_dropout: float = 0.8
    max_len: int = GPU.max_seq_len
    seed: int = SEED

# ── DALM ──
@dataclass
class DALMConfig:
    model_name: str = "gpt2"
    batch_size: int = GPU.batch_size_bert
    lr: float = 5e-5
    epochs: int = 20
    max_len: int = GPU.max_seq_len
    seed: int = SEED

# ── LLMSynABSA ──
@dataclass
class LLMSynABSAConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_len: int = 96
    epochs: int = 100
    epochs_syntax: int = 50
    lr: float = 5e-5
    lr_syntax: float = 1e-4
    dropout: float = 0.2
    rho: float = 0.5
    batch_size: int = GPU.batch_size_llm
    num_prompts: int = 10
    lora_r: int = PEFT.lora_r
    lora_alpha: int = PEFT.lora_alpha
    lora_dropout: float = PEFT.lora_dropout
    seed: int = SEED

# ── KGAN ──
@dataclass
class KGANConfig:
    model: str = "KGNN"
    dim_w: int = 768
    dim_k: int = 200
    batch_size: int = GPU.batch_size_bert
    lr: float = 3e-5
    epochs: int = 20
    dropout: float = 0.5
    is_bert: int = 2     # use RoBERTa-style
    gcn: int = 0
    kge: str = "distmult"
    max_len: int = GPU.max_seq_len
    seed: int = SEED

# ── SenticGCN ──
@dataclass
class SenticGCNConfig:
    model_name: str = "senticgcn"
    embed_dim: int = 300
    hidden_dim: int = 300
    batch_size: int = 16
    lr: float = 0.001
    l2reg: float = 1e-5
    epochs: int = 100
    polarities_dim: int = 3
    seed: int = SEED

# ══════════════════════════════════════════════════════════════════════════
# LLM API SETTINGS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class APIBudgetConfig:
    """API cost control. Adjust tier to control max spend."""
    # ── $20 tier (active) ──
    max_budget_usd: float = 20.0
    batch_size: int = 10             # concurrent requests
    requests_per_minute: int = 200   # rate limit
    max_retries: int = 3
    retry_delay: float = 2.0

    # # ── $50 tier ──
    # max_budget_usd: float = 50.0
    # batch_size: int = 20
    # requests_per_minute: int = 500
    # max_retries: int = 3
    # retry_delay: float = 1.0

    # # ── $100 tier ──
    # max_budget_usd: float = 100.0
    # batch_size: int = 50
    # requests_per_minute: int = 1000
    # max_retries: int = 5
    # retry_delay: float = 0.5

API_BUDGET = APIBudgetConfig()

# LLM models for inference (Tables 1, 3, 6, 7)
API_MODELS = {
    "gpt-4o":         {"provider": "openai", "model": "gpt-4o",         "key_env": "OPENAI_API_KEY"},
    "gpt-4-turbo":    {"provider": "openai", "model": "gpt-4-turbo",   "key_env": "OPENAI_API_KEY"},
    "gpt-4.1":        {"provider": "openai", "model": "gpt-4.1",       "key_env": "OPENAI_API_KEY"},
    "gpt-4.1-mini":   {"provider": "openai", "model": "gpt-4.1-mini",  "key_env": "OPENAI_API_KEY"},
    "claude-sonnet":  {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "key_env": "ANTHROPIC_API_KEY"},
    "gemini-2.5-pro": {"provider": "google", "model": "gemini-2.5-pro", "key_env": "GOOGLE_API_KEY"},
}

# Open-source LLMs (inference via HF / vLLM)
OPENSOURCE_LLMS = {
    "llama-3.1-8b-instruct":  "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5-7b-instruct":    "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct":   "Qwen/Qwen2.5-14B-Instruct",
    # Qwen2.5-72B: inference-only, loaded in 4-bit
    # "qwen2.5-72b-instruct":  "Qwen/Qwen2.5-72B-Instruct",
}

# ══════════════════════════════════════════════════════════════════════════
# ABLATION SETTINGS (Tables 12, 13)
# ══════════════════════════════════════════════════════════════════════════

FUSION_METHODS = [
    "concat_mlp",
    "weighted_sum",
    "gated_fusion",
    "bilinear_fusion",
    "cross_attention",
    "kan_fusion",       # default / paper
]

KG_SOURCES = [
    "none",             # no KG
    "wordnet",
    "senticnet",
    "conceptnet",
    "hybrid",           # default / paper
]

# ══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATE FOR LLM BIO TAGGING
# ══════════════════════════════════════════════════════════════════════════

BIO_PROMPT_TEMPLATE = """You are an expert in aspect-based sentiment analysis. Given a sentence, identify aspect terms and their sentiment polarity using BIO tagging.

Tags: O (non-aspect), B-POS (begin positive aspect), I-POS (inside positive aspect), B-NEG (begin negative aspect), I-NEG (inside negative aspect), B-NEU (begin neutral aspect), I-NEU (inside neutral aspect).

Sentence: {sentence}

Output ONLY the BIO tags separated by spaces, one tag per token. The number of tags must exactly match the number of tokens.
Tags:"""

# ══════════════════════════════════════════════════════════════════════════
# CHECKPOINT NAMING
# ══════════════════════════════════════════════════════════════════════════

def get_checkpoint_path(model_name: str, src: str, tgt: str,
                        variant: str = "full") -> str:
    """Return standardized checkpoint path."""
    d = os.path.join(SAVED_MODELS_DIR, model_name, variant)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{src}_to_{tgt}.pt")

def get_result_path(model_name: str, setting: str,
                    src: str, tgt: str) -> str:
    """Return standardized result JSON path."""
    d = os.path.join(RESULTS_DIR, setting)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{model_name}_{src}_to_{tgt}.json")

def get_training_log_path(model_name: str, src: str, tgt: str,
                          variant: str = "full") -> str:
    """Return standardized training log path."""
    d = os.path.join(RESULTS_DIR, "training_logs", model_name, variant)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{src}_to_{tgt}.json")

# ══════════════════════════════════════════════════════════════════════════
# TABLE / FIGURE REFERENCE NUMBERS
# ══════════════════════════════════════════════════════════════════════════

# Models that appear in each table
TABLE1_MODELS = [
    "bert_uda", "ahf", "transproto", "bgca", "ketgm", "dalm",
    "llmsynabsa",
    "kgan", "senticgcn",  # new models added
    "gpt-4-turbo", "gpt-4o",
    "llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "qwen2.5-14b-instruct",
    "llm_kgkan",
]

TABLE3_MODELS = TABLE1_MODELS + [
    "gpt-4.1", "claude-sonnet", "gemini-2.5-pro",
]

# Display names for tables
MODEL_DISPLAY_NAMES = {
    "bert_uda":       "BERT-UDA",
    "ahf":            "AHF",
    "transproto":     "TransProto",
    "bgca":           "BGCA",
    "ketgm":          "KETGM",
    "dalm":           "DALM",
    "llmsynabsa":     "LLM-Augment-Syn-DA",
    "kgan":           "KGAN",
    "senticgcn":      "SenticGCN",
    "gpt-4-turbo":    "GPT-4 Turbo",
    "gpt-4o":         "GPT-4o",
    "gpt-4.1":        "GPT-4.1",
    "claude-sonnet":  "Claude Sonnet",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "llama-3.1-8b-instruct":  "LLaMA-3.1-8B-Instruct",
    "qwen2.5-7b-instruct":    "Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct":   "Qwen2.5-14B-Instruct",
    "llm_kgkan":      "LLM-KGKAN",
    "llm_kgkan_wo_kg":  "w/o KG",
    "llm_kgkan_wo_syn": "w/o Syn",
    "llm_kgkan_wo_arg": "w/o ARG",
    "llm_kgkan_wo_kan": "w/o KAN",
}

print(f"[config] Project root: {PROJECT_ROOT}")
print(f"[config] GPU profile: {GPU.name} ({GPU.vram_gb}GB, {GPU.tflops} TFLOPS)")
print(f"[config] Device: {DEVICE}, dtype: {DTYPE}")
print(f"[config] Seed: {SEED}")
print(f"[config] PEFT: LoRA r={PEFT.lora_r}, alpha={PEFT.lora_alpha}, 4bit={PEFT.load_in_4bit}")
print(f"[config] API budget: ${API_BUDGET.max_budget_usd}")
print(f"[config] HF_TOKEN: {'set' if HF_TOKEN else 'NOT SET'}")
print(f"[config] OPENAI_API_KEY: {'set' if OPENAI_API_KEY else 'NOT SET'}")
